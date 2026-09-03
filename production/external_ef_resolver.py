"""External A1-A3 GWP-factor resolver for ELCD-unmatched BOM materials.

This module is deliberately separate from Qwen material/process matching and runs
only for rows that were not approved for an ELCD/openLCA process.

Publication-facing hierarchy handled here:
- Class 3 — EXTERNAL_VERIFIED. Phase A searches strict Nepal/target-geography,
  direct-product evidence first. If unresolved, Phase B starts a fresh independent
  clock and broadens geography, source type, and technically relevant product
  terminology. Every accepted numerical value must still be explicitly supported
  by retrieved evidence and pass unit, indicator, boundary, and product/family
  checks. Phase A/B remain auditable but are both reported as Class 3.
- Class 4 — UNVERIFIED_FALLBACK_ESTIMATE. Used only after both Class-3 phases fail.
  Database-anchored same-family and semantic-analog routes are attempted before one
  terminal model-only estimate.

No material-specific emission factor, density, source URL, preselected EPD, or
plausible GWP range is stored in this module. Class 4 is excluded from the verified
subtotal; Classes 1-3 form the verified result and Classes 1-4 form the complete
exploratory estimate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
import re
import time
from typing import Any
from urllib.parse import urlparse, urljoin

import pandas as pd

from unit_conversion import (
    norm_unit, convert_factor_reference_basis,
)
from evidence_consensus import (
    robust_positive_consensus, canonical_factor_basis, CONSENSUS_METHOD_VERSION,
)
from semantic_analog import infer_analog_plan
from technical_equivalence import infer_technical_equivalents, TECHNICAL_EQUIVALENCE_VERSION
from evidence_cache import RuntimeEvidenceCache, CACHE_VERSION
from material_taxonomy import (
    MATERIAL_TAXONOMY_VERSION, classify_material, external_title_compatibility,
    BIOGENIC_STORAGE_EXCLUDED_FAMILIES, choose_resolution_material, process_compatibility,
)
from guardrails import (
    GUARDRAIL_VERSION, emission_factor_plausible, emission_guardrail_context,
)

EXTERNAL_EF_RESOLVER_VERSION = "17.0-four-class-external-verified-two-phase"
TARGET_GEOGRAPHY = "Nepal"

TRACEABLE_EF_SYSTEM_PROMPT = """You are a technical LCA evidence extractor.

You are given web evidence for ONE construction material. Your job is only to
extract a defensible product-stage climate-change factor using the requested
indicator policy.

You are NOT allowed to invent a value, source, URL, year, boundary, declared unit,
or evidence quote.

Acceptance rules:
1. Use ONLY the supplied evidence candidates.
2. The source must represent the requested material/product or a clearly stated
   product-level equivalent. Do not substitute a different product family.
3. Follow required_indicator exactly:
   - GWP-total: extract total climate change / GWP-total only.
   - GWP-GHG_NO_BIOGENIC_STORAGE: prefer GWP-GHG, which excludes biogenic CO2
     uptake/emissions and biogenic carbon stored in the product. If GWP-GHG is
     unavailable, an explicitly reported GWP-fossil may be used as a conservative
     fallback. Do NOT use negative GWP-total or convert a negative value to positive.
4. Accept explicit A1-A3, cradle-to-gate, product-stage, production-stage, manufacturing-stage, or factory-gate values when the source clearly describes the product manufacturing boundary. Reject A1 only, A1-A5, B/C/D modules, whole-life/cradle-to-grave totals, and unspecified-boundary values.
5. The evidence must state the declared/reference quantity and unit associated
   with the impact value. Do not guess them.
6. source_result_id must exactly match one supplied result ID.
7. evidence_quote must be copied verbatim from the supplied snippet/extracted text.
8. If the source reports separate A1, A2 and A3 values but no explicit A1-A3 total
   in the supplied evidence, return found=false rather than doing an undocumented sum.
9. For GWP-GHG_NO_BIOGENIC_STORAGE, impact_value must be strictly positive.
10. If the SAME supplied source explicitly reports quantitative uncertainty for the factor,
    extract it without reinterpretation. Otherwise return null for every uncertainty field.
    Supported forms are: geometric standard deviation (GSD), coefficient of variation (CV),
    or a numeric lower/upper interval with its stated confidence level. Do not invent an
    uncertainty range. uncertainty_evidence_quote must be copied verbatim from the supplied
    evidence whenever any uncertainty field is non-null.
11. Return JSON only with exactly these keys:
found, impact_value, impact_unit, declared_quantity, declared_unit, boundary,
indicator, source_result_id, source_year, evidence_quote, uncertainty_type,
uncertainty_lower_value, uncertainty_upper_value, uncertainty_gsd, uncertainty_cv,
uncertainty_confidence_level, uncertainty_evidence_quote, reason
"""

RELAXED_EXTERNAL_EF_SYSTEM_PROMPT = """You are a numerical LCA evidence extractor for ONE retrieved source.

Strict verification has already failed. Extract only SOURCE-SUPPORTED GWP / global-warming / climate-change impact data explicitly present in the supplied evidence. Never estimate from memory. GWP is the only accepted environmental impact indicator. A generic carbon-footprint number is NOT sufficient unless the same retrieved evidence explicitly identifies it as GWP, Global Warming Potential, or the climate-change impact category. The relaxed External Verified phase may use an explicit A1-A3/cradle-to-gate result OR a GWP/climate-change result whose surrounding retrieved source clearly concerns production/manufacturing/embodied carbon and does not show a broader life-cycle boundary.

Rules:
1. Use ONLY the single supplied evidence candidate. Never invent a value, source, URL, boundary, unit, year, or evidence quote.
2. The product must represent the requested material or stated search proxy; do not knowingly substitute a different product family.
3. GWP is the only accepted environmental indicator. For ordinary materials use GWP-total / Global Warming Potential / the total climate-change category. For required_indicator=GWP-GHG_NO_BIOGENIC_STORAGE, use only GWP-GHG, GWP-fossil, or climate-change-fossil and never use a negative GWP-total. Never substitute AP, EP, ODP, POCP, ADP, energy, water, waste, or another impact category.
4. Prefer an explicit A1-A3/product-stage/cradle-to-gate total. If the literal boundary is absent, a TOTAL may still be returned when the SAME retrieved evidence clearly places the GWP/carbon value in material production/manufacturing/embodied-carbon context and does not indicate cradle-to-grave, whole-life, A1-A5, use-stage, or end-of-life coverage. Report boundary exactly as stated when available; otherwise use a short evidence-based phrase such as production/manufacturing context, not an invented A1-A3 label.
5. If no explicit A1-A3 total is present but the SAME source explicitly reports separate A1, A2 and A3 values for the same GWP indicator, impact unit and declared unit, you may return value_mode=SUM_A1_A2_A3. Python, not you, will add the three modules. Evidence may contain lines beginning `[PDF TABLE PAGE ...]`; pipe characters preserve table columns. Use only the GWP/global-warming/climate-change row and its A1/A2/A3 columns; ignore neighboring AP, EP, ODP, POCP, ADP, energy, water and waste rows.
6. For value_mode=TOTAL, set impact_value and set a1_value/a2_value/a3_value to null.
7. For value_mode=SUM_A1_A2_A3, set impact_value=null and return all three explicit module values.
8. The declared/reference quantity and unit must be explicit or directly represented by an intensity denominator (for example kg CO2e/kg).
9. evidence_quote must be copied verbatim from supplied text and support the value(s), unit/basis and product-stage interpretation as far as possible.
10. Do not average sources, convert units, or choose a final factor. Python handles normalization and consensus.
11. Return JSON only with one top-level key records. records must contain exactly one object with exactly these keys:
source_result_id, found, value_mode, impact_value, a1_value, a2_value, a3_value, impact_unit, declared_quantity, declared_unit, boundary, indicator, source_year, evidence_quote, reason
"""

TERMINAL_LLM_ONLY_EF_SINGLE_SYSTEM_PROMPT = """You are producing ONE terminal exploratory A1-A3 GWP estimate for a construction material after Class-3 source-supported and dynamically database-anchored proxy routes have failed.

This value is intentionally UNVERIFIED and will be isolated from verified subtotal. No material-specific GWP value, density, plausible range, correction factor, or expected magnitude is supplied. Do not claim a source, citation, EPD, URL, publication year, or verification.

Return exactly one JSON object with these keys:
found, central_value, lower_value, upper_value, reference_unit, boundary, indicator, geography_assumption, product_interpretation, estimation_basis, rationale

Rules:
- Use requested_reference_unit exactly and reason explicitly in that denominator basis.
- Estimate A1-A3/cradle-to-gate only.
- lower_value < central_value < upper_value and all values must be finite and positive.
- Follow required_indicator exactly; for bio-based materials exclude storage/sequestration credits and use positive GWP-GHG or GWP-fossil.
- This is a terminal model-only estimate, not source-supported evidence.
- There is no repeated-value consensus step. Produce only this single estimate and do not imitate any prior model number.
- Preserve the requested product identity explicitly in product_interpretation and rationale. The caller supplies deterministic_material_family derived only from the BOM text/taxonomy; treat it as a binding identity constraint, not as environmental evidence.
- Do not merely echo or reinterpret an acronym. product_interpretation must spell out enough chemistry/form/function words to be independently compatible with deterministic_material_family. If an acronym could mean something else, use the supplied family rather than inventing another expansion.
- A deterministic family gate will reject contradictions before any terminal number is retained.
- JSON only.
"""

DYNAMIC_DATABASE_PROXY_SELECTOR_SYSTEM_PROMPT = """You are selecting a numerical proxy PROCESS for a construction-material A1-A3 screening fallback.

The caller supplies only process identity metadata from a frozen ELCD/openLCA catalog. No emission-factor values are shown to you. Your task is semantic/process selection only.

Rules:
1. Preserve the requested product's dominant chemistry, manufacturing route, physical form, and construction function.
2. Prefer the same product family. A broader analog is allowed only when the candidate explicitly states which dynamically inferred analog it represents.
3. Do not invent a process, UUID, emission factor, density, conversion factor, source, or citation.
4. Rank only supplied process UUIDs. If no supplied candidate is technically defensible, return an empty ranking.
5. Geography is secondary to technical compatibility.
6. Return JSON only with exactly these keys: ranked_process_uuids, selection_reason.
7. ranked_process_uuids must be a flat list of at most three supplied UUID strings.
"""

TERMINAL_MINIMAL_EF_SYSTEM_PROMPT = """You are the last numerical fallback for an exploratory A1-A3 construction-material GWP screen.

No source support or material-specific expected value/range is supplied. The result will be explicitly labeled UNVERIFIED and excluded from verified subtotal. Return a usable positive number only if you can make a model-only engineering estimate. Do not claim a source.

Return JSON only with exactly these keys:
found, central_value, reference_unit, boundary, indicator, product_interpretation, rationale

Rules:
- Use requested_reference_unit exactly.
- Estimate A1-A3/cradle-to-gate only.
- central_value must be finite and positive.
- Follow required_indicator; bio-based materials must exclude storage/sequestration credit and use positive GWP-GHG or GWP-fossil.
- deterministic_material_family is a binding identity constraint derived from the BOM taxonomy. product_interpretation must explicitly describe that family rather than merely echoing or re-expanding an acronym.
"""



def _clean(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _call_matcher(matcher, system_prompt: str, payload: dict[str, Any], *, max_new_tokens: int | None = None) -> str:
    """Call Qwen with a task-specific output budget without changing benchmark settings.

    Lightweight mocked matchers used by the deterministic tests may not accept
    the override keyword, so fall back to the legacy two-argument signature.
    """
    try:
        return matcher.generate_with_system(
            system_prompt, payload, max_new_tokens_override=max_new_tokens
        )
    except TypeError:
        return matcher.generate_with_system(system_prompt, payload)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return _clean(v).lower() in {"1", "true", "yes", "y"}


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str) and not v.strip():
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


_LOW_QUALITY_HOSTS = (
    "scribd.com", "pinterest.", "quora.", "reddit.", "wisdomanswer.com",
    "blog.welcu.com", "leukstekadotip.nl", "grokipedia.com",
)

_EPD_PROGRAM_HOSTS = (
    "eco-platform.org", "environdec.com", "epd-global.no", "ibu-epd.com",
    "epdhub.com", "epditaly.it", "epddanmark.dk", "aenor.com", "bregroup.com",
    "igbc.ie", "bau-epd.at", "daphabitat.pt", "globalgreentag.com",
    "epd-australasia.com", "ul.com",
)

_PEER_REVIEWED_HOSTS = (
    "sciencedirect.com", "springer.com", "springeropen.com", "wiley.com",
    "tandfonline.com", "mdpi.com", "sagepub.com", "nature.com",
)

_TECHNICAL_DATABASE_HOSTS = (
    "buildingtransparency.org", "carbonleadershipforum.org", "nrmca.org",
    "athenasmi.org", "cerib.com", "ice.org.uk",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", _clean(text)).strip().lower()


def _source_quality(candidate: dict[str, Any]) -> tuple[bool, str]:
    host = _url_host(candidate.get("url", ""))
    evidence = _normalize_text(
        f"{candidate.get('title','')} {candidate.get('snippet','')} {candidate.get('excerpt','')}"
    )
    if not host:
        return False, "UNKNOWN_SOURCE"
    if any(bad in host for bad in _LOW_QUALITY_HOSTS):
        return False, "LOW_QUALITY_WEB"
    if any(host == h or host.endswith('.' + h) for h in _EPD_PROGRAM_HOSTS):
        return True, "VERIFIED_EPD_PROGRAM"
    if host.endswith(".gov.np") or host == "gov.np":
        return True, "NEPAL_GOVERNMENT"
    if ".gov." in host or host.endswith(".gov"):
        return True, "GOVERNMENT"
    if ".edu." in host or host.endswith(".edu") or ".ac." in host:
        return True, "ACADEMIC"
    if any(host == h or host.endswith('.' + h) for h in _PEER_REVIEWED_HOSTS):
        return True, "PEER_REVIEWED_PUBLICATION"
    if any(host == h or host.endswith('.' + h) for h in _TECHNICAL_DATABASE_HOSTS):
        return True, "TECHNICAL_LCA_DATABASE"
    # A manufacturer-hosted EPD can be useful, but require strong EPD/EN15804 markers.
    if (
        "environmental product declaration" in evidence
        and ("en 15804" in evidence or "iso 14025" in evidence)
        and ("verified" in evidence or "verification" in evidence or "epd" in evidence)
    ):
        return True, "MANUFACTURER_EPD"
    return False, "UNVERIFIED_WEB_SOURCE"


def _infer_geography(candidate: dict[str, Any]) -> str:
    host = _url_host(candidate.get("url", ""))
    evidence = _normalize_text(
        f"{candidate.get('title','')} {candidate.get('snippet','')} {candidate.get('excerpt','')}"
    )
    checks = [
        ("Nepal", ("nepal",), (".np",)),
        ("India", ("india",), (".in",)),
        ("Bangladesh", ("bangladesh",), (".bd",)),
        ("Sri Lanka", ("sri lanka",), (".lk",)),
        ("Pakistan", ("pakistan",), (".pk",)),
        ("Bhutan", ("bhutan",), (".bt",)),
        ("China", ("china",), (".cn",)),
        ("Japan", ("japan",), (".jp",)),
        ("Australia", ("australia",), (".au",)),
        ("United Kingdom", ("united kingdom", " uk "), (".uk",)),
        ("United States", ("united states", " usa ", " u.s. "), (".us",)),
        ("Canada", ("canada", " canadian "), (".ca",)),
        ("Germany", ("germany",), (".de",)),
        ("Norway", ("norway",), (".no",)),
    ]
    padded = f" {evidence} "
    for geo, words, suffixes in checks:
        if any(w in padded for w in words) or any(host.endswith(s) for s in suffixes):
            return geo
    if "north america" in evidence:
        return "North America"
    if "global" in evidence or "worldwide" in evidence:
        return "Global"
    return "Unspecified"


def _tier_geography_ok(tier: str, geography: str) -> bool:
    if tier == "NEPAL":
        return geography == "Nepal"
    if tier == "SOUTH_ASIA":
        return geography in {"Nepal", "India", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan"}
    if tier == "ASIA":
        return geography in {
            "Nepal", "India", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan",
            "China", "Japan",
        }
    return True


def _target_geography_supported(candidate: dict[str, Any], target_geography: str) -> bool:
    """Return True only when the retrieved source itself supports the target geography.

    This is the Class-3 geography gate. Search-query wording is deliberately
    ignored so a query containing the target country cannot make a foreign or
    global source appear local. The relaxed External Verified phase does NOT use this gate.
    """
    target = _normalize_text(target_geography)
    if not target:
        return False
    inferred = _normalize_text(_infer_geography(candidate))
    if inferred and inferred != "unspecified" and inferred == target:
        return True

    evidence = _normalize_text(
        f"{candidate.get('title','')} {candidate.get('snippet','')} {candidate.get('excerpt','')}"
    )
    host = _url_host(candidate.get('url', ''))
    padded = f" {evidence} "

    aliases = {
        "nepal": ("nepal",),
        "india": ("india",),
        "bangladesh": ("bangladesh",),
        "sri lanka": ("sri lanka",),
        "pakistan": ("pakistan",),
        "bhutan": ("bhutan",),
        "china": ("china",),
        "japan": ("japan",),
        "australia": ("australia",),
        "united kingdom": ("united kingdom", " uk "),
        "united states": ("united states", " usa ", " u.s. "),
        "germany": ("germany",),
        "norway": ("norway",),
    }
    if any(a in padded for a in aliases.get(target, (target,))):
        return True

    suffixes = {
        "nepal": ".np", "india": ".in", "bangladesh": ".bd",
        "sri lanka": ".lk", "pakistan": ".pk", "bhutan": ".bt",
        "china": ".cn", "japan": ".jp", "australia": ".au",
        "united kingdom": ".uk", "united states": ".us",
        "germany": ".de", "norway": ".no",
    }
    suffix = suffixes.get(target)
    return bool(suffix and host.endswith(suffix))


def _quote_supported(quote: str, candidate: dict[str, Any]) -> bool:
    q = _normalize_text(quote)
    if not q:
        return False
    evidence = _normalize_text(f"{candidate.get('snippet','')} {candidate.get('excerpt','')}")
    return q in evidence


def _requires_no_biogenic_storage(material: str) -> bool:
    return classify_material(material) in BIOGENIC_STORAGE_EXCLUDED_FAMILIES


def _required_indicator_policy(material: str) -> str:
    return "GWP-GHG_NO_BIOGENIC_STORAGE" if _requires_no_biogenic_storage(material) else "GWP-total"


def _biogenic_generic_gwp_is_separate_from_storage(indicator: str, source_text: str) -> bool:
    """Allow a positive generic GWP row only when the source distinguishes biogenic GWP.

    This is an indicator-structure rule, not a material value. It is useful for
    EPDs that publish a conventional GWP row plus a separate GWPBIO / biogenic-CO2
    row rather than the newer GWP-GHG or GWP-fossil labels.
    """
    ind = _normalize_text(indicator).replace("–", "-").replace("—", "-")
    compact_ind = re.sub(r"[\s_-]+", "", ind)
    if not ("gwp" in ind or "global warming" in ind or "climate change" in ind):
        return False
    if any(x in compact_ind for x in ("gwptotal", "gwpbiogenic", "gwpbio", "gwpluluc")):
        return False
    if "fossil" in ind:
        return False
    src = _normalize_text(source_text).replace("–", "-").replace("—", "-")
    compact_src = re.sub(r"[\s_-]+", "", src)
    separate_biogenic = (
        "gwpbiogenic" in compact_src
        or "gwpbio" in compact_src
        or "withbiogenicco2" in compact_src
        or "w/biogenicco2" in compact_src
    )
    return separate_biogenic


def _indicator_component_flags(text: str) -> set[str]:
    """Classify explicit GWP component labels without using any numeric priors.

    The earlier parser could miss labels such as ``Climate change (GWP) -
    biogenic`` because punctuation sat between ``GWP`` and ``biogenic``.  This
    helper deliberately inspects normalized words instead of one fragile compact
    substring, so component rows cannot masquerade as GWP-total.
    """
    t = _normalize_text(text).replace("_", "-").replace("–", "-").replace("—", "-")
    compact = re.sub(r"[^a-z0-9]+", "", t)
    flags: set[str] = set()
    if "biogenic" in t or "gwpbio" in compact:
        flags.add("BIOGENIC")
    if "luluc" in t or "land use and land use change" in t or "land-use and land-use change" in t:
        flags.add("LULUC")
    if "fossil" in t:
        flags.add("FOSSIL")
    if "gwp-ghg" in t or "gwp ghg" in t or "gwpghg" in compact:
        flags.add("GHG")
    if "gwp-total" in t or "gwp total" in t or "gwptotal" in compact or "climate change - total" in t or "climate change total" in t:
        flags.add("TOTAL")
    return flags


def _gwp_indicator_ok(indicator: str, material: str = "", source_text: str = "") -> bool:
    t = _normalize_text(indicator).replace("_", "-").replace("–", "-").replace("—", "-")
    if not t or not ("gwp" in t or "global warming" in t or "climate change" in t):
        return False
    flags = _indicator_component_flags(t)
    if _requires_no_biogenic_storage(material):
        # GWP-GHG is preferred; explicit fossil climate change is the allowed
        # conservative fallback. A biogenic or LULUC component is never accepted.
        if "BIOGENIC" in flags or "LULUC" in flags:
            return False
        if "GHG" in flags or "FOSSIL" in flags:
            return True
        return _biogenic_generic_gwp_is_separate_from_storage(indicator, source_text)

    # Ordinary materials require the total/generic climate-change result. Reject
    # explicit component rows regardless of punctuation or row naming style.
    if flags & {"BIOGENIC", "LULUC", "FOSSIL"}:
        return False
    return True


def _boundary_ok(boundary: str) -> bool:
    """Accept explicit A1-A3 or an unambiguously product-stage/cradle-to-gate basis.

    This is intentionally a little more permissive than the previous version so
    credible external datasets/literature are not discarded merely because they
    write "cradle-to-gate" instead of the EN 15804 module label. Broader life-cycle
    boundaries remain excluded.
    """
    t = _normalize_text(boundary).replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", t)
    if any(x in t for x in ("cradle-to-grave", "cradle to grave", "whole life", "a1-a5", "a1 to a5", "a1-c", "a1 to c")):
        return False
    if "a1-a3" in compact or "a1to a3" in t or "a1 to a3" in t:
        return True
    return any(x in t for x in (
        "cradle-to-gate", "cradle to gate", "product stage", "product-stage",
        "production stage", "manufacturing stage", "factory gate",
    ))


def _impact_to_kg_co2e(value: float, unit: str) -> float | None:
    u = _normalize_text(unit).replace("₂", "2")
    u = u.replace("co₂", "co2")
    if "kg" in u and ("co2" in u or "co₂" in unit.lower()):
        return value
    if ("tonne" in u or re.search(r"\bt\b", u)) and ("co2" in u or "co₂" in unit.lower()):
        return value * 1000.0
    if "g" in u and "kg" not in u and ("co2" in u or "co₂" in unit.lower()):
        return value / 1000.0
    return None


def _normalize_declared_unit(unit: str) -> str:
    u = norm_unit(unit)
    aliases = {
        "unit": "item", "each": "item", "piece": "item", "pcs": "item",
        "ton": "t", "tonne": "t", "metrictonne": "t",
    }
    return aliases.get(u, u)


def normalize_factor(impact_value: float, impact_unit: str, declared_quantity: float, declared_unit: str):
    impact_kg = _impact_to_kg_co2e(impact_value, impact_unit)
    if impact_kg is None:
        return None
    q = _num(declared_quantity)
    if q is None or q <= 0:
        return None
    ref = _normalize_declared_unit(declared_unit)
    if ref not in {"kg", "g", "t", "m3", "m2", "item"}:
        return None
    return impact_kg / q, ref


def _material_family(material: str) -> str:
    m = _normalize_text(material)
    if any(x in m for x in ("steel", "iron", "nail", "binding wire", "rebar", "galvanized", "galvanised", "cgi", "metal sheet", "corrugated")):
        return "METAL"
    if any(x in m for x in ("plywood", "timber", "wood", "bamboo")):
        return "BIO_BASED"
    if any(x in m for x in ("soil", "earth", "mud", "stone", "gravel", "sand", "aggregate", "rock")):
        return "MINERAL_EARTH"
    return "OTHER"


def _llm_fallback_plausible(material: str, ef: float, ref: str, lower: float | None, upper: float | None) -> tuple[bool, str]:
    """Deterministic structural QA for an UNVERIFIED estimate.

    No material-specific expected emission factor or numeric range is encoded
    here. The guard checks only sign, interval consistency, and reference-unit
    compatibility. Unverified estimates remain clearly separated from the
    traceable subtotal.
    """
    if ef <= 0:
        return False, "nonpositive_unverified_a1a3_factor_not_allowed"
    if lower is not None and lower < 0:
        return False, "negative_unverified_lower_bound_not_allowed"
    if upper is not None and upper < 0:
        return False, "negative_unverified_upper_bound_not_allowed"
    if lower is not None and upper is not None and lower > upper:
        return False, "lower_bound_greater_than_upper_bound"
    if lower is not None and ef < lower:
        return False, "base_value_below_lower_bound"
    if upper is not None and ef > upper:
        return False, "base_value_above_upper_bound"
    unit_ok, unit_reason = _reference_unit_compatible(material, ref)
    if not unit_ok:
        return False, unit_reason
    return True, "ok"

def _focus_lca_text(text: str, max_chars: int = 12000) -> str:
    """Keep multiple high-value windows around EPD/LCA result tables.

    This is material-agnostic: it searches only for environmental-result and
    declared-unit vocabulary, never for a prescribed factor value.
    """
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    low = cleaned.lower().replace("–", "-").replace("—", "-")
    needles = [
        "declared unit", "functional unit", "reference flow", "declared unit mass",
        "a1-a3", "a1 a3", "a1", "a2", "a3", "gwp-total", "gwp fossil",
        "gwp-fossil", "gwp-ghg", "global warming", "climate change",
        "environmental data summary", "environmental impact", "kg co2", "kgco2",
        "cradle-to-gate", "cradle to gate",
    ]
    centers = []
    for needle in needles:
        start = 0
        while len(centers) < 24:
            i = low.find(needle, start)
            if i < 0:
                break
            centers.append(i)
            start = i + len(needle)
    if not centers:
        return cleaned[:max_chars]
    pieces = []
    for i in sorted(set(centers))[:18]:
        a = max(0, i - 650)
        b = min(len(cleaned), i + 1400)
        pieces.append(cleaned[a:b])
    return " ... ".join(pieces)[:max_chars]




_GWP_TEXT_SIGNALS = (
    "gwp", "global warming potential", "global warming", "climate change",
)
_GWP_FOSSIL_SIGNALS = (
    "gwp-ghg", "gwp ghg", "gwp-fossil", "gwp fossil",
    "climate change - fossil", "climate change fossil",
)


def _has_gwp_signal(text: str) -> bool:
    low = _normalize_text(text).replace("–", "-").replace("—", "-")
    return any(term in low for term in _GWP_TEXT_SIGNALS)


def _gwp_page_score(text: str) -> int:
    """Rank PDF pages using only GWP/schema vocabulary, never factor values."""
    low = _normalize_text(text).replace("–", "-").replace("—", "-")
    if not low:
        return 0
    score = 0
    if "gwp" in low:
        score += 8
    if "global warming" in low:
        score += 7
    if "climate change" in low:
        score += 6
    if all(x in low for x in ("a1", "a2", "a3")):
        score += 6
    elif any(x in low for x in ("a1-a3", "a1 a3", "cradle-to-gate", "cradle to gate")):
        score += 5
    if any(x in low for x in ("declared unit", "functional unit", "reference unit", "reference flow")):
        score += 3
    if "kg co2" in low or "kgco2" in low:
        score += 2
    if any(x in low for x in ("environmental impact", "lcia", "impact assessment", "environmental data summary")):
        score += 2
    return score


def _focus_gwp_lines(text: str, max_chars: int = 10000) -> str:
    """Preserve line/table order around GWP and A1-A3 evidence."""
    raw = (text or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    centers = []
    for i, line in enumerate(lines):
        low = _normalize_text(line).replace("–", "-").replace("—", "-")
        if (_has_gwp_signal(low)
                or any(x in low for x in ("a1-a3", "a1 a3", "declared unit", "functional unit", "reference unit"))
                or ("a1" in low and "a2" in low and "a3" in low)):
            centers.append(i)
    if not centers:
        return _focus_lca_text("\n".join(lines), max_chars=max_chars)
    keep = set()
    for i in centers[:24]:
        for j in range(max(0, i - 5), min(len(lines), i + 7)):
            keep.add(j)
    out = []
    last = None
    for j in sorted(keep):
        if last is not None and j > last + 1:
            out.append("...")
        out.append(lines[j])
        last = j
    return "\n".join(out)[:max_chars]


def _normalize_table_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _table_to_gwp_text(table: Any, page_number: int) -> str:
    """Render only GWP-relevant table structure as auditable plain text."""
    if not isinstance(table, list) or not table:
        return ""
    rows = []
    for row in table:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [_normalize_table_cell(x) for x in row]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    row_text = [" | ".join(r) for r in rows]
    whole = "\n".join(row_text)
    if not _has_gwp_signal(whole):
        return ""
    selected = set()
    for i, line in enumerate(row_text):
        low = _normalize_text(line).replace("–", "-").replace("—", "-")
        is_header = (
            ("a1" in low and "a2" in low and "a3" in low)
            or ("indicator" in low and "unit" in low)
            or any(x in low for x in ("declared unit", "functional unit", "reference unit", "reference flow"))
        )
        if _has_gwp_signal(low) or is_header:
            selected.add(i)
    if not selected:
        return ""
    lines = [f"[PDF TABLE PAGE {page_number}]" ]
    lines.extend(row_text[i] for i in sorted(selected))
    return "\n".join(lines)


def _extract_pdf_gwp_evidence(content: bytes, page_limit: int, max_chars: int = 24000) -> str:
    """Extract GWP-only evidence from a PDF while preserving table/module structure.

    No OCR and no material-specific values are used. pypdf provides page text and
    pdfplumber is used only on GWP-relevant pages to preserve tabular A1/A2/A3 data.
    """
    if not content or not content.lstrip().startswith(b"%PDF"):
        return ""
    try:
        import logging
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content), strict=False)
    except Exception:
        return ""

    count = min(len(reader.pages), max(1, int(page_limit)))
    page_texts: list[str] = []
    scored: list[tuple[int, int]] = []
    for i in range(count):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        page_texts.append(txt)
        scored.append((_gwp_page_score(txt), i))

    target_indices = [i for score, i in sorted(scored, reverse=True) if score > 0][:12]

    # If pypdf did not expose the relevant labels, use pdfplumber text as a
    # secondary non-OCR page detector before giving up.
    plumber = None
    try:
        import pdfplumber
        plumber = pdfplumber.open(io.BytesIO(content))
        if not target_indices:
            alt_scores = []
            for i in range(min(len(plumber.pages), count)):
                try:
                    txt = plumber.pages[i].extract_text(x_tolerance=2, y_tolerance=3) or ""
                except Exception:
                    txt = ""
                if txt and not page_texts[i]:
                    page_texts[i] = txt
                alt_scores.append((_gwp_page_score(txt), i))
            target_indices = [i for score, i in sorted(alt_scores, reverse=True) if score > 0][:12]
    except Exception:
        plumber = None

    if not target_indices:
        target_indices = list(range(min(count, 6)))

    table_blocks = []
    page_blocks = []
    for i in sorted(set(target_indices)):
        txt = page_texts[i] if i < len(page_texts) else ""
        focused = _focus_gwp_lines(txt, max_chars=7000)
        if focused:
            page_blocks.append(f"[PDF PAGE {i+1}]\n{focused}")
        if plumber is not None and i < len(plumber.pages):
            try:
                tables = plumber.pages[i].extract_tables() or []
            except Exception:
                tables = []
            for table in tables[:8]:
                block = _table_to_gwp_text(table, i + 1)
                if block:
                    table_blocks.append(block)

    if plumber is not None:
        try:
            plumber.close()
        except Exception:
            pass

    # Tables go first so the bounded excerpt passed to Qwen keeps A1/A2/A3
    # columns even when the PDF contains a lot of narrative text.
    combined = "\n\n".join(table_blocks + page_blocks)
    if not combined:
        combined = _focus_gwp_lines("\n".join(page_texts), max_chars=max_chars)
    return combined[:max_chars]




def _pdf_table_blocks(text: str) -> list[str]:
    """Return table blocks emitted by the GWP-focused PDF reader."""
    raw = text or ""
    starts = [m.start() for m in re.finditer(r"(?m)^\[PDF TABLE PAGE \d+\]\s*$", raw)]
    blocks: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(raw)
        # Stop before the first non-table page block when present.
        chunk = raw[start:end]
        page_marker = re.search(r"(?m)^\[PDF PAGE \d+\]\s*$", chunk)
        if page_marker:
            chunk = chunk[:page_marker.start()]
        chunk = chunk.strip()
        if chunk:
            blocks.append(chunk)
    return blocks


def _table_rows_from_block(block: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (block or "").splitlines():
        line = line.strip()
        if not line or line.startswith("[PDF TABLE PAGE"):
            continue
        if "|" not in line:
            continue
        cells = [_normalize_table_cell(x) for x in line.split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _module_label(cell: str) -> str | None:
    t = _normalize_text(cell).replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", t)
    if re.fullmatch(r"a1(?:-|to)a3", compact):
        return "A1-A3"
    if compact == "a1":
        return "A1"
    if compact == "a2":
        return "A2"
    if compact == "a3":
        return "A3"
    return None


def _indicator_row_score(cells: list[str], material: str, source_text: str = "") -> int:
    text = " | ".join(cells)
    low = _normalize_text(text).replace("–", "-").replace("—", "-")
    if not _has_gwp_signal(low) or not _gwp_indicator_ok(text, material, source_text):
        return -1
    flags = _indicator_component_flags(low)
    if _requires_no_biogenic_storage(material):
        if "GHG" in flags:
            return 100
        if "FOSSIL" in flags:
            return 90
        if _biogenic_generic_gwp_is_separate_from_storage(text, source_text):
            return 70
        return -1
    if "TOTAL" in flags:
        return 100
    if "global warming potential" in low:
        return 95
    if "climate change" in low:
        return 90
    if "gwp" in low:
        return 85
    return -1


def _parse_first_numeric(cell: str) -> float | None:
    for token in re.findall(_SCI_NUM, cell or ""):
        v = _parse_decimal_token(token)
        if v is not None and math.isfinite(v):
            return float(v)
    return None


def _impact_unit_from_text(text: str) -> tuple[str | None, str | None]:
    """Return normalized impact unit plus optional declared-unit denominator.

    This parses unit syntax only; it does not contain any material-specific factor.
    """
    t = (text or "").replace("CO₂", "CO2").replace("co₂", "co2").replace("m³", "m3").replace("m²", "m2")
    low = _normalize_text(t)
    if "co2" not in low:
        return None, None
    if re.search(r"\bkg\s*co2", low):
        impact = "kg CO2e"
    elif re.search(r"(?:^|\W)g\s*co2", low):
        impact = "g CO2e"
    elif re.search(r"\b(?:t|tonne|ton)\s*co2", low):
        impact = "t CO2e"
    else:
        return None, None

    denom = None
    patterns = [
        r"(?:/|per\s+)(kg|g|t|tonne|ton|m3|m2|item|piece|each|unit)\b",
        r"co2(?:e|\s*eq(?:uivalent)?)?\s*(?:/|per\s+)(kg|g|t|tonne|ton|m3|m2|item|piece|each|unit)\b",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            denom = _normalize_declared_unit(m.group(1))
            break
    return impact, denom


def _declared_basis_from_evidence(text: str) -> tuple[float | None, str | None]:
    """Extract only an explicitly stated declared/reference quantity and unit."""
    normalized = (text or "").replace("m³", "m3").replace("m²", "m2")
    # Prefer explicit declared/functional/reference basis statements.
    patterns = [
        rf"(?:declared|functional|reference)\s+(?:unit|flow)\s*(?:is|:|=|-)?\s*({_SCI_NUM})?\s*(kg|g|t|tonne|ton|m3|m2|item|piece|each|unit)\b",
        rf"(?:declared|functional|reference)\s+(?:unit|flow).*?({_SCI_NUM})\s*(kg|g|t|tonne|ton|m3|m2|item|piece|each|unit)\b",
    ]
    for pat in patterns:
        m = re.search(pat, normalized, flags=re.I | re.S)
        if not m:
            continue
        q = _parse_decimal_token(m.group(1)) if m.group(1) else 1.0
        u = _normalize_declared_unit(m.group(2))
        if q is not None and q > 0 and u in {"kg", "g", "t", "m3", "m2", "item"}:
            return float(q), u
    return None, None


def _indicator_name_from_row(cells: list[str]) -> str:
    for cell in cells:
        if _has_gwp_signal(cell):
            return cell.strip()
    return "GWP"


def _deterministic_gwp_table_record(material: str, candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Parse a structured retrieved EPD table before asking the LLM.

    The parser is deliberately schema-driven and GWP-only. It never contains or
    supplies a material-specific GWP value. It accepts an explicit A1-A3 total,
    or explicit A1, A2 and A3 module cells and lets Python sum those source values.
    """
    full = _clean(candidate.get("full_text")) or _candidate_evidence_text(candidate)
    blocks = _pdf_table_blocks(full)
    if not blocks:
        return None, "no_structured_pdf_table"

    declared_q, declared_u = _declared_basis_from_evidence(full)
    best: tuple[int, dict[str, Any]] | None = None

    for block in blocks:
        raw_rows = _table_rows_from_block(block)
        if len(raw_rows) < 2:
            continue

        # pdfplumber sometimes emits visual spacer columns as empty cells. Try
        # both the raw matrix and a compacted matrix with empty spacer cells
        # removed; no numeric values are invented or reordered within a row.
        row_variants = [raw_rows]
        compact_rows = [[c for c in row if c] for row in raw_rows]
        if compact_rows != raw_rows:
            row_variants.append(compact_rows)

        for rows in row_variants:
            # Find a header row carrying product-stage module labels. A table may
            # have multiple header rows, so use the row with the most module labels.
            header_idx = None
            header_map: dict[str, int] = {}
            for r_i, row in enumerate(rows):
                local: dict[str, int] = {}
                for c_i, cell in enumerate(row):
                    label = _module_label(cell)
                    if label:
                        local[label] = c_i
                if len(local) > len(header_map):
                    header_map = local
                    header_idx = r_i
            if not header_map:
                continue

            for r_i, row in enumerate(rows):
                if r_i == header_idx:
                    continue
                score = _indicator_row_score(row, material, full)
                if score < 0:
                    continue

                impact_unit = None
                denominator = None
                for cell in row:
                    iu, den = _impact_unit_from_text(cell)
                    if iu:
                        impact_unit = iu
                        denominator = den
                        break
                if impact_unit is None:
                    # Occasionally the unit is placed in a separate header row.
                    for hrow in rows[: max(1, (header_idx or 0) + 1)]:
                        for cell in hrow:
                            iu, den = _impact_unit_from_text(cell)
                            if iu:
                                impact_unit = iu
                                denominator = den
                                break
                        if impact_unit:
                            break
                if impact_unit is None:
                    continue

                dq, du = declared_q, declared_u
                if denominator:
                    dq, du = 1.0, denominator
                if dq is None or du is None:
                    continue

                indicator = _indicator_name_from_row(row)
                rec: dict[str, Any] | None = None
                if "A1-A3" in header_map:
                    idx = header_map["A1-A3"]
                    if idx < len(row):
                        total = _parse_first_numeric(row[idx])
                        if total is not None:
                            rec = {
                                "source_result_id": candidate.get("result_id"),
                                "found": True,
                                "value_mode": "TOTAL",
                                "impact_value": total,
                                "a1_value": None, "a2_value": None, "a3_value": None,
                                "impact_unit": impact_unit,
                                "declared_quantity": dq,
                                "declared_unit": du,
                                "boundary": "A1-A3",
                                "indicator": indicator,
                                "source_year": "",
                                "evidence_quote": block,
                                "reason": "Deterministic GWP table parser extracted an explicit A1-A3 source value.",
                                "extraction_method": "DETERMINISTIC_GWP_TABLE_A1_A3_TOTAL",
                            }
                elif all(k in header_map for k in ("A1", "A2", "A3")):
                    vals = []
                    ok = True
                    for k in ("A1", "A2", "A3"):
                        idx = header_map[k]
                        if idx >= len(row):
                            ok = False
                            break
                        v = _parse_first_numeric(row[idx])
                        if v is None:
                            ok = False
                            break
                        vals.append(v)
                    if ok:
                        rec = {
                            "source_result_id": candidate.get("result_id"),
                            "found": True,
                            "value_mode": "SUM_A1_A2_A3",
                            "impact_value": None,
                            "a1_value": vals[0], "a2_value": vals[1], "a3_value": vals[2],
                            "impact_unit": impact_unit,
                            "declared_quantity": dq,
                            "declared_unit": du,
                            "boundary": "A1+A2+A3",
                            "indicator": indicator,
                            "source_year": "",
                            "evidence_quote": block,
                            "reason": "Deterministic GWP table parser extracted explicit A1, A2 and A3 source values; Python will sum them.",
                            "extraction_method": "DETERMINISTIC_GWP_TABLE_A1_A2_A3",
                        }
                if rec is not None and (best is None or score > best[0]):
                    best = (score, rec)

    if best is None:
        return None, "no_usable_gwp_a1_a3_table_record"
    return best[1], "ok"

def _family_search_term(material: str) -> str | None:
    """Return a generic taxonomy-family phrase for query expansion.

    This is a search vocabulary only. It contains no emission factor, source URL,
    manufacturer, EPD identifier, or predetermined answer.
    """
    family = classify_material(material)
    if family == "UNKNOWN":
        return None
    return family.lower().replace("_", " ")


def _query_sets(material: str, target_geography: str):
    """Build broad live-search plans without preselecting any source or factor."""
    m = material.strip()
    family_term = _family_search_term(material)

    def direct_queries(geo_label: str, geo_term: str) -> list[str]:
        q = [
            f'"{m}" "A1-A3" GWP {geo_term} EPD',
            f'"{m}" "GWP-total" "declared unit" {geo_term}',
            f'"{m}" "environmental product declaration" {geo_term} filetype:pdf',
            f'"{m}" "cradle-to-gate" "kg CO2" {geo_term}',
        ]
        if geo_label == "GLOBAL":
            q.extend([
                f'site:environdec.com/library "{m}"',
                f'site:manage.epdhub.com "{m}" EPD',
                f'"{m}" EPD "environmental data summary"',
            ])
        return q

    geo_terms = {
        "NEPAL": target_geography,
        "SOUTH_ASIA": "India OR South Asia",
        "ASIA": "Asia",
        "GLOBAL": "",
    }
    for tier in ("NEPAL", "SOUTH_ASIA", "ASIA", "GLOBAL"):
        yield tier, "DIRECT_PRODUCT", None, direct_queries(tier, geo_terms[tier])
        if family_term and _normalize_text(family_term) != _normalize_text(m):
            fq = [
                f'"{family_term}" "A1-A3" GWP {geo_terms[tier]} EPD',
                f'"{family_term}" "GWP-total" "declared unit" {geo_terms[tier]}',
                f'"{family_term}" "environmental product declaration" {geo_terms[tier]} filetype:pdf',
            ]
            yield tier, "PRODUCT_PROXY", f"Taxonomy-family query expansion: {family_term}", fq

def _strip_html(raw: str) -> str:
    import html as _html
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw or "")
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = _html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_decimal_token(token: str) -> float | None:
    s = _clean(token).replace("−", "-").replace("–", "-")
    if not s:
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


_SCI_NUM = r"[-+]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


def _candidate_evidence_text(candidate: dict[str, Any]) -> str:
    return " ".join([
        _clean(candidate.get("snippet")),
        _clean(candidate.get("excerpt")),
    ]).strip()


def _source_boundary_supported(candidate: dict[str, Any]) -> bool:
    text = _candidate_evidence_text(candidate).lower()
    text = text.replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", text)
    # A valid EPD may report A1-A3 together with later modules elsewhere in the
    # same document. Explicit product-stage evidence therefore takes precedence
    # over the mere presence of A4/C/D terminology in another table/section.
    if "a1-a3" in compact or bool(re.search(r"\ba1\s*(?:-|to)\s*a3\b", text)):
        return True
    if all(label in text for label in ("a1", "a2", "a3")):
        return True
    if any(term in text for term in (
        "cradle-to-gate", "cradle to gate", "product stage", "product-stage",
        "production stage", "manufacturing stage", "factory gate",
    )):
        return True
    return False


def _source_indicator_supported(material: str, candidate: dict[str, Any]) -> bool:
    text = _candidate_evidence_text(candidate).lower()
    text = text.replace("–", "-").replace("—", "-")
    if _requires_no_biogenic_storage(material):
        return any(term in text for term in (
            "gwp-ghg", "gwp ghg", "gwp-fossil", "gwp fossil",
            "climate change - fossil", "climate change fossil",
        ))
    return any(term in text for term in (
        "gwp-total", "gwp total", "global warming potential", "gwp",
        "climate change - total", "climate change total", "climate change",
    ))


def _provisional_source_allowed(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Minimum source guard for relaxed-phase Class-3 evidence.

    This deliberately does not require a verified EPD/program host. It does
    require a real retrievable host and rejects known low-quality/community
    sources. The numerical value still has to be copied from the source and pass
    product/unit/boundary checks before it can contribute to a relaxed-phase evidence median.
    """
    host = _url_host(candidate.get("url", ""))
    if not host:
        return False, "UNKNOWN_SOURCE"
    if any(bad in host for bad in _LOW_QUALITY_HOSTS):
        return False, "LOW_QUALITY_WEB"
    strict_ok, strict_class = _source_quality(candidate)
    if strict_ok:
        return True, strict_class
    return True, "SOURCE_SUPPORTED_WEB"



def _promising_class4_candidate(material: str, candidate: dict[str, Any]) -> tuple[bool, str]:
    """Detect GWP evidence worth deeper relaxed-phase Class-3 extraction.

    Geography may differ. A declaration is not rejected merely because it also
    reports A4/C/D modules; explicit A1-A3/product-stage GWP evidence can still be
    isolated. Broader-life-cycle wording blocks only the looser production-context
    pathway where no explicit product-stage result is present.
    """
    source_ok, source_class = _provisional_source_allowed(candidate)
    if not source_ok:
        return False, f"source_not_allowed:{source_class}"
    identity_ok, identity_reason = _product_identity_ok(material, candidate)
    if not identity_ok:
        return False, f"identity_not_supported:{identity_reason}"
    text = _candidate_evidence_text(candidate).lower().replace("–", "-").replace("—", "-")
    climate = any(x in text for x in ("gwp", "global warming", "climate change"))
    product_stage = any(x in text for x in (
        "a1-a3", "a1 a3", "cradle-to-gate", "cradle to gate", "product stage",
        "product-stage", "production stage", "manufacturing stage", "factory gate"
    )) or all(x in text for x in ("a1", "a2", "a3"))
    production_context = any(x in text for x in (
        "manufacturing", "manufacture of", "manufactured", "production of",
        "material production", "production process", "factory production",
        "embodied carbon", "product carbon footprint", "environmental impact of production"
    ))
    broader = any(x in text for x in (
        "cradle-to-grave", "cradle to grave", "whole life", "whole-life",
        "a1-a5", "a1 to a5", "a1-c", "a1 to c", "end of life", "end-of-life"
    ))
    declared_basis = any(x in text for x in (
        "declared unit", "functional unit", "reference unit", "reference flow", "/kg", "/m2", "/m3",
        "per kg", "per m2", "per m3"
    ))
    if climate and product_stage:
        return True, "PROMISING_GWP_AND_EXPLICIT_PRODUCT_STAGE" + ("_WITH_DECLARED_BASIS" if declared_basis else "")
    if climate and production_context and not broader:
        return True, "PROMISING_GWP_AND_PRODUCTION_CONTEXT" + ("_WITH_DECLARED_BASIS" if declared_basis else "")
    if broader and not product_stage:
        return False, "broader_lifecycle_boundary_without_isolatable_product_stage"
    return False, "insufficient_relaxed_class3_gwp_production_signals"


def _provisional_boundary_supported(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Relaxed-phase Class-3 boundary gate for an extracted GWP result.

    Explicit A1-A3/product-stage evidence wins even if the declaration also
    reports downstream modules elsewhere. Broader-life-cycle wording is a hard
    rejection only when the relaxed External Verified phase relies on an inferred production context
    rather than an isolatable product-stage result.
    """
    text = _candidate_evidence_text(candidate).lower()
    text = text.replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", text)
    if "a1-a3" in compact or re.search(r"\ba1\s*(?:-|to)\s*a3\b", text):
        return True, "EXPLICIT_A1_A3"
    if all(label in text for label in ("a1", "a2", "a3")):
        return True, "EXPLICIT_A1_A2_A3_MODULES"
    if any(term in text for term in (
        "cradle-to-gate", "cradle to gate", "product stage", "product-stage",
        "production stage", "manufacturing stage", "factory gate"
    )):
        return True, "PRODUCT_STAGE_CRADLE_TO_GATE"
    broader = any(term in text for term in (
        "cradle-to-grave", "cradle to grave", "whole life", "whole-life",
        "a1-a5", "a1 to a5", "a1-c", "a1 to c", "end of life", "end-of-life"
    ))
    climate = any(term in text for term in ("gwp", "global warming", "climate change"))
    production_context = any(term in text for term in (
        "manufacturing", "manufacture of", "manufactured", "production of",
        "material production", "production process", "factory production",
        "embodied carbon", "product carbon footprint", "environmental impact of production"
    ))
    if climate and production_context and not broader:
        return True, "PRODUCTION_CONTEXT_INFERRED_RELAXED"
    if broader:
        return False, "BROADER_LIFECYCLE_WITHOUT_ISOLATABLE_PRODUCT_STAGE"
    return False, "PRODUCT_STAGE_OR_PRODUCTION_CONTEXT_NOT_SUPPORTED"


def _provisional_indicator_supported(material: str, candidate: dict[str, Any]) -> bool:
    text = _candidate_evidence_text(candidate).lower().replace("–", "-").replace("—", "-")
    if _requires_no_biogenic_storage(material):
        if any(term in text for term in (
            "gwp-ghg", "gwp ghg", "gwp-fossil", "gwp fossil",
            "climate change - fossil", "climate change fossil",
        )):
            return True
        # Older declarations may publish generic GWP plus a distinct biogenic row.
        return any(term in text for term in ("gwp", "global warming", "climate change")) and (
            "gwpbio" in re.sub(r"[\s_-]+", "", text)
            or "gwpbiogenic" in re.sub(r"[\s_-]+", "", text)
            or "with biogenic co2" in text
            or "w/ biogenic co2" in text
        )
    return any(term in text for term in (
        "gwp", "global warming", "climate change",
    ))


def _canonicalize_provisional_factor(value: float, reference_unit: str):
    return canonical_factor_basis(value, reference_unit)


def _number_close_in_text(value: float, text: str) -> bool:
    for token in re.findall(_SCI_NUM, text or ""):
        parsed = _parse_decimal_token(token)
        if parsed is None:
            continue
        tol = max(1e-9, abs(value) * 1e-6)
        if abs(parsed - value) <= tol:
            return True
    return False


def _gwp_quote_supports_values(
    quote: str, values: list[float], material: str, source_text: str = "", indicator: str = ""
) -> bool:
    """Require the copied evidence to link each number to the allowed GWP row.

    For table excerpts containing several climate-change component rows, the
    numeric value is checked on its own line whenever possible. This prevents an
    extractor from labelling a biogenic/fossil/LULUC component as GWP-total just
    because a different total-GWP row exists elsewhere in the same table block.
    """
    q = _clean(quote)
    if not q or not _has_gwp_signal(q):
        return False
    if indicator and not _gwp_indicator_ok(indicator, material, source_text):
        return False
    for value in values:
        if not _number_close_in_text(float(value), q):
            return False
        supporting = [
            line for line in q.splitlines()
            if _has_gwp_signal(line) and _number_close_in_text(float(value), line)
        ]
        if supporting and not any(_gwp_indicator_ok(line, material, source_text) for line in supporting):
            return False
    if _requires_no_biogenic_storage(material) and not indicator:
        low = _normalize_text(q).replace("–", "-").replace("—", "-")
        if not any(term in low for term in _GWP_FOSSIL_SIGNALS):
            if not _biogenic_generic_gwp_is_separate_from_storage(q, source_text):
                return False
    return True


def _declared_value_supported(quantity: float, unit: str, candidate: dict[str, Any]) -> bool:
    text = _candidate_evidence_text(candidate)
    normalized = text.replace("m³", "m3").replace("m²", "m2")
    ref = _normalize_declared_unit(unit)
    aliases = {
        "kg": ("kg",), "g": ("g",), "t": ("t", "tonne", "ton"),
        "m3": ("m3", "m^3"), "m2": ("m2", "m^2"),
        "item": ("item", "piece", "each", "unit"),
    }.get(ref, (ref,))
    for m in re.finditer(rf"({_SCI_NUM})\s*([A-Za-z0-9/^³²]+)", normalized, flags=re.I):
        q = _parse_decimal_token(m.group(1))
        if q is None:
            continue
        u = _normalize_declared_unit(m.group(2))
        if u == ref or m.group(2).lower() in aliases:
            tol = max(1e-9, abs(quantity) * 1e-6)
            if abs(q - quantity) <= tol:
                return True
    # A published intensity is often written directly as kg CO2e/kg, kg CO2e/m2,
    # kg CO2e per m3, etc., with no explicit "1 kg" declared quantity. For the
    # common declared_quantity=1 case, accept that denominator notation.
    low = normalized.lower().replace("co₂", "co2")
    if abs(float(quantity) - 1.0) <= 1e-9:
        denominator_patterns = {
            "kg": (r"/\s*kg\b", r"per\s+kg\b", r"kg\s*co2e?\s*/\s*kg\b"),
            "g": (r"/\s*g\b", r"per\s+g\b"),
            "t": (r"/\s*(?:t|tonne|ton)\b", r"per\s+(?:t|tonne|ton)\b"),
            "m3": (r"/\s*m3\b", r"per\s+m3\b"),
            "m2": (r"/\s*m2\b", r"per\s+m2\b"),
            "item": (r"/\s*(?:item|piece|unit)\b", r"per\s+(?:item|piece|unit)\b"),
        }.get(ref, ())
        if any(re.search(p, low) for p in denominator_patterns):
            return True
    # Tables may put the unit before the numeric value. Require both the exact
    # quantity and a recognized unit somewhere in the retrieved evidence.
    return _number_close_in_text(quantity, normalized) and any(a.lower() in low for a in aliases)


def _impact_value_supported(value: float, unit: str, candidate: dict[str, Any]) -> bool:
    text = _candidate_evidence_text(candidate)
    normalized = (text.replace("CO₂", "CO2").replace("co₂", "co2")
                      .replace("–", "-").replace("—", "-"))
    if not _number_close_in_text(value, normalized):
        return False
    u = _normalize_text(unit).replace("co₂", "co2")
    low = normalized.lower()
    if "co2" not in u:
        return False
    if "kg" in u:
        return bool(re.search(r"\bkg\s*co2", low))
    if "g" in u and "kg" not in u:
        return bool(re.search(r"(?:^|\W)g\s*co2", low))
    if "tonne" in u or re.search(r"\bt\b", u):
        return "tonne co2" in low or bool(re.search(r"\bt\s*co2", low))
    return False


def _source_year_supported(year: str, candidate: dict[str, Any]) -> str:
    y = _clean(year)
    if not y:
        return ""
    if not re.fullmatch(r"20\d{2}", y):
        return ""
    return y if y in _candidate_evidence_text(candidate) else ""


def _declared_pair_context(quantity: float, unit: str, candidate: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", _candidate_evidence_text(candidate)).strip()
    if not text:
        return ""
    # Find the first occurrence of the quantity token; the complete excerpt is
    # already retained separately, so this is only a compact audit aid.
    for m in re.finditer(_SCI_NUM, text):
        parsed = _parse_decimal_token(m.group(0))
        if parsed is None:
            continue
        tol = max(1e-9, abs(quantity) * 1e-6)
        if abs(parsed - quantity) <= tol:
            a=max(0,m.start()-120); b=min(len(text),m.end()+180)
            return text[a:b]
    return ""


def _query_subject(candidate: dict[str, Any]) -> str:
    q = _clean(candidate.get("query"))
    quoted = re.findall(r'"([^"]+)"', q)
    for phrase in quoted:
        p = _normalize_text(phrase)
        if p in {"a1-a3", "gwp-total", "kg co2", "kg co2e"}:
            continue
        return phrase
    return ""


def _product_identity_ok(material: str, candidate: dict[str, Any]) -> tuple[bool, str]:
    """Conservative external-source product identity check.

    The source title is checked first.  For cement, an additional evidence guard
    rejects sources that are actually declarations for ready-mix/precast
    concrete even when the manufacturer/company name contains the word
    ``cement``.  This directly prevents a concrete EPD from being used as a
    cement factor.
    """
    match_type = _clean(candidate.get("match_type")) or "DIRECT_PRODUCT"
    title = _clean(candidate.get("title"))
    ok, reason = external_title_compatibility(
        material,
        title,
        match_type=match_type,
        proxy_subject=_query_subject(candidate),
    )
    if not ok:
        # EPD portals often expose a generic browser title even though the PDF
        # itself begins with the actual declared product name. Only when the
        # title is genuinely generic do we allow the retrieved source text to
        # provide the identity label; a clearly different titled product is not
        # overridden by this fallback.
        generic_title = (
            not title
            or _normalize_text(title) in {"environmental product declaration", "epd", "product declaration"}
            or _normalize_text(title).startswith("environmental product declaration -")
        )
        if generic_title:
            evidence = re.sub(r"\s+", " ", _clean(candidate.get("excerpt")))
            # Use context around a literal material-name token when possible, so
            # an unrelated generic EPD is not accepted merely because its tables
            # contain generic construction vocabulary.
            tokens = sorted(
                {t for t in re.findall(r"[A-Za-z]{3,}", _clean(material).lower()) if t not in {"local", "commercial"}},
                key=len, reverse=True,
            )
            context = ""
            low_evidence = evidence.lower()
            for token in tokens:
                pos = low_evidence.find(token)
                if pos >= 0:
                    context = evidence[max(0, pos - 300): min(len(evidence), pos + 700)]
                    break
            evidence_label = context or evidence[:1000]
            ok2, reason2 = external_title_compatibility(
                material,
                evidence_label,
                match_type=match_type,
                proxy_subject=_query_subject(candidate),
            )
            if ok2:
                ok, reason = True, "generic_epd_title_source_text_" + reason2
        if not ok:
            return ok, reason

    family = classify_material(material)
    if family == "CEMENT" and match_type == "DIRECT_PRODUCT":
        # Use only strong product-identity phrases here; generic discussion of
        # concrete in a valid cement EPD is not enough to trigger rejection.
        evidence = _normalize_text(
            " ".join([
                _clean(candidate.get("title")),
                _clean(candidate.get("snippet")),
                _clean(candidate.get("excerpt"))[:1800],
            ])
        )
        downstream = (
            "ready mix concrete", "ready mixed concrete", "ready mix concrete rmc",
            "precast concrete", "pre cast concrete", "concrete block",
            "masonry block", "environmental product declaration of ready mixed concrete",
            "declared unit for the epd is 1m3 of ready mix concrete",
            "1 m3 of ready mix concrete",
        )
        if any(term in evidence for term in downstream):
            return False, "cement_source_declares_downstream_concrete_product"
    return True, reason


def _reference_unit_compatible(material: str, reference_unit: str) -> tuple[bool, str]:
    """High-specificity unit-family guard for external factors.

    This is intentionally conservative and only enforces families whose
    declared-unit expectations are strong enough to detect a wrong product
    without guessing a conversion.
    """
    family = classify_material(material)
    ref = _normalize_declared_unit(reference_unit)
    if family == "CEMENT" and ref not in {"kg", "t"}:
        return False, "cement_external_factor_requires_mass_reference_unit"
    if family in {"REBAR", "STEEL_WIRE", "STEEL_FASTENER", "GALVANIZED_FLAT_STEEL"} and ref not in {"kg", "t"}:
        return False, "metal_product_external_factor_requires_mass_reference_unit"
    return True, "compatible"


class ExternalEFResolver:
    def __init__(
        self,
        matcher,
        *,
        factor_snapshot: dict[str, Any] | None = None,
        target_geography: str = TARGET_GEOGRAPHY,
        allow_source_supported_provisional: bool = True,
        allow_llm_unverified_estimate: bool = True,
        allow_conservative_analog_estimate: bool = True,
        search_results_per_query: int = 5,
        extract_top_n: int = 5,
        max_evidence_candidates: int = 5,
        excerpt_chars: int = 6000,
        timeout: int = 8,
        class3_source_budget: int = 3,
        class4_total_source_budget: int = 10,
        max_search_queries_per_material: int = 6,
        max_external_seconds_per_material: float = 60.0,
        adaptive_max_search_queries_per_material: int = 12,
        adaptive_total_source_budget: int = 10,
        adaptive_external_seconds_per_material: float = 120.0,
        pdf_max_pages: int = 24,
        adaptive_pdf_max_pages: int = 60,
        max_download_bytes: int = 15 * 1024 * 1024,
        linked_documents_per_page: int = 1,
        adaptive_linked_documents_per_page: int = 2,
        progress_callback=None,
        evidence_cache_path: str | None = None,
    ):
        self.matcher = matcher
        self.factor_snapshot = factor_snapshot if isinstance(factor_snapshot, dict) else {}
        self.target_geography = target_geography
        self.allow_source_supported_provisional = bool(allow_source_supported_provisional)
        self.allow_llm_unverified_estimate = bool(allow_llm_unverified_estimate)
        self.allow_conservative_analog_estimate = bool(allow_conservative_analog_estimate)
        self.search_results_per_query = search_results_per_query
        self.extract_top_n = extract_top_n
        self.max_evidence_candidates = max_evidence_candidates
        self.excerpt_chars = excerpt_chars
        self.timeout = int(timeout)
        self.class3_source_budget = max(1, int(class3_source_budget))
        self.class4_total_source_budget = max(self.class3_source_budget, int(class4_total_source_budget))
        self.max_search_queries_per_material = max(1, int(max_search_queries_per_material))
        self.max_external_seconds_per_material = max(5.0, float(max_external_seconds_per_material))
        self.adaptive_max_search_queries_per_material = max(self.max_search_queries_per_material, int(adaptive_max_search_queries_per_material))
        self.adaptive_total_source_budget = max(self.class4_total_source_budget, int(adaptive_total_source_budget))
        self.adaptive_external_seconds_per_material = max(self.max_external_seconds_per_material, float(adaptive_external_seconds_per_material))
        self.pdf_max_pages = max(1, int(pdf_max_pages))
        self.adaptive_pdf_max_pages = max(self.pdf_max_pages, int(adaptive_pdf_max_pages))
        self.max_download_bytes = max(1024 * 1024, int(max_download_bytes))
        self.linked_documents_per_page = max(0, int(linked_documents_per_page))
        self.adaptive_linked_documents_per_page = max(self.linked_documents_per_page, int(adaptive_linked_documents_per_page))
        self.progress_callback = progress_callback
        self.evidence_cache = RuntimeEvidenceCache(evidence_cache_path)
        self._search_cache: dict[str, list[dict[str, Any]]] = {}
        self._resolution_cache: dict[str, dict[str, Any]] = {}
        self._last_fallback_attempts: list[dict[str, Any]] = []

    def _progress(self, message: str) -> None:
        if callable(self.progress_callback):
            try:
                self.progress_callback(message)
            except Exception:
                pass

    def _ddgs(self):
        from ddgs import DDGS
        return DDGS(timeout=self.timeout)

    def _search(self, query: str):
        if query in self._search_cache:
            return [dict(x) for x in self._search_cache[query]]
        rows = []
        for region in ("np-en", "us-en"):
            try:
                result = self._ddgs().text(
                    query, region=region, safesearch="moderate",
                    max_results=self.search_results_per_query, backend="auto",
                )
                rows = list(result or [])
                if rows:
                    break
            except Exception:
                rows = []
                continue
        out = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            url = _clean(r.get("href") or r.get("url"))
            if not url:
                continue
            out.append({
                "title": _clean(r.get("title")),
                "url": url,
                "snippet": _clean(r.get("body") or r.get("snippet")),
            })
        self._search_cache[query] = out
        return [dict(x) for x in out]

    def _extract(self, url: str):
        """Retrieve one bounded evidence document safely with GWP-focused PDF parsing.

        PDF parsing is attempted only when the bytes start with the PDF magic
        header. For PDFs, GWP-relevant pages are detected first and table structure
        is preserved with pdfplumber. No OCR and no material-specific values are used.
        """
        import requests

        request_timeout = (min(5, self.timeout), self.timeout)

        def pdf_text(content: bytes) -> str:
            if (not content or len(content) > self.max_download_bytes
                    or not content.lstrip().startswith(b"%PDF")):
                return ""
            return _extract_pdf_gwp_evidence(
                content,
                page_limit=self.pdf_max_pages,
                max_chars=24000 if self.pdf_max_pages <= 24 else 32000,
            )

        try:
            resp = requests.get(
                url, timeout=request_timeout, allow_redirects=True, stream=False,
                headers={"User-Agent": "Mozilla/5.0 LCA-research-resolver/5.0"},
            )
            resp.raise_for_status()
            if len(resp.content) > self.max_download_bytes:
                return ""
            ctype = (resp.headers.get("content-type") or "").lower()

            if resp.content.lstrip().startswith(b"%PDF"):
                return pdf_text(resp.content)

            is_html = (
                "html" in ctype
                or resp.content.lstrip()[:100].lower().startswith((b"<!doctype", b"<html"))
                or b"<html" in resp.content[:3000].lower()
            )
            if is_html:
                try:
                    raw_html = resp.text
                except Exception:
                    raw_html = resp.content.decode("utf-8", errors="ignore")
                plain = _strip_html(raw_html)
                parts = [_focus_gwp_lines(plain, max_chars=10000)] if plain else []

                if self.linked_documents_per_page > 0:
                    links = re.findall(r"href\s*=\s*['\"]([^'\"]+)['\"]", raw_html, flags=re.I)
                    document_urls = []
                    for href in links:
                        absolute = urljoin(resp.url, href)
                        low = absolute.lower()
                        if ".pdf" in low or "download" in low or "declaration" in low or "epd" in low:
                            if absolute not in document_urls:
                                document_urls.append(absolute)
                        if len(document_urls) >= self.linked_documents_per_page:
                            break
                    for doc_url in document_urls:
                        try:
                            r2 = requests.get(
                                doc_url, timeout=request_timeout, allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0 LCA-research-resolver/5.0"},
                            )
                            r2.raise_for_status()
                            if len(r2.content) <= self.max_download_bytes and r2.content.lstrip().startswith(b"%PDF"):
                                t = pdf_text(r2.content)
                                if t:
                                    parts.append(t)
                        except Exception:
                            continue
                combined = "\n\n... LINKED_DOCUMENT ...\n\n".join(x for x in parts if x)
                if combined:
                    return combined[:32000]
        except Exception:
            pass

        # One compact extraction fallback only; no recursive crawling.
        try:
            extracted = self._ddgs().extract(url, fmt="text_rich")
            if isinstance(extracted, dict):
                raw = extracted.get("content") or extracted.get("text") or json.dumps(extracted, ensure_ascii=False)
            else:
                raw = str(extracted)
            return _focus_gwp_lines(raw, max_chars=16000)
        except Exception:
            return ""

    def _collect_evidence(
        self, queries, tier, item_id, material, match_type="DIRECT_PRODUCT", proxy_basis=None,
        *, max_candidates: int | None = None, max_queries: int | None = None,
        seen_urls: set[str] | None = None,
    ):
        """Collect a bounded set of unique candidate sources.

        The budget applies to the material as a whole when the caller shares
        ``seen_urls``. Search results are ranked generically by source quality
        before document extraction; no material-specific source is preselected.
        """
        seen = seen_urls if seen_urls is not None else set()
        limit = max(1, int(max_candidates or self.max_evidence_candidates))
        qlimit = max(1, int(max_queries or len(list(queries))))
        pool = []
        queries_used = 0
        for query in list(queries)[:qlimit]:
            queries_used += 1
            for r in self._search(query):
                if r["url"] in seen or any(x["url"] == r["url"] for x in pool):
                    continue
                pool.append({
                    **r, "query": query, "tier": tier, "match_type": match_type,
                    "proxy_basis": proxy_basis,
                })
            if len(pool) >= limit * 2:
                break

        def rank(c):
            ok, cls = _source_quality(c)
            class_rank = {
                "VERIFIED_EPD_PROGRAM": 0, "NEPAL_GOVERNMENT": 1,
                "GOVERNMENT": 2, "TECHNICAL_LCA_DATABASE": 3,
                "PEER_REVIEWED_PUBLICATION": 4, "ACADEMIC": 5,
                "MANUFACTURER_EPD": 6,
            }.get(cls, 20)
            return (0 if ok else 1, class_rank)

        pool.sort(key=rank)
        candidates = pool[:limit]
        rows = []
        for i, c in enumerate(candidates):
            seen.add(c["url"])
            c["result_id"] = f"S{i+1}"
            self._progress(f"External Verified · inspecting source {i+1}/{len(candidates)} · {material}")
            c["full_text"] = self._extract(c["url"])
            c["excerpt"] = c["full_text"][: max(self.excerpt_chars, 2500)]
            identity_ok, identity_reason = _product_identity_ok(material, c)
            rows.append({
                "item_id": item_id, "material": material, "search_tier": tier,
                "match_type": match_type, "proxy_basis": proxy_basis,
                "query": c["query"], "query_subject": _query_subject(c),
                "result_id": c["result_id"], "title": c["title"], "url": c["url"],
                "snippet": c["snippet"], "excerpt": c["excerpt"],
                "product_identity_ok": identity_ok, "product_identity_reason": identity_reason,
                "gwp_table_detected": "[PDF TABLE PAGE" in c["full_text"],
                "gwp_page_detected": "[PDF PAGE" in c["full_text"],
                "relaxed_extraction_status": None, "relaxed_extraction_method": None, "relaxed_rejection_reason": None,
                "selected": False, "uncertainty_peer": False,
                "relaxed_candidate": False, "relaxed_retained": False,
                "relaxed_outlier": False, "relaxed_reason": None,
                "relaxed_normalized_ef": None, "relaxed_reference_unit": None,
            })
        return candidates, rows, queries_used

    def _traceable_from_deterministic(self, material, tier, selected, rec):
        """Validate a deterministic GWP-table extraction for strict Class 3."""
        source_ok, source_class = _source_quality(selected)
        if not source_ok:
            return None, f"source_quality_rejected:{source_class}"
        identity_ok, identity_reason = _product_identity_ok(material, selected)
        if not identity_ok:
            return None, f"product_identity_rejected:{identity_reason}"

        quote = _clean(rec.get("evidence_quote"))
        if not _quote_supported(quote, selected):
            return None, "deterministic_evidence_quote_not_in_retrieved_source"
        indicator = _clean(rec.get("indicator"))
        if not _gwp_indicator_ok(indicator, material, _candidate_evidence_text(selected)):
            return None, "deterministic_indicator_not_allowed_by_reporting_policy"
        if not _source_indicator_supported(material, selected):
            return None, "indicator_not_supported_by_retrieved_source"
        if not _source_boundary_supported(selected):
            return None, "a1_a3_boundary_not_supported_by_retrieved_source"

        dq = _num(rec.get("declared_quantity"))
        declared_unit = _clean(rec.get("declared_unit"))
        if dq is None or dq <= 0 or not _declared_value_supported(dq, declared_unit, selected):
            return None, "declared_quantity_or_unit_not_supported_by_retrieved_source"

        impact_unit = _clean(rec.get("impact_unit"))
        mode = _clean(rec.get("value_mode")).upper()
        calculation_basis = None
        if mode == "TOTAL":
            iv = _num(rec.get("impact_value"))
            if iv is None or not _impact_value_supported(iv, impact_unit, selected):
                return None, "deterministic_impact_value_or_unit_not_supported_by_source"
            if not _gwp_quote_supports_values(quote, [iv], material, _candidate_evidence_text(selected), indicator):
                return None, "deterministic_quote_does_not_link_total_to_gwp"
            calculation_basis = "DETERMINISTIC_EXPLICIT_SOURCE_A1_A3_TOTAL"
        elif mode == "SUM_A1_A2_A3":
            vals = [_num(rec.get(k)) for k in ("a1_value", "a2_value", "a3_value")]
            if any(v is None for v in vals):
                return None, "deterministic_a1_a2_a3_incomplete"
            text = _candidate_evidence_text(selected).lower().replace("–", "-").replace("—", "-")
            if not all(label in text for label in ("a1", "a2", "a3")):
                return None, "deterministic_a1_a2_a3_labels_not_supported"
            if not all(_impact_value_supported(float(v), impact_unit, selected) for v in vals):
                return None, "deterministic_module_value_or_unit_not_supported"
            if not _gwp_quote_supports_values(quote, [float(v) for v in vals], material, _candidate_evidence_text(selected), indicator):
                return None, "deterministic_quote_does_not_link_modules_to_gwp"
            iv = float(sum(float(v) for v in vals))
            calculation_basis = "PYTHON_SUM_OF_EXPLICIT_SOURCE_A1_A2_A3"
        else:
            return None, "deterministic_unsupported_value_mode"

        norm = normalize_factor(iv, impact_unit, dq, declared_unit)
        if norm is None:
            return None, "impact_or_reference_unit_invalid"
        ef_value, ref_unit = norm
        unit_ok, unit_reason = _reference_unit_compatible(material, ref_unit)
        if not unit_ok:
            return None, unit_reason
        if _requires_no_biogenic_storage(material) and ef_value <= 0:
            return None, "biogenic_storage_excluded_requires_positive_factor"
        plausible, plausibility_reason = emission_factor_plausible(material, ef_value, ref_unit)
        if not plausible:
            return None, f"factor_plausibility_rejected:{plausibility_reason}"

        geography = _infer_geography(selected)
        if not _target_geography_supported(selected, self.target_geography):
            return None, f"target_geography_not_verified:{geography or 'Unspecified'}"
        if _clean(selected.get("match_type")) not in {"", "DIRECT_PRODUCT"}:
            return None, "class3_requires_direct_product_evidence"

        selected["class3_extraction_method"] = _clean(rec.get("extraction_method")) or "DETERMINISTIC_GWP_TABLE"
        return {
            "ef_value": ef_value,
            "reference_unit": ref_unit,
            "lower_value": None,
            "upper_value": None,
            "verification": "EXTERNAL_VERIFIED",
            "verification_tier": "STRICT",
            "source_class": source_class,
            "source_title": selected["title"],
            "source_url": selected["url"],
            "source_geography": geography,
            "source_year": _source_year_supported(_clean(rec.get("source_year")), selected),
            "search_tier": tier,
            "search_query": selected["query"],
            "evidence_quote": quote,
            "declared_unit_evidence": _declared_pair_context(dq, declared_unit, selected),
            "boundary": _clean(rec.get("boundary")) or "A1-A3",
            "indicator": indicator,
            "declared_impact_value": iv,
            "declared_impact_unit": impact_unit,
            "declared_quantity": dq,
            "declared_unit": declared_unit,
            "uncertainty_type": None,
            "uncertainty_lower_value": None,
            "uncertainty_upper_value": None,
            "uncertainty_gsd": None,
            "uncertainty_cv": None,
            "uncertainty_confidence_level": None,
            "uncertainty_evidence_quote": None,
            "reason": (_clean(rec.get("reason")) + f" [{calculation_basis}]").strip(),
            "calculation_basis": calculation_basis,
            "extraction_method": selected["class3_extraction_method"],
            "raw_model_output": None,
            "selected_result_id": selected.get("result_id"),
            "external_match_type": selected.get("match_type", "DIRECT_PRODUCT"),
            "external_proxy_basis": selected.get("proxy_basis"),
            "product_identity_status": "PASS",
            "product_identity_reason": identity_reason,
            "product_identity_title": selected.get("title"),
            "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
            "proxy_representativeness": "DIRECT_TARGET_GEOGRAPHY",
        }, "ok"

    def _choose_traceable(self, material, tier, candidates):
        if not candidates:
            return None, "no_candidates"

        # Deterministic structured-table extraction is attempted before Qwen.
        # A foreign/proxy source that fails strict Class 3 remains available to
        # the independent Class-4 pathway later.
        deterministic_reasons = []
        for selected in candidates:
            rec, det_reason = _deterministic_gwp_table_record(material, selected)
            if rec is None:
                deterministic_reasons.append(f"{selected.get('result_id')}:{det_reason}")
                continue
            out, reason = self._traceable_from_deterministic(material, tier, selected, rec)
            selected["class3_deterministic_status"] = "ACCEPTED" if out is not None else "REJECTED"
            selected["class3_deterministic_reason"] = reason
            if out is not None:
                return out, "ok"
            deterministic_reasons.append(f"{selected.get('result_id')}:{reason}")

        payload = {
            "material": material,
            "target_geography": self.target_geography,
            "search_tier": tier,
            "required_indicator": _required_indicator_policy(material),
            "required_boundary": "A1-A3",
            "evidence_candidates": [
                {
                    "result_id": c["result_id"],
                    "title": c["title"],
                    "url": c["url"],
                    "search_snippet": c["snippet"][:700],
                    "extracted_text": c["excerpt"][: self.excerpt_chars],
                }
                for c in candidates
            ],
        }
        raw = _call_matcher(self.matcher, TRACEABLE_EF_SYSTEM_PROMPT, payload, max_new_tokens=768)
        try:
            obj = json.loads(raw.strip())
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                return None, "parse_error"
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None, "parse_error"

        required = {
            "found", "impact_value", "impact_unit", "declared_quantity", "declared_unit",
            "boundary", "indicator", "source_result_id", "source_year", "evidence_quote",
            "uncertainty_type", "uncertainty_lower_value", "uncertainty_upper_value",
            "uncertainty_gsd", "uncertainty_cv", "uncertainty_confidence_level",
            "uncertainty_evidence_quote", "reason",
        }
        if set(obj.keys()) != required:
            return None, "schema_invalid"
        if not bool(obj.get("found")):
            return None, _clean(obj.get("reason")) or "not_found"

        sid = _clean(obj.get("source_result_id"))
        by_id = {c["result_id"]: c for c in candidates}
        if sid not in by_id:
            return None, "source_id_invalid"
        selected = by_id[sid]

        source_ok, source_class = _source_quality(selected)
        if not source_ok:
            return None, f"source_quality_rejected:{source_class}"

        identity_ok, identity_reason = _product_identity_ok(material, selected)
        if not identity_ok:
            return None, f"product_identity_rejected:{identity_reason}"

        quote = _clean(obj.get("evidence_quote"))
        if not _quote_supported(quote, selected):
            return None, "evidence_quote_not_in_retrieved_source"
        if not _gwp_indicator_ok(_clean(obj.get("indicator")), material, _candidate_evidence_text(selected)):
            return None, "indicator_not_allowed_by_reporting_policy"
        if not _source_indicator_supported(material, selected):
            return None, "indicator_not_supported_by_retrieved_source"
        if not _boundary_ok(_clean(obj.get("boundary"))):
            return None, "boundary_not_a1_a3"
        if not _source_boundary_supported(selected):
            return None, "a1_a3_boundary_not_supported_by_retrieved_source"

        iv = _num(obj.get("impact_value"))
        dq = _num(obj.get("declared_quantity"))
        if iv is None or dq is None or dq <= 0:
            return None, "value_or_declared_quantity_invalid"
        if not _impact_value_supported(iv, _clean(obj.get("impact_unit")), selected):
            return None, "impact_value_or_unit_not_supported_by_retrieved_source"
        if not _declared_value_supported(dq, _clean(obj.get("declared_unit")), selected):
            return None, "declared_quantity_or_unit_not_supported_by_retrieved_source"
        norm = normalize_factor(iv, _clean(obj.get("impact_unit")), dq, _clean(obj.get("declared_unit")))
        if norm is None:
            return None, "impact_or_reference_unit_invalid"
        ef_value, ref_unit = norm
        unit_ok, unit_reason = _reference_unit_compatible(material, ref_unit)
        if not unit_ok:
            return None, unit_reason
        if _requires_no_biogenic_storage(material) and ef_value <= 0:
            return None, "biogenic_storage_excluded_requires_positive_factor"
        plausible, plausibility_reason = emission_factor_plausible(material, ef_value, ref_unit)
        if not plausible:
            return None, f"factor_plausibility_rejected:{plausibility_reason}"

        geography = _infer_geography(selected)

        # CLASS 3 IS STRICT: the source itself must explicitly support the target
        # study geography. Global, regional, foreign, and geography-unspecified
        # sources are not promoted to EXTERNAL_VERIFIED; they remain eligible for
        # the relaxed External Verified screening phase below.
        if not _target_geography_supported(selected, self.target_geography):
            return None, f"target_geography_not_verified:{geography or 'Unspecified'}"

        # Class 3 is also reserved for direct-product evidence. A broader
        # taxonomy-family/source proxy is still usable, but only in the relaxed External Verified phase.
        if _clean(selected.get("match_type")) not in {"", "DIRECT_PRODUCT"}:
            return None, "class3_requires_direct_product_evidence"

        # Optional source-reported quantitative uncertainty. It is accepted only
        # when the model copied an uncertainty quote from the retrieved evidence
        # and every non-null numeric field is literally present in that evidence.
        uncertainty_type = _clean(obj.get("uncertainty_type"))
        uncertainty_lower = _num(obj.get("uncertainty_lower_value"))
        uncertainty_upper = _num(obj.get("uncertainty_upper_value"))
        uncertainty_gsd = _num(obj.get("uncertainty_gsd"))
        uncertainty_cv = _num(obj.get("uncertainty_cv"))
        uncertainty_conf = _num(obj.get("uncertainty_confidence_level"))
        uncertainty_quote = _clean(obj.get("uncertainty_evidence_quote"))
        uncertainty_values = [x for x in (uncertainty_lower, uncertainty_upper, uncertainty_gsd, uncertainty_cv, uncertainty_conf) if x is not None]
        if uncertainty_values:
            if not uncertainty_quote or not _quote_supported(uncertainty_quote, selected):
                return None, "uncertainty_quote_not_in_retrieved_source"
            evidence_text = _candidate_evidence_text(selected)
            for val in uncertainty_values:
                if not _number_close_in_text(val, evidence_text):
                    return None, "uncertainty_value_not_supported_by_retrieved_source"
            if uncertainty_gsd is not None and uncertainty_gsd < 1:
                return None, "source_reported_gsd_less_than_one"
            if uncertainty_lower is not None and uncertainty_upper is not None and uncertainty_lower >= uncertainty_upper:
                return None, "source_reported_uncertainty_interval_invalid"
        else:
            uncertainty_type = ""
            uncertainty_quote = ""

        return {
            "ef_value": ef_value,
            "reference_unit": ref_unit,
            "lower_value": None,
            "upper_value": None,
            "verification": "EXTERNAL_VERIFIED",
            "verification_tier": "STRICT",
            "source_class": source_class,
            "source_title": selected["title"],
            "source_url": selected["url"],
            "source_geography": geography,
            "source_year": _source_year_supported(_clean(obj.get("source_year")), selected),
            "search_tier": tier,
            "search_query": selected["query"],
            "evidence_quote": quote,
            "declared_unit_evidence": _declared_pair_context(dq, _clean(obj.get("declared_unit")), selected),
            "boundary": _clean(obj.get("boundary")),
            "indicator": _clean(obj.get("indicator")),
            "declared_impact_value": iv,
            "declared_impact_unit": _clean(obj.get("impact_unit")),
            "declared_quantity": dq,
            "declared_unit": _clean(obj.get("declared_unit")),
            "uncertainty_type": uncertainty_type or None,
            "uncertainty_lower_value": uncertainty_lower,
            "uncertainty_upper_value": uncertainty_upper,
            "uncertainty_gsd": uncertainty_gsd,
            "uncertainty_cv": uncertainty_cv,
            "uncertainty_confidence_level": uncertainty_conf,
            "uncertainty_evidence_quote": uncertainty_quote or None,
            "reason": _clean(obj.get("reason")),
            "raw_model_output": raw,
            "selected_result_id": sid,
            "external_match_type": selected.get("match_type", "DIRECT_PRODUCT"),
            "external_proxy_basis": selected.get("proxy_basis"),
            "product_identity_status": "PASS",
            "product_identity_reason": identity_reason,
            "product_identity_title": selected.get("title"),
            "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
            "proxy_representativeness": "DIRECT_TARGET_GEOGRAPHY",
        }, "ok"

    def _collect_peer_factors(self, material, tier, candidates, primary, max_total: int = 3):
        """Collect a small same-tier set of independently accepted factors for uncertainty.

        These peers are never used to replace the primary selected factor. They
        are retained only to estimate empirical dispersion when two or more
        defensible factors with the same reference unit are available.
        """
        peers = [primary]
        remaining = [c for c in candidates if c.get("result_id") != primary.get("selected_result_id")]
        while remaining and len(peers) < max_total:
            alt, _ = self._choose_traceable(material, tier, remaining)
            if alt is None:
                break
            sid = alt.get("selected_result_id")
            remaining = [c for c in remaining if c.get("result_id") != sid]
            if alt.get("reference_unit") != primary.get("reference_unit"):
                continue
            val = _num(alt.get("ef_value"))
            if val is None or val <= 0:
                continue
            # Different URLs are required so duplicate search hits do not create
            # artificial evidence of variability.
            if any(_clean(x.get("source_url")) == _clean(alt.get("source_url")) for x in peers):
                continue
            peers.append(alt)
        return peers

    def _extract_relaxed_candidates(self, material, tier, candidates):
        """Extract relaxed-phase Class-3 GWP values with deterministic-table-first logic.

        Structured GWP/A1/A2/A3 tables are parsed by Python first. Qwen is used
        only when deterministic extraction is absent or fails validation.
        """
        if not candidates:
            return []

        accepted = []
        required = {
            "source_result_id", "found", "value_mode", "impact_value",
            "a1_value", "a2_value", "a3_value", "impact_unit",
            "declared_quantity", "declared_unit", "boundary", "indicator",
            "source_year", "evidence_quote", "reason",
        }

        for selected in candidates:
            selected["relaxed_extraction_status"] = "STARTED"
            selected["relaxed_rejection_reason"] = None
            selected["relaxed_extraction_method"] = None
            attempts: list[dict[str, Any]] = []

            def validate_record(rec: dict[str, Any], extraction_method: str):
                sid = _clean(rec.get("source_result_id"))
                if sid != selected.get("result_id"):
                    return None, "source_result_id_mismatch"
                if not bool(rec.get("found")):
                    return None, "gwp_not_extracted:" + (_clean(rec.get("reason")) or "not_found")

                source_ok, source_class = _provisional_source_allowed(selected)
                if not source_ok:
                    return None, f"source_not_allowed:{source_class}"
                identity_ok, identity_reason = _product_identity_ok(material, selected)
                if not identity_ok:
                    return None, f"product_identity_rejected:{identity_reason}"
                quote = _clean(rec.get("evidence_quote"))
                if not _quote_supported(quote, selected):
                    return None, "evidence_quote_not_in_retrieved_source"

                dq = _num(rec.get("declared_quantity"))
                if dq is None or dq <= 0:
                    return None, "declared_quantity_missing_or_invalid"
                declared_unit = _clean(rec.get("declared_unit"))
                if not _declared_value_supported(dq, declared_unit, selected):
                    return None, "declared_unit_or_quantity_not_supported_by_source"

                impact_unit = _clean(rec.get("impact_unit"))
                indicator = _clean(rec.get("indicator"))
                if not _gwp_indicator_ok(indicator, material, _candidate_evidence_text(selected)):
                    return None, "extracted_indicator_is_not_allowed_gwp_category"
                mode = _clean(rec.get("value_mode")).upper()
                calculation_basis = None
                if mode == "TOTAL":
                    iv = _num(rec.get("impact_value"))
                    if iv is None or not _impact_value_supported(iv, impact_unit, selected):
                        return None, "gwp_total_value_or_impact_unit_not_supported_by_source"
                    if not _gwp_quote_supports_values(quote, [iv], material, _candidate_evidence_text(selected), indicator):
                        return None, "evidence_quote_does_not_link_numeric_value_to_gwp"
                    calculation_basis = (
                        "DETERMINISTIC_EXPLICIT_SOURCE_A1_A3_TOTAL"
                        if extraction_method.startswith("DETERMINISTIC_")
                        else "EXPLICIT_SOURCE_GWP_TOTAL"
                    )
                elif mode == "SUM_A1_A2_A3":
                    vals = [_num(rec.get(k)) for k in ("a1_value", "a2_value", "a3_value")]
                    if any(v is None for v in vals):
                        return None, "a1_a2_a3_numeric_values_incomplete"
                    text = _candidate_evidence_text(selected).lower().replace("–", "-").replace("—", "-")
                    if not all(label in text for label in ("a1", "a2", "a3")):
                        return None, "a1_a2_a3_labels_not_supported_by_source"
                    if not all(_impact_value_supported(float(v), impact_unit, selected) for v in vals):
                        return None, "one_or_more_a1_a2_a3_values_not_supported_by_source"
                    if not _gwp_quote_supports_values(quote, [float(v) for v in vals], material, _candidate_evidence_text(selected), indicator):
                        return None, "evidence_quote_does_not_link_a1_a2_a3_values_to_gwp"
                    iv = float(sum(float(v) for v in vals))
                    calculation_basis = "PYTHON_SUM_OF_EXPLICIT_SOURCE_A1_A2_A3"
                else:
                    return None, "unsupported_value_mode"

                norm = normalize_factor(iv, impact_unit, dq, declared_unit)
                if norm is None:
                    return None, "factor_normalization_failed"
                ef_value, ref_unit = norm
                canonical = _canonicalize_provisional_factor(ef_value, ref_unit)
                if canonical is None:
                    return None, "canonical_factor_basis_failed"
                ef_value, ref_unit = canonical
                unit_ok, unit_reason = _reference_unit_compatible(material, ref_unit)
                if not unit_ok or ef_value <= 0:
                    return None, "reference_unit_incompatible:" + unit_reason

                boundary_ok, boundary_quality = _provisional_boundary_supported(selected)
                if not boundary_ok:
                    return None, "relaxed_boundary_rejected:" + boundary_quality
                if mode == "TOTAL" and not extraction_method.startswith("DETERMINISTIC_"):
                    if boundary_quality == "EXPLICIT_A1_A3":
                        calculation_basis = "EXPLICIT_SOURCE_A1_A3_TOTAL"
                    elif boundary_quality == "EXPLICIT_A1_A2_A3_MODULES":
                        calculation_basis = "EXPLICIT_SOURCE_PRODUCT_STAGE_GWP_TOTAL_WITH_MODULE_LABELS"
                    elif boundary_quality == "PRODUCT_STAGE_CRADLE_TO_GATE":
                        calculation_basis = "EXPLICIT_SOURCE_PRODUCT_STAGE_GWP_TOTAL"
                    elif boundary_quality == "PRODUCTION_CONTEXT_INFERRED_RELAXED":
                        calculation_basis = "EXPLICIT_SOURCE_GWP_WITH_PRODUCTION_CONTEXT_RELAXED"
                if not _provisional_indicator_supported(material, selected):
                    return None, "gwp_indicator_not_supported_by_retrieved_source"
                if _requires_no_biogenic_storage(material) and ef_value <= 0:
                    return None, "biogenic_policy_requires_positive_gwp_ghg_or_fossil"
                plausible, plausible_reason = emission_factor_plausible(material, ef_value, ref_unit)
                if not plausible:
                    return None, "guardrail_rejected:" + plausible_reason

                geography = _infer_geography(selected)
                out = {
                    "ef_value": float(ef_value),
                    "reference_unit": ref_unit,
                    "verification": "EXTERNAL_VERIFIED",
                    "verification_tier": "RELAXED",
                    "source_class": "EXTERNAL_VERIFIED_RELAXED",
                    "source_title": selected.get("title"),
                    "source_url": selected.get("url"),
                    "source_geography": geography,
                    "geography_representativeness": (
                        "TARGET_GEOGRAPHY" if _target_geography_supported(selected, self.target_geography)
                        else "NON_TARGET_OR_UNSPECIFIED_ACCEPTED_IN_RELAXED_PHASE"
                    ),
                    "source_year": _source_year_supported(_clean(rec.get("source_year")), selected),
                    "search_tier": tier,
                    "search_query": selected.get("query"),
                    "evidence_quote": quote,
                    "boundary": _clean(rec.get("boundary")) or boundary_quality,
                    "boundary_quality": boundary_quality,
                    "indicator": indicator,
                    "declared_impact_value": iv,
                    "declared_impact_unit": impact_unit,
                    "declared_quantity": dq,
                    "declared_unit": declared_unit,
                    "reason": (_clean(rec.get("reason")) + f" [{calculation_basis}]").strip(),
                    "calculation_basis": calculation_basis,
                    "extraction_method": extraction_method,
                    "raw_model_output": json.dumps(attempts, ensure_ascii=False),
                    "selected_result_id": sid,
                    "external_match_type": selected.get("match_type", "DIRECT_PRODUCT"),
                    "external_proxy_basis": selected.get("proxy_basis"),
                    "product_identity_status": "PASS_SOURCE_SUPPORTED",
                    "product_identity_reason": identity_reason,
                    "product_identity_title": selected.get("title"),
                    "gwp_table_detected": "[PDF TABLE PAGE" in _clean(selected.get("full_text")),
                    "gwp_page_detected": "[PDF PAGE" in _clean(selected.get("full_text")),
                    "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
                }
                return out, "ok"

            # Attempt 1: deterministic structured EPD/GWP table parser.
            det, det_reason = _deterministic_gwp_table_record(material, selected)
            if det is not None:
                method = _clean(det.pop("extraction_method", "")) or "DETERMINISTIC_GWP_TABLE"
                attempts.append({"method": method, "record": det})
                out, reason = validate_record(det, method)
                if out is not None:
                    selected["relaxed_extraction_status"] = "ACCEPTED"
                    selected["relaxed_extraction_method"] = method
                    selected["relaxed_rejection_reason"] = None
                    out["raw_model_output"] = json.dumps(attempts, ensure_ascii=False)
                    accepted.append(out)
                    continue
                attempts.append({"method": method, "validation_rejection": reason})
            else:
                attempts.append({"method": "DETERMINISTIC_GWP_TABLE", "status": det_reason})

            # Attempt 2: Qwen source extractor only when deterministic extraction
            # did not produce a validated factor.
            payload = {
                "material": material,
                "target_geography": self.target_geography,
                "search_tier": tier,
                "required_indicator": _required_indicator_policy(material),
                "indicator_scope": "GWP_ONLY",
                "evidence_candidates": [{
                    "result_id": selected["result_id"],
                    "title": selected["title"],
                    "url": selected["url"],
                    "search_snippet": selected["snippet"][:900],
                    "extracted_text": selected["excerpt"][: self.excerpt_chars],
                }],
            }
            raw = _call_matcher(self.matcher, RELAXED_EXTERNAL_EF_SYSTEM_PROMPT, payload, max_new_tokens=640)
            attempts.append({"method": "QWEN_EVIDENCE_EXTRACTOR", "raw": raw})
            try:
                obj = json.loads(raw.strip())
            except Exception:
                m = re.search(r"\{.*\}", raw or "", flags=re.S)
                if not m:
                    selected["relaxed_extraction_status"] = "REJECTED"
                    selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
                    selected["relaxed_rejection_reason"] = "model_json_parse_error_after_deterministic_fallback"
                    continue
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    selected["relaxed_extraction_status"] = "REJECTED"
                    selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
                    selected["relaxed_rejection_reason"] = "model_json_parse_error_after_deterministic_fallback"
                    continue
            if set(obj.keys()) != {"records"} or not isinstance(obj.get("records"), list) or len(obj["records"]) != 1:
                selected["relaxed_extraction_status"] = "REJECTED"
                selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
                selected["relaxed_rejection_reason"] = "model_schema_invalid"
                continue
            rec = obj["records"][0]
            if not isinstance(rec, dict) or set(rec.keys()) != required:
                selected["relaxed_extraction_status"] = "REJECTED"
                selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
                selected["relaxed_rejection_reason"] = "model_record_schema_invalid"
                continue
            out, reason = validate_record(rec, "QWEN_EVIDENCE_EXTRACTOR")
            if out is None:
                selected["relaxed_extraction_status"] = "REJECTED"
                selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
                selected["relaxed_rejection_reason"] = reason
                continue
            selected["relaxed_extraction_status"] = "ACCEPTED"
            selected["relaxed_extraction_method"] = "QWEN_EVIDENCE_EXTRACTOR"
            selected["relaxed_rejection_reason"] = None
            out["raw_model_output"] = json.dumps(attempts, ensure_ascii=False)
            accepted.append(out)

        return accepted

    def _build_provisional_consensus(self, material, factors):
        if not factors:
            return None

        # De-duplicate URLs so repeated search hits do not masquerade as
        # independent evidence. Prefer direct-product evidence whenever any
        # exists; taxonomy-family proxies are used only if no direct evidence is
        # source-supported.
        unique = []
        seen_urls = set()
        for f in factors:
            url = _clean(f.get("source_url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            unique.append(dict(f))
        if not unique:
            return None
        direct = [x for x in unique if _clean(x.get("external_match_type")) == "DIRECT_PRODUCT"]
        pool = direct if direct else unique

        # Values with different declared bases cannot be combined without a
        # physical conversion assumption. Select the best-supported common basis
        # by source count, then geography tier.
        tier_rank = {"NEPAL": 0, "SOUTH_ASIA": 1, "ASIA": 2, "GLOBAL": 3}
        groups: dict[str, list[dict[str, Any]]] = {}
        for f in pool:
            groups.setdefault(_clean(f.get("reference_unit")), []).append(f)
        ranked = sorted(
            groups.items(),
            key=lambda kv: (
                -len(kv[1]),
                min(tier_rank.get(_clean(x.get("search_tier")).upper(), 9) for x in kv[1]),
                kv[0],
            ),
        )
        ref_unit, group = ranked[0]
        values = [float(x["ef_value"]) for x in group]
        consensus = robust_positive_consensus(values)
        if consensus is None:
            return None
        retained = [group[i] for i in consensus.retained_indices]
        outliers = [group[i] for i in consensus.outlier_indices]
        if not retained:
            return None

        retained_values = [float(x["ef_value"]) for x in retained]
        retained_urls = [x.get("source_url") for x in retained]
        geos = sorted({_clean(x.get("source_geography")) for x in retained if _clean(x.get("source_geography"))})
        years = [int(x["source_year"]) for x in retained if str(x.get("source_year") or "").isdigit()]
        source_details = []
        retained_url_set = set(retained_urls)
        outlier_url_set = {_clean(x.get("source_url")) for x in outliers}
        for f in group:
            source_details.append({
                "ef_value": float(f["ef_value"]),
                "reference_unit": f.get("reference_unit"),
                "source_title": f.get("source_title"),
                "source_url": f.get("source_url"),
                "source_class": f.get("source_class"),
                "source_geography": f.get("source_geography"),
                "geography_representativeness": f.get("geography_representativeness"),
                "source_year": f.get("source_year"),
                "search_tier": f.get("search_tier"),
                "match_type": f.get("external_match_type"),
                "boundary_quality": f.get("boundary_quality"),
                "indicator": f.get("indicator"),
                "evidence_quote": f.get("evidence_quote"),
                "retained_for_consensus": _clean(f.get("source_url")) in retained_url_set,
                "outlier": _clean(f.get("source_url")) in outlier_url_set,
            })

        one = retained[0]
        multi = len(retained) > 1
        proxy_used = all(_clean(x.get("external_match_type")) != "DIRECT_PRODUCT" for x in retained)
        return {
            "ef_value": float(consensus.central_value),
            "reference_unit": ref_unit,
            "lower_value": min(retained_values) if multi else None,
            "upper_value": max(retained_values) if multi else None,
            "verification": "EXTERNAL_VERIFIED",
            "verification_tier": "RELAXED",
            "source_class": "EXTERNAL_VERIFIED_RELAXED",
            "source_title": (
                f"Median of {len(retained)} source-supported factors" if multi
                else one.get("source_title")
            ),
            "source_url": None if multi else one.get("source_url"),
            "source_geography": ", ".join(geos) if geos else "Unspecified",
            "source_year": str(max(years)) if years else None,
            "search_tier": min(
                (x.get("search_tier") for x in retained),
                key=lambda t: tier_rank.get(_clean(t).upper(), 9),
            ),
            "search_query": None if multi else one.get("search_query"),
            "evidence_quote": None if multi else one.get("evidence_quote"),
            "declared_unit_evidence": None,
            "boundary": "Source-supported product-stage/production-context screening basis",
            "indicator": (
                "GWP-GHG/GWP-fossil external verified (relaxed phase)"
                if _requires_no_biogenic_storage(material)
                else "GWP external verified (relaxed phase)"
            ),
            "declared_impact_value": None,
            "declared_impact_unit": None,
            "declared_quantity": 1.0,
            "declared_unit": ref_unit,
            "uncertainty_type": "MULTI_SOURCE_DISPERSION" if multi else None,
            "uncertainty_lower_value": None,
            "uncertainty_upper_value": None,
            "uncertainty_gsd": None,
            "uncertainty_cv": None,
            "uncertainty_confidence_level": None,
            "uncertainty_evidence_quote": None,
            "peer_values_json": json.dumps(retained_values, ensure_ascii=False),
            "peer_sources_json": json.dumps(source_details, ensure_ascii=False),
            "relaxed_sources_json": json.dumps(source_details, ensure_ascii=False),
            "relaxed_source_count": len(group),
            "relaxed_retained_count": len(retained),
            "relaxed_outlier_count": len(outliers),
            "relaxed_consensus_method": consensus.method,
            "relaxed_consensus_version": CONSENSUS_METHOD_VERSION,
            "reason": (
                f"External Verified relaxed-phase factor from {len(retained)} retained independent source(s) "
                f"on a common {ref_unit} basis using {consensus.method}; {len(outliers)} source-set outlier(s) excluded."
            ),
            "raw_model_output": None,
            "selected_result_id": one.get("selected_result_id") if not multi else None,
            "external_match_type": "EXTERNAL_VERIFIED_RELAXED_PROXY" if proxy_used else "EXTERNAL_VERIFIED_RELAXED_DIRECT",
            "external_proxy_basis": one.get("external_proxy_basis") if proxy_used else None,
            "product_identity_status": "PASS_SOURCE_SUPPORTED",
            "product_identity_reason": "Every retained source passed deterministic material/product identity checks.",
            "product_identity_title": None if multi else one.get("product_identity_title"),
            "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
            "proxy_representativeness": "EXTERNAL_VERIFIED_RELAXED",
        }

    def _resolve_one(self, item_id, material):
        key = material.strip().lower()
        if key in self._resolution_cache:
            return dict(self._resolution_cache[key]), []

        cached = self.evidence_cache.get(
            category="emission_factor", material=material,
            target_geography=self.target_geography,
        )
        if cached and cached.get("verification") in {"EXTERNAL_VERIFIED", "PROVISIONAL_SOURCE_SUPPORTED"}:
            cached = dict(cached)
            cached["reason"] = (_clean(cached.get("reason")) + " Runtime evidence-cache reuse.").strip()
            cached["source_class"] = _clean(cached.get("source_class")) or "RUNTIME_EVIDENCE_CACHE"
            cached["evidence_cache_version"] = CACHE_VERSION
            self._progress(f"EF cache hit · {material} · {cached.get('verification')}")
            self._resolution_cache[key] = dict(cached)
            cache_evidence = [{
                "item_id": item_id, "material": material, "search_tier": "RUNTIME_CACHE",
                "match_type": cached.get("external_match_type"), "proxy_basis": cached.get("external_proxy_basis"),
                "query": None, "query_subject": material, "result_id": "CACHE1",
                "title": cached.get("source_title"), "url": cached.get("source_url"),
                "snippet": cached.get("evidence_quote"), "excerpt": cached.get("evidence_quote"),
                "product_identity_ok": True, "product_identity_reason": "Previously accepted retrieved evidence",
                "selected": True, "uncertainty_peer": False, "relaxed_candidate": cached.get("verification") == "PROVISIONAL_SOURCE_SUPPORTED",
                "relaxed_retained": cached.get("verification") == "PROVISIONAL_SOURCE_SUPPORTED",
                "relaxed_outlier": False, "relaxed_reason": "Runtime evidence-cache reuse",
                "relaxed_normalized_ef": cached.get("ef_value"), "relaxed_reference_unit": cached.get("reference_unit"),
            }]
            return cached, cache_evidence

        all_evidence = []
        provisional_factors = []
        seen_urls: set[str] = set()
        promising_reasons: list[str] = []
        promising_candidates: list[dict[str, Any]] = []
        plans = list(_query_sets(material, self.target_geography))

        # ------------------------------------------------------------------
        # CLASS 3 / EXTERNAL VERIFIED — Phase A: strict target-geography/direct-product search.
        # This clock is independent from the relaxed Phase B. Phase B never inherits an
        # exhausted strict-phase timer or source/query budget.
        # ------------------------------------------------------------------
        class3_started = time.monotonic()
        class3_inspected = 0
        class3_queries_used = 0
        class3_plan = next((p for p in plans if p[0] == "NEPAL" and p[1] == "DIRECT_PRODUCT"), None)
        if class3_plan is not None:
            tier, match_type, proxy_basis, queries = class3_plan
            self._progress(
                f"External Verified · Phase A strict {self.target_geography} search · {material} · "
                f"independent budget {self.max_external_seconds_per_material:.0f} s · "
                f"max {self.class3_source_budget} sources"
            )
            candidates, rows, used = self._collect_evidence(
                queries, tier, item_id, material, match_type=match_type, proxy_basis=proxy_basis,
                max_candidates=self.class3_source_budget,
                max_queries=min(3, self.max_search_queries_per_material),
                seen_urls=seen_urls,
            )
            class3_queries_used += used
            class3_inspected += len(candidates)
            all_evidence.extend(rows)
            for c in candidates:
                ok_promising, why_promising = _promising_class4_candidate(material, c)
                if ok_promising:
                    promising_reasons.append(why_promising)
                    if c.get("url") and not any(x.get("url") == c.get("url") for x in promising_candidates):
                        promising_candidates.append(dict(c))

            selected, _ = self._choose_traceable(material, tier, candidates)
            if selected is not None:
                sid = selected.get("selected_result_id")
                for row in all_evidence:
                    if row["search_tier"] == tier and row["result_id"] == sid and row["url"] == selected["source_url"]:
                        row["selected"] = True
                peers = self._collect_peer_factors(material, tier, candidates, selected, max_total=3)
                selected = dict(selected)
                selected["peer_values_json"] = json.dumps([float(x["ef_value"]) for x in peers], ensure_ascii=False)
                selected["peer_sources_json"] = json.dumps([{
                    "ef_value": float(x["ef_value"]), "reference_unit": x.get("reference_unit"),
                    "source_title": x.get("source_title"), "source_url": x.get("source_url"),
                    "source_geography": x.get("source_geography"), "source_year": x.get("source_year"),
                } for x in peers], ensure_ascii=False)
                selected["external_verified_strict_elapsed_seconds"] = round(time.monotonic() - class3_started, 3)
                selected["evidence_cache_version"] = CACHE_VERSION
                self.evidence_cache.put(
                    category="emission_factor", material=material, target_geography=self.target_geography,
                    resolved=selected,
                )
                self._resolution_cache[key] = dict(selected)
                self._progress(f"EF Class 3 accepted · {material}")
                return selected, all_evidence

            # A target-geography source that fails strict Class-3 verification can
            # still be considered later under the relaxed External Verified Phase B.
            if self.allow_source_supported_provisional and candidates:
                provisional_factors.extend(self._extract_relaxed_candidates(material, tier, candidates))

        self._progress(
            f"EF Class 3 finished · {material} · {class3_inspected} source(s), "
            f"{class3_queries_used} query/queries · starting External Verified Phase B with a fresh clock"
        )

        # ------------------------------------------------------------------
        # CLASS 4 — independent source-supported search.
        # Geography is deliberately relaxed. Technical equivalents are allowed.
        # Explicit A1-A3 is preferred but not mandatory: a retrieved GWP/carbon
        # value can qualify when the source clearly places it in production or
        # manufacturing context and does not show a broader life-cycle boundary.
        # ------------------------------------------------------------------
        if not self.allow_source_supported_provisional:
            self._progress(f"External Verified Phase B disabled · {material}; moving to Class 4 fallback")
            return None, all_evidence

        class4_started = time.monotonic()
        class4_inspected = 0
        external_verified_relaxed_queries_used = 0
        class4_source_budget = self.adaptive_total_source_budget
        class4_query_budget = self.adaptive_max_search_queries_per_material
        class4_time_budget = self.adaptive_external_seconds_per_material

        self._progress(
            f"External Verified · Phase B relaxed geography/source-supported search · {material} · "
            f"budget {class4_time_budget:.0f} s · up to {class4_query_budget} queries / "
            f"{class4_source_budget} unique sources · geography may differ"
        )

        class4_plans = [p for p in plans if not (p[0] == "NEPAL" and p[1] == "DIRECT_PRODUCT")]
        class4_priority = {
            ("SOUTH_ASIA", "DIRECT_PRODUCT"): 0,
            ("GLOBAL", "DIRECT_PRODUCT"): 1,
            ("ASIA", "DIRECT_PRODUCT"): 2,
            ("SOUTH_ASIA", "PRODUCT_PROXY"): 4,
            ("GLOBAL", "PRODUCT_PROXY"): 5,
            ("ASIA", "PRODUCT_PROXY"): 6,
            ("NEPAL", "PRODUCT_PROXY"): 7,
        }
        class4_plans.sort(key=lambda p: class4_priority.get((p[0], p[1]), 9))

        # Generate terminology only; the model is forbidden to supply any factor.
        equivalent_plan = infer_technical_equivalents(
            self.matcher, material=material, target_context=self.target_geography
        )
        equivalent_plans = []
        if equivalent_plan:
            for rec in equivalent_plan.get("search_terms", []):
                subject = _clean(rec.get("term"))
                basis = _clean(rec.get("equivalence_basis"))
                if not subject:
                    continue
                equivalent_plans.extend([
                    ("GLOBAL", "PRODUCT_PROXY", f"Dynamic technical equivalence: {basis}", [
                        f'"{subject}" GWP EPD',
                        f'"{subject}" "cradle-to-gate" GWP',
                        f'"{subject}" manufacturing "carbon footprint"',
                        f'"{subject}" GWP USA Canada EPD',
                    ]),
                    ("SOUTH_ASIA", "PRODUCT_PROXY", f"Dynamic technical equivalence: {basis}", [
                        f'"{subject}" GWP India EPD',
                        f'"{subject}" production carbon footprint South Asia',
                    ]),
                ])

        # Direct product searches first; then dynamic technical-equivalence terms;
        # then the remaining generic proxy plans. Search/url caches remove repeats.
        direct_plans = [p for p in class4_plans if p[1] == "DIRECT_PRODUCT"]
        proxy_plans = [p for p in class4_plans if p[1] != "DIRECT_PRODUCT"]
        search_plans = direct_plans + equivalent_plans + proxy_plans

        for tier, match_type, proxy_basis, queries in search_plans:
            if time.monotonic() - class4_started >= class4_time_budget:
                self._progress(f"External Verified Phase B time budget reached · {material}; moving toward Class 4 fallback")
                break
            if class4_inspected >= class4_source_budget or external_verified_relaxed_queries_used >= class4_query_budget:
                break
            remaining_sources = class4_source_budget - class4_inspected
            remaining_queries = class4_query_budget - external_verified_relaxed_queries_used
            candidates, rows, used = self._collect_evidence(
                queries, tier, item_id, material, match_type=match_type, proxy_basis=proxy_basis,
                max_candidates=min(2, remaining_sources),
                max_queries=min(2, remaining_queries),
                seen_urls=seen_urls,
            )
            external_verified_relaxed_queries_used += used
            class4_inspected += len(candidates)
            all_evidence.extend(rows)

            for c in candidates:
                ok_promising, why_promising = _promising_class4_candidate(material, c)
                if ok_promising:
                    promising_reasons.append(why_promising)
                    if c.get("url") and not any(x.get("url") == c.get("url") for x in promising_candidates):
                        promising_candidates.append(dict(c))

            if candidates:
                extracted = self._extract_relaxed_candidates(material, tier, candidates)
                provisional_factors.extend(extracted)
                by_url = {_clean(x.get("source_url")): x for x in extracted}
                candidate_by_url = {_clean(x.get("url")): x for x in candidates}
                for row in all_evidence:
                    url = _clean(row.get("url"))
                    cdiag = candidate_by_url.get(url)
                    if cdiag is not None:
                        row["relaxed_extraction_status"] = cdiag.get("relaxed_extraction_status")
                        row["relaxed_extraction_method"] = cdiag.get("relaxed_extraction_method")
                        row["relaxed_rejection_reason"] = cdiag.get("relaxed_rejection_reason")
                        row["gwp_table_detected"] = "[PDF TABLE PAGE" in _clean(cdiag.get("full_text"))
                        row["gwp_page_detected"] = "[PDF PAGE" in _clean(cdiag.get("full_text"))
                    f = by_url.get(url)
                    if f is not None:
                        row["relaxed_candidate"] = True
                        row["relaxed_reason"] = f.get("reason")
                        row["relaxed_normalized_ef"] = f.get("ef_value")
                        row["relaxed_reference_unit"] = f.get("reference_unit")

                provisional = self._build_provisional_consensus(material, provisional_factors)
                if provisional is not None:
                    try:
                        source_details = json.loads(provisional.get("relaxed_sources_json") or "[]")
                    except Exception:
                        source_details = []
                    retained_urls = {_clean(x.get("source_url")) for x in source_details if x.get("retained_for_consensus")}
                    outlier_urls = {_clean(x.get("source_url")) for x in source_details if x.get("outlier")}
                    for row in all_evidence:
                        url = _clean(row.get("url"))
                        if url in retained_urls:
                            row["relaxed_retained"] = True
                            row["selected"] = True
                        if url in outlier_urls:
                            row["relaxed_outlier"] = True
                    provisional = dict(provisional)
                    provisional["technical_equivalence_version"] = TECHNICAL_EQUIVALENCE_VERSION if equivalent_plan else None
                    provisional["external_verified_strict_elapsed_seconds"] = round(time.monotonic() - class3_started, 3)
                    provisional["external_verified_relaxed_elapsed_seconds"] = round(time.monotonic() - class4_started, 3)
                    provisional["external_verified_relaxed_queries_used"] = external_verified_relaxed_queries_used
                    provisional["external_verified_relaxed_sources_inspected"] = class4_inspected
                    provisional["evidence_cache_version"] = CACHE_VERSION
                    self.evidence_cache.put(
                        category="emission_factor", material=material, target_geography=self.target_geography,
                        resolved=provisional,
                    )
                    self._resolution_cache[key] = dict(provisional)
                    self._progress(
                        f"External Verified Phase B accepted · {material} · "
                        f"{provisional.get('relaxed_retained_count', 1)} retained source(s)"
                    )
                    return provisional, all_evidence

        # If a source looked promising but the first extraction was incomplete,
        # use remaining relaxed-phase time to re-open at most two such documents more
        # deeply. This is still source extraction only; no GWP value is in code.
        if promising_candidates and time.monotonic() - class4_started < class4_time_budget:
            old_pages = self.pdf_max_pages
            old_links = self.linked_documents_per_page
            try:
                self.pdf_max_pages = self.adaptive_pdf_max_pages
                self.linked_documents_per_page = self.adaptive_linked_documents_per_page
                deep_candidates = []
                for c0 in promising_candidates[:2]:
                    if time.monotonic() - class4_started >= class4_time_budget:
                        break
                    c = dict(c0)
                    self._progress(f"External Verified · Phase B deeper extraction of promising source · {material}")
                    c["full_text"] = self._extract(c.get("url"))
                    c["excerpt"] = c["full_text"][: max(self.excerpt_chars, 2500)]
                    deep_candidates.append(c)
                for c in deep_candidates:
                    provisional_factors.extend(
                        self._extract_relaxed_candidates(material, c.get("tier") or "GLOBAL", [c])
                    )
                provisional = self._build_provisional_consensus(material, provisional_factors) if provisional_factors else None
                if provisional is not None:
                    provisional = dict(provisional)
                    provisional["adaptive_search_used"] = True
                    provisional["adaptive_search_trigger"] = "; ".join(sorted(set(promising_reasons)))
                    provisional["adaptive_deep_extraction_used"] = True
                    provisional["external_verified_strict_elapsed_seconds"] = round(time.monotonic() - class3_started, 3)
                    provisional["external_verified_relaxed_elapsed_seconds"] = round(time.monotonic() - class4_started, 3)
                    provisional["external_verified_relaxed_queries_used"] = external_verified_relaxed_queries_used
                    provisional["external_verified_relaxed_sources_inspected"] = class4_inspected
                    provisional["evidence_cache_version"] = CACHE_VERSION
                    self.evidence_cache.put(
                        category="emission_factor", material=material,
                        target_geography=self.target_geography, resolved=provisional,
                    )
                    self._resolution_cache[key] = dict(provisional)
                    self._progress(f"External Verified Phase B accepted after deeper extraction · {material}")
                    return provisional, all_evidence
            finally:
                self.pdf_max_pages = old_pages
                self.linked_documents_per_page = old_links

        self._progress(
            f"External Verified Phase B finished without accepted factor · {material} · "
            f"{class4_inspected} source(s), {external_verified_relaxed_queries_used} query/queries · moving to Class 4 fallback"
        )
        return None, all_evidence

    @staticmethod
    def _parse_model_json(raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw.strip())
            return obj if isinstance(obj, dict) else None
        except Exception:
            m = re.search(r"\{.*\}", raw or "", flags=re.S)
            if not m:
                return None
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None


    def _validated_model_factor(
        self, obj: dict[str, Any], material: str, expected_reference_unit: str,
        preferred_reference_unit: str, *, analog: bool,
    ):
        """Validate one Class-4 model estimate and normalize its denominator.

        No environmental magnitude prior is used. The model must answer in the
        unit requested for that independent call. Python then converts that
        denominator, when dimensionally possible without a material property, to
        the final reference unit used by the BOM calculation. This makes kg/g/t
        or kWh/MJ scale mistakes visible as disagreement between independent
        calls instead of silently accepting one denominator interpretation.
        """
        required = {
            "found", "central_value", "lower_value", "upper_value", "reference_unit",
            "boundary", "indicator", "geography_assumption", "estimation_basis",
            ("analog_material" if analog else "product_interpretation"), "rationale",
        }
        if not isinstance(obj, dict) or set(obj.keys()) != required or not bool(obj.get("found")):
            return None, "schema_or_found_invalid"

        raw_central = _num(obj.get("central_value"))
        raw_lower = _num(obj.get("lower_value"))
        raw_upper = _num(obj.get("upper_value"))
        if raw_central is None or raw_lower is None or raw_upper is None or not (0 < raw_lower < raw_central < raw_upper):
            return None, "range_invalid"

        expected_ref = norm_unit(expected_reference_unit)
        preferred_ref = norm_unit(preferred_reference_unit)
        generated_ref = norm_unit(obj.get("reference_unit"))
        if generated_ref != expected_ref:
            return None, f"reference_unit_mismatch:{generated_ref or 'blank'}_vs_requested_{expected_ref}"

        if not _boundary_ok(_clean(obj.get("boundary"))):
            return None, "boundary_not_product_stage_a1_a3"

        indicator = _clean(obj.get("indicator")).upper().replace("_", "-")
        if _requires_no_biogenic_storage(material):
            if "GWP-GHG" not in indicator and "GWP-FOSSIL" not in indicator:
                return None, "biogenic_indicator_invalid"
            final_indicator = "GWP-GHG" if "GWP-GHG" in indicator else "GWP-fossil"
        else:
            if "GWP" not in indicator and "CLIMATE" not in indicator:
                return None, "indicator_invalid"
            final_indicator = "GWP-total"

        # First validate the raw model output structurally in the unit requested
        # for this call. This uses no material-specific magnitude limit.
        plausible, reason = emission_factor_plausible(
            material, raw_central, generated_ref, raw_lower, raw_upper
        )
        if not plausible:
            return None, reason

        central = convert_factor_reference_basis(raw_central, generated_ref, preferred_ref)
        lower = convert_factor_reference_basis(raw_lower, generated_ref, preferred_ref)
        upper = convert_factor_reference_basis(raw_upper, generated_ref, preferred_ref)
        if central is None or lower is None or upper is None:
            return None, f"reference_unit_not_dimensionally_convertible:{generated_ref}_to_{preferred_ref}"
        if not (0 < lower < central < upper):
            return None, "normalized_range_invalid"

        plausible, reason = emission_factor_plausible(
            material, central, preferred_ref, lower, upper
        )
        if not plausible:
            return None, f"normalized_{reason}"

        if not analog:
            identity_ok, identity_reason = self._model_product_identity_compatible(
                material,
                _clean(obj.get("product_interpretation")),
                _clean(obj.get("rationale")),
                _clean(obj.get("estimation_basis")),
            )
            if not identity_ok:
                return None, f"semantic_identity_veto:{identity_reason}"

        return {
            "central": float(central), "lower": float(lower), "upper": float(upper),
            "ref_unit": preferred_ref, "indicator": final_indicator,
            "basis": _clean(obj.get("estimation_basis")),
            "interpretation": _clean(obj.get("analog_material" if analog else "product_interpretation")),
            "rationale": _clean(obj.get("rationale")),
            "geography": _clean(obj.get("geography_assumption")),
            "generated_reference_unit": generated_ref,
            "requested_reference_unit": expected_ref,
            "generated_central_value": float(raw_central),
            "generated_lower_value": float(raw_lower),
            "generated_upper_value": float(raw_upper),
        }, "ok"

    @staticmethod
    def _invalid_factor_candidate(*, analog: bool, analog_material: str = "") -> dict[str, Any]:
        d = {
            "found": False, "central_value": None, "lower_value": None,
            "upper_value": None, "reference_unit": None, "boundary": None,
            "indicator": None, "geography_assumption": None,
            "estimation_basis": None, "rationale": None,
        }
        if analog:
            d["analog_material"] = analog_material
        else:
            d["product_interpretation"] = None
        return d

    def _generate_one_factor_candidate(
        self, *, system_prompt: str, payload: dict[str, Any], material: str,
        requested_reference_unit: str, preferred_reference_unit: str, analog: bool,
        analog_material: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        attempts = []
        working = dict(payload)
        for attempt_no in range(1, 3):
            raw = _call_matcher(self.matcher, system_prompt, working, max_new_tokens=480)
            obj = self._parse_model_json(raw)
            reason = "parse_error"
            parsed = None
            if isinstance(obj, dict):
                parsed, reason = self._validated_model_factor(
                    obj, material, requested_reference_unit, preferred_reference_unit, analog=analog
                )
                if analog and parsed is not None and _clean(obj.get("analog_material")) != _clean(analog_material):
                    parsed = None
                    reason = "analog_name_not_preserved"
            attempts.append({
                "attempt": attempt_no, "accepted": parsed is not None,
                "requested_reference_unit": norm_unit(requested_reference_unit),
                "normalized_reference_unit": norm_unit(preferred_reference_unit),
                "reason": reason, "raw_model_output": raw,
            })
            if parsed is not None:
                return obj, attempts
            working = {
                **payload,
                "repair_instruction": (
                    "Return one complete JSON object only. Correct the schema/unit/boundary/interval issue stated below. "
                    "Use requested_reference_unit exactly. Do not use any prescribed material-specific numeric range."
                ),
                "previous_validation_error": reason,
                "previous_output": (raw or "")[-1400:],
            }
        return self._invalid_factor_candidate(analog=analog, analog_material=analog_material), attempts


    def _snapshot_process_factor(self, process: dict[str, Any], preferred: str):
        """Return one frozen-snapshot factor normalized to the BOM reference unit.

        This helper never invents or clamps a magnitude. It accepts only a finite
        positive factor already present in the hash-verified openLCA snapshot and
        uses universal denominator conversion when the process basis is directly
        convertible to the BOM basis.
        """
        uid = _clean(process.get("process_uuid"))
        factors = self.factor_snapshot.get("factors") if isinstance(self.factor_snapshot, dict) else None
        if not uid or not isinstance(factors, dict):
            return None, "factor_snapshot_unavailable"
        snap = factors.get(uid)
        if not isinstance(snap, dict) or _clean(snap.get("status")).upper() != "OK":
            return None, "snapshot_factor_not_ok"
        raw_factor = _num(snap.get("emission_factor"))
        raw_ref = norm_unit(snap.get("reference_unit") or process.get("ref_unit"))
        if raw_factor is None or raw_factor <= 0 or not raw_ref:
            return None, "snapshot_factor_missing_or_nonpositive"
        normalized = convert_factor_reference_basis(float(raw_factor), raw_ref, preferred)
        if normalized is None or not math.isfinite(float(normalized)) or float(normalized) <= 0:
            return None, f"snapshot_reference_unit_not_convertible:{raw_ref}_to_{preferred}"
        ok, reason = emission_factor_plausible("", float(normalized), preferred)
        if not ok:
            return None, f"snapshot_normalized_{reason}"
        return {
            "process_uuid": uid,
            "process_name": _clean(process.get("process_name")),
            "location": _clean(process.get("location")),
            "category": _clean(process.get("category")),
            "process_type": _clean(process.get("process_type")),
            "process_reference_unit": raw_ref,
            "snapshot_factor_raw": float(raw_factor),
            "reference_unit": preferred,
            "normalized_factor": float(normalized),
            "impact_basis": snap.get("impact_basis"),
        }, "ok"

    def _rank_dynamic_database_candidates(
        self,
        material: str,
        candidates: list[dict[str, Any]],
        *,
        proxy_level: str,
    ) -> tuple[list[str], dict[str, Any]]:
        """Rank only supplied catalog processes; factor magnitudes are hidden from Qwen."""
        if not candidates:
            return [], {"accepted": False, "reason": "no_candidates"}
        allowed = {_clean(x.get("process_uuid")) for x in candidates if _clean(x.get("process_uuid"))}
        presented = []
        for c in candidates[:18]:
            presented.append({
                "process_uuid": c.get("process_uuid"),
                "process_name": c.get("process_name"),
                "category": c.get("category"),
                "location": c.get("location"),
                "process_type": c.get("process_type"),
                "reference_unit": c.get("process_reference_unit"),
                "query_origin": c.get("query_origin"),
                "analog_material": c.get("analog_material"),
                "similarity_basis": c.get("similarity_basis"),
            })
        payload = {
            "material": material,
            "target_context": self.target_geography,
            "proxy_level": proxy_level,
            "candidates": presented,
            "important": "No emission-factor values are supplied. Rank only technical/process suitability.",
        }
        raw = _call_matcher(
            self.matcher, DYNAMIC_DATABASE_PROXY_SELECTOR_SYSTEM_PROMPT, payload,
            max_new_tokens=320,
        )
        obj = self._parse_model_json(raw)
        ranked: list[str] = []
        reason = ""
        if isinstance(obj, dict) and set(obj.keys()) == {"ranked_process_uuids", "selection_reason"}:
            xs = obj.get("ranked_process_uuids")
            if isinstance(xs, list) and len(xs) <= 3 and all(isinstance(x, str) for x in xs):
                ranked = [x for x in xs if x in allowed]
                if len(ranked) != len(xs):
                    ranked = []
                reason = _clean(obj.get("selection_reason"))
        return ranked, {
            "accepted": bool(ranked),
            "reason": reason or ("valid_ranked_processes" if ranked else "model_ranking_invalid_or_empty"),
            "raw_model_output": raw,
            "candidate_count": len(candidates),
            "factor_values_hidden_from_model": True,
        }

    @staticmethod
    def _sort_catalog_proxy_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deterministic fallback order after all candidates already pass compatibility."""
        return sorted(
            candidates,
            key=lambda x: (
                int(x.get("analog_rank") or 0),
                -float(x.get("retrieval_score") or 0.0),
                _clean(x.get("process_name")).lower(),
                _clean(x.get("process_uuid")),
            ),
        )

    def _resolved_dynamic_snapshot_proxy(
        self,
        material: str,
        selected: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        method: str,
        proxy_level: str,
        selection_reason: str,
        selector_diag: dict[str, Any],
        planning_diag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        peers = [
            {
                "process_uuid": c.get("process_uuid"),
                "process_name": c.get("process_name"),
                "location": c.get("location"),
                "reference_unit": c.get("reference_unit"),
                "normalized_factor": c.get("normalized_factor"),
                "query_origin": c.get("query_origin"),
                "analog_material": c.get("analog_material"),
                "similarity_basis": c.get("similarity_basis"),
            }
            for c in candidates[:12]
        ]
        selected_factor = float(selected["normalized_factor"])
        attempt = {
            "stage": method,
            "accepted": True,
            "proxy_level": proxy_level,
            "selected_process_uuid": selected.get("process_uuid"),
            "selected_process_name": selected.get("process_name"),
            "selection_reason": selection_reason,
            "selector": selector_diag,
            "planning": planning_diag,
            "candidate_count": len(candidates),
            "factor_origin": "HASH_VERIFIED_FROZEN_OPENLCA_SNAPSHOT",
            "llm_generated_emission_factor": False,
        }
        self._last_fallback_attempts.append(attempt)
        analog_text = _clean(selected.get("analog_material"))
        sim_text = _clean(selected.get("similarity_basis"))
        basis_bits = [
            f"Frozen ELCD/openLCA process: {selected.get('process_name')}",
            f"dynamic proxy level: {proxy_level}",
        ]
        if analog_text:
            basis_bits.append(f"analog: {analog_text}")
        if sim_text:
            basis_bits.append(f"analog basis: {sim_text}")
        return {
            "ef_value": selected_factor,
            "reference_unit": selected.get("reference_unit"),
            "lower_value": None,
            "upper_value": None,
            "verification": "UNVERIFIED_FALLBACK_ESTIMATE",
            "source_class": "UNVERIFIED_FALLBACK_ESTIMATE",
            "fallback_method": method,
            "source_title": f"Frozen ELCD/openLCA dynamic proxy: {selected.get('process_name')}",
            "source_url": None,
            "source_geography": selected.get("location") or "ELCD process location not stated",
            "source_year": None,
            "search_tier": proxy_level,
            "search_query": selected.get("query_origin"),
            "evidence_quote": None,
            "declared_unit_evidence": f"Frozen snapshot reference unit: {selected.get('process_reference_unit')}",
            "boundary": "Selected LCIA of frozen ELCD/openLCA proxy process; proxy selection is unverified",
            "indicator": "GWP-total",
            "declared_impact_value": selected_factor,
            "declared_impact_unit": "kg CO2e",
            "declared_quantity": 1.0,
            "declared_unit": selected.get("reference_unit"),
            "uncertainty_type": "DYNAMIC_DATABASE_PROXY_PEDIGREE_REQUIRED",
            "uncertainty_lower_value": None,
            "uncertainty_upper_value": None,
            "uncertainty_gsd": None,
            "uncertainty_cv": None,
            "uncertainty_confidence_level": None,
            "uncertainty_evidence_quote": None,
            "reason": (
                "No acceptable external/source-supported factor was found. The numerical factor was NOT generated by Qwen. "
                "Qwen was used only to expand terminology/analogs and rank supplied frozen catalog processes; Python then read the selected process factor from the hash-verified openLCA snapshot. "
                + "; ".join(basis_bits)
            ),
            "raw_model_output": selector_diag.get("raw_model_output"),
            "fallback_attempts_json": json.dumps(self._last_fallback_attempts, ensure_ascii=False),
            "guardrail_version": GUARDRAIL_VERSION,
            "guardrail_status": "PASS_DYNAMIC_FROZEN_DATABASE_PROXY",
            "guardrail_reason": selection_reason,
            "selected_result_id": selected.get("process_uuid"),
            "external_match_type": "UNVERIFIED_FALLBACK_ESTIMATE",
            "external_proxy_basis": "; ".join(basis_bits),
            "product_identity_status": "DYNAMIC_DATABASE_PROXY_NOT_SOURCE_VERIFIED",
            "product_identity_reason": selection_reason,
            "product_identity_title": material,
            "biogenic_carbon_policy": None,
            "proxy_representativeness": proxy_level,
            "peer_values_json": json.dumps([c.get("normalized_factor") for c in candidates[:12]], ensure_ascii=False),
            "peer_sources_json": json.dumps(peers, ensure_ascii=False),
            "unit_basis_records_json": json.dumps(peers, ensure_ascii=False),
            "terminal_quality_flag": "DATABASE_ANCHORED_CLASS5",
        }

    def _estimate_dynamic_frozen_database_proxy(self, material: str, source_unit: Any):
        """Class-4A: same-family dynamic proxy using only frozen database numbers."""
        preferred = norm_unit(source_unit)
        if preferred not in {"kg", "g", "t", "m3", "l", "cm3", "m2", "cm2", "mm2", "item", "kwh", "mj"}:
            self._last_fallback_attempts.append({
                "stage": "CLASS5A_DYNAMIC_FROZEN_DATABASE_PROXY", "accepted": False,
                "reason": f"unsupported_reference_unit:{preferred or 'blank'}",
            })
            return None
        if _requires_no_biogenic_storage(material):
            self._last_fallback_attempts.append({
                "stage": "CLASS5A_DYNAMIC_FROZEN_DATABASE_PROXY", "accepted": False,
                "reason": "skipped_biogenic_policy_requires_gwp_ghg_or_fossil",
            })
            return None
        if not isinstance(self.factor_snapshot.get("factors"), dict):
            self._last_fallback_attempts.append({
                "stage": "CLASS5A_DYNAMIC_FROZEN_DATABASE_PROXY", "accepted": False,
                "reason": "factor_snapshot_unavailable",
            })
            return None

        queries = [material]
        plan = infer_technical_equivalents(
            self.matcher, material=material, target_context=self.target_geography
        )
        if plan:
            normalized_product = _clean(plan.get("normalized_product"))
            if normalized_product:
                compatible, _ = process_compatibility(material, normalized_product)
                if compatible:
                    queries.append(normalized_product)
            queries.extend(_clean(x.get("term")) for x in plan.get("search_terms", []) if _clean(x.get("term")))
        queries = list(dict.fromkeys(q for q in queries if q))

        by_uuid: dict[str, dict[str, Any]] = {}
        retrieval_diag = []
        for query_rank, query in enumerate(queries, start=1):
            try:
                retrieved = list(self.matcher.retriever.retrieve(query) or [])
            except Exception as exc:
                retrieval_diag.append({"query": query, "error": str(exc)})
                continue
            retrieval_diag.append({"query": query, "retrieved": len(retrieved)})
            for r in retrieved:
                pname = _clean(r.get("process_name"))
                compatible, why = process_compatibility(material, pname)
                if not compatible:
                    continue
                factor_rec, freason = self._snapshot_process_factor(r, preferred)
                if factor_rec is None:
                    continue
                factor_rec.update({
                    "query_origin": query,
                    "query_rank": query_rank,
                    "analog_rank": 0,
                    "retrieval_score": float(r.get("retrieval_score") or 0.0),
                    "compatibility_reason": why,
                })
                uid = factor_rec["process_uuid"]
                prev = by_uuid.get(uid)
                if prev is None or float(factor_rec.get("retrieval_score") or 0.0) > float(prev.get("retrieval_score") or 0.0):
                    by_uuid[uid] = factor_rec

        candidates = self._sort_catalog_proxy_candidates(list(by_uuid.values()))
        if not candidates:
            self._last_fallback_attempts.append({
                "stage": "CLASS5A_DYNAMIC_FROZEN_DATABASE_PROXY", "accepted": False,
                "reason": "no_compatible_unit_convertible_frozen_process",
                "queries": queries, "retrieval": retrieval_diag,
                "technical_equivalence_plan": plan,
            })
            return None

        ranked, selector_diag = self._rank_dynamic_database_candidates(
            material, candidates, proxy_level="SAME_FAMILY_DYNAMIC_FROZEN_DATABASE_PROXY"
        )
        selected = next((c for uid in ranked for c in candidates if c.get("process_uuid") == uid), None)
        selection_reason = selector_diag.get("reason") or ""
        if selected is None:
            # This deterministic fallback is safe because every candidate has
            # already passed the requested material's family compatibility gate
            # and has a usable factor in the hash-verified snapshot.
            selected = candidates[0]
            selection_reason = "MODEL_RANKING_EMPTY; deterministic best retrieval among pre-vetted same-family processes"
        return self._resolved_dynamic_snapshot_proxy(
            material, selected, candidates,
            method="DYNAMIC_FROZEN_ELCD_SAME_FAMILY_PROXY",
            proxy_level="SAME_FAMILY_DYNAMIC_FROZEN_DATABASE_PROXY",
            selection_reason=selection_reason,
            selector_diag=selector_diag,
            planning_diag={"technical_equivalence_plan": plan, "queries": queries, "retrieval": retrieval_diag},
        )

    def _estimate_dynamic_frozen_analog_proxy(self, material: str, source_unit: Any):
        """Class-4B: dynamically inferred nearby analog, but factor remains database-derived."""
        preferred = norm_unit(source_unit)
        if preferred not in {"kg", "g", "t", "m3", "l", "cm3", "m2", "cm2", "mm2", "item", "kwh", "mj"}:
            self._last_fallback_attempts.append({
                "stage": "CLASS5B_DYNAMIC_FROZEN_ANALOG_PROXY", "accepted": False,
                "reason": f"unsupported_reference_unit:{preferred or 'blank'}",
            })
            return None
        if _requires_no_biogenic_storage(material):
            self._last_fallback_attempts.append({
                "stage": "CLASS5B_DYNAMIC_FROZEN_ANALOG_PROXY", "accepted": False,
                "reason": "skipped_biogenic_policy_requires_gwp_ghg_or_fossil",
            })
            return None
        if not isinstance(self.factor_snapshot.get("factors"), dict):
            return None

        plan = infer_analog_plan(
            self.matcher, material=material,
            estimation_target=f"select a frozen ELCD/openLCA A1-A3 GWP proxy convertible to {preferred}; do not estimate GWP",
            target_context=self.target_geography,
            feedback="Keep analogs as close as possible to the requested product family; no numerical values.",
        )
        if not plan:
            self._last_fallback_attempts.append({
                "stage": "CLASS5B_DYNAMIC_FROZEN_ANALOG_PROXY", "accepted": False,
                "reason": "semantic_analog_plan_invalid",
            })
            return None

        requested_family = classify_material(material)
        by_uuid: dict[str, dict[str, Any]] = {}
        analog_diag = []
        for analog_rank, analog in enumerate(plan.get("analogs", []), start=1):
            analog_name = _clean(analog.get("analog_material"))
            similarity_basis = _clean(analog.get("similarity_basis"))
            if not analog_name:
                continue
            # For known requested families, the inferred analog must itself pass
            # the same deterministic family-compatibility gate. This prevents a
            # galvanized-steel product from drifting to aluminium, paperboard,
            # cement/plastic, etc. Unknown families retain the LLM-planned analog.
            if requested_family != "UNKNOWN":
                analog_ok, analog_reason = process_compatibility(material, analog_name)
                if not analog_ok:
                    analog_diag.append({
                        "analog": analog_name, "accepted": False,
                        "reason": f"requested_family_veto:{analog_reason}",
                    })
                    continue
            try:
                retrieved = list(self.matcher.retriever.retrieve(analog_name) or [])
            except Exception as exc:
                analog_diag.append({"analog": analog_name, "accepted": False, "reason": str(exc)})
                continue
            analog_diag.append({"analog": analog_name, "accepted": True, "retrieved": len(retrieved)})
            for r in retrieved:
                pname = _clean(r.get("process_name"))
                process_ok, process_reason = process_compatibility(analog_name, pname)
                if not process_ok:
                    continue
                factor_rec, freason = self._snapshot_process_factor(r, preferred)
                if factor_rec is None:
                    continue
                factor_rec.update({
                    "query_origin": analog_name,
                    "analog_material": analog_name,
                    "similarity_basis": similarity_basis,
                    "semantic_family": plan.get("family_description"),
                    "analog_rank": analog_rank,
                    "retrieval_score": float(r.get("retrieval_score") or 0.0),
                    "compatibility_reason": process_reason,
                })
                uid = factor_rec["process_uuid"]
                prev = by_uuid.get(uid)
                if prev is None or analog_rank < int(prev.get("analog_rank") or 999):
                    by_uuid[uid] = factor_rec

        candidates = self._sort_catalog_proxy_candidates(list(by_uuid.values()))
        if not candidates:
            self._last_fallback_attempts.append({
                "stage": "CLASS5B_DYNAMIC_FROZEN_ANALOG_PROXY", "accepted": False,
                "reason": "no_compatible_unit_convertible_dynamic_analog_process",
                "semantic_analog_plan": plan, "analog_diagnostics": analog_diag,
            })
            return None

        ranked, selector_diag = self._rank_dynamic_database_candidates(
            material, candidates, proxy_level="SEMANTIC_ANALOG_DYNAMIC_FROZEN_DATABASE_PROXY"
        )
        selected = next((c for uid in ranked for c in candidates if c.get("process_uuid") == uid), None)
        selection_reason = selector_diag.get("reason") or ""
        if selected is None:
            selected = candidates[0]
            selection_reason = "MODEL_RANKING_EMPTY; deterministic closest analog order among pre-vetted analog processes"
        return self._resolved_dynamic_snapshot_proxy(
            material, selected, candidates,
            method="DYNAMIC_FROZEN_ELCD_SEMANTIC_ANALOG_PROXY",
            proxy_level="SEMANTIC_ANALOG_DYNAMIC_FROZEN_DATABASE_PROXY",
            selection_reason=selection_reason,
            selector_diag=selector_diag,
            planning_diag={"semantic_analog_plan": plan, "analog_diagnostics": analog_diag},
        )

    @staticmethod
    def _model_product_identity_compatible(material: str, interpretation: str, rationale: str = "", basis: str = "") -> tuple[bool, str]:
        """Deterministic semantic veto for free-form terminal model estimates.

        The product_interpretation itself must pass the same taxonomy/process
        compatibility logic used elsewhere. Rationale/basis may veto a result but
        can never rescue a contradictory interpretation. This prevents circular
        acceptance where a model merely repeats an acronym (for example ``CGI``)
        while expanding it into a different product family. No GWP magnitude or
        material-specific environmental value is used here.
        """
        expected = classify_material(material)
        if expected == "UNKNOWN":
            return True, "unknown_material_family_no_deterministic_identity_veto"

        interp = _clean(interpretation)
        if not interp:
            return False, "model_product_interpretation_blank"

        # Identity must be explicit in product_interpretation itself. Do not let
        # rationale, estimation basis, or an echoed shorthand token rescue it.
        interp_ok, interp_reason = process_compatibility(material, interp)
        if not interp_ok:
            return False, f"model_product_interpretation_not_family_compatible:{expected}:{interp_reason}"

        ferrous = {"REBAR", "STEEL_WIRE", "STEEL_FASTENER", "GALVANIZED_FLAT_STEEL"}
        cementitious_or_stone = {"CEMENT", "PLAIN_CONCRETE", "STONECRETE_BLOCK", "PLASTER", "NATURAL_STONE", "GRAVEL_AGGREGATE", "SAND"}
        bio = {"TIMBER", "PLYWOOD", "BAMBOO"}
        earth = {"SOIL_EARTH", "SOIL_BLOCK"}

        rationale_text = " ".join(x for x in (_clean(rationale), _clean(basis)) if x)
        rationale_family = classify_material(rationale_text) if rationale_text else "UNKNOWN"
        if expected in ferrous and rationale_family not in ferrous | {"UNKNOWN"}:
            return False, f"model_rationale_chemistry_contradiction:{expected}_vs_{rationale_family}"
        if expected in bio and rationale_family in ferrous | cementitious_or_stone | earth:
            return False, f"model_rationale_chemistry_contradiction:{expected}_vs_{rationale_family}"
        if expected in earth and rationale_family in ferrous | bio | cementitious_or_stone:
            return False, f"model_rationale_chemistry_contradiction:{expected}_vs_{rationale_family}"
        if expected in cementitious_or_stone and rationale_family in ferrous | bio | earth:
            return False, f"model_rationale_chemistry_contradiction:{expected}_vs_{rationale_family}"

        combined = " ".join(x for x in (interp, _clean(rationale), _clean(basis)) if x)
        combined_ok, combined_reason = process_compatibility(material, combined)
        if not combined_ok:
            return False, f"model_combined_identity_not_family_compatible:{expected}:{combined_reason}"
        return True, "model_identity_passes_interpretation_first_family_gate"

    def _parse_terminal_minimal_factor(self, raw: str, material: str, preferred: str):
        obj = self._parse_model_json(raw)
        required = {
            "found", "central_value", "reference_unit", "boundary", "indicator",
            "product_interpretation", "rationale",
        }
        if not isinstance(obj, dict) or set(obj.keys()) != required or not bool(obj.get("found")):
            return None, "terminal_minimal_schema_invalid"
        central = _num(obj.get("central_value"))
        if central is None or central <= 0:
            return None, "terminal_minimal_value_invalid"
        generated_ref = norm_unit(obj.get("reference_unit"))
        if generated_ref != preferred:
            return None, f"terminal_minimal_reference_unit_mismatch:{generated_ref or 'blank'}_vs_{preferred}"
        if not _boundary_ok(_clean(obj.get("boundary"))):
            return None, "terminal_minimal_boundary_invalid"
        indicator = _clean(obj.get("indicator")).upper().replace("_", "-")
        if _requires_no_biogenic_storage(material):
            if "GWP-GHG" not in indicator and "GWP-FOSSIL" not in indicator:
                return None, "terminal_minimal_biogenic_indicator_invalid"
            final_indicator = "GWP-GHG" if "GWP-GHG" in indicator else "GWP-fossil"
        else:
            if "GWP" not in indicator and "CLIMATE" not in indicator:
                return None, "terminal_minimal_indicator_invalid"
            final_indicator = "GWP-total"
        ok, reason = emission_factor_plausible(material, central, preferred)
        if not ok:
            return None, f"terminal_minimal_{reason}"
        identity_ok, identity_reason = self._model_product_identity_compatible(
            material, _clean(obj.get("product_interpretation")), _clean(obj.get("rationale")), ""
        )
        if not identity_ok:
            return None, f"terminal_minimal_semantic_identity_veto:{identity_reason}"
        return {
            "central": float(central), "lower": None, "upper": None,
            "ref_unit": preferred, "indicator": final_indicator,
            "interpretation": _clean(obj.get("product_interpretation")),
            "rationale": _clean(obj.get("rationale")),
        }, "ok"

    def _estimate_terminal_llm_only_value(self, material: str, source_unit: Any):
        """Last-resort single model-only value after database anchors are unavailable.

        Unlike the earlier implementation, repeated model outputs are never used
        as a validation signal and no pooled/median candidate rule can turn shared
        hallucination into an accepted value. One fresh product-level estimate is
        requested in the final reference unit. It must pass structural validation
        AND the deterministic material-family semantic veto. A minimal-schema
        retry exists only for JSON/parser recovery and must pass the same identity
        veto. The result remains Class 4 and is excluded from verified and
        verified subtotal.
        """
        preferred = norm_unit(source_unit)
        supported = {"kg", "g", "t", "m3", "l", "cm3", "m2", "cm2", "mm2", "item", "kwh", "mj"}
        if preferred not in supported:
            self._last_fallback_attempts.append({
                "stage": "CLASS5C_TERMINAL_MODEL_ONLY", "accepted": False,
                "reason": f"unsupported_reference_unit:{preferred or 'blank'}",
            })
            return None

        self._progress(f"EF Class 4 · Method C · one terminal model-only estimate · {material} · basis {preferred}")
        payload = {
            "material": material,
            "target_context": self.target_geography,
            "requested_reference_unit": preferred,
            "final_calculation_reference_unit": preferred,
            "required_indicator": _required_indicator_policy(material),
            "deterministic_material_family": classify_material(material),
            "identity_validation_requirement": "product_interpretation must explicitly describe this family; do not merely echo or re-expand an acronym",
            "estimation_lens": "fresh product-level composition and manufacturing reconstruction",
            **emission_guardrail_context(material, preferred),
            "status": (
                "External evidence and dynamic frozen-database proxy routes failed. "
                "Return one terminal unverified numerical estimate only if the product identity can be preserved."
            ),
            "important": (
                "Do not treat agreement with any earlier model output as evidence. "
                "Describe the requested product itself; a deterministic family gate will veto contradictory interpretations."
            ),
        }
        obj, attempts = self._generate_one_factor_candidate(
            system_prompt=TERMINAL_LLM_ONLY_EF_SINGLE_SYSTEM_PROMPT,
            payload=payload,
            material=material,
            requested_reference_unit=preferred,
            preferred_reference_unit=preferred,
            analog=False,
        )
        parsed, reason = (
            self._validated_model_factor(obj, material, preferred, preferred, analog=False)
            if isinstance(obj, dict) and bool(obj.get("found")) else (None, "invalid_terminal_candidate")
        )
        self._last_fallback_attempts.append({
            "stage": "CLASS5C_TERMINAL_MODEL_ONLY",
            "accepted": parsed is not None,
            "reason": reason,
            "generation_attempts": attempts,
            "repeated_value_consensus_used": False,
        })
        if parsed is not None:
            central = float(parsed["central"])
            lower = float(parsed["lower"])
            upper = float(parsed["upper"])
            return {
                "ef_value": central,
                "reference_unit": preferred,
                "lower_value": lower,
                "upper_value": upper,
                "verification": "UNVERIFIED_FALLBACK_ESTIMATE",
                "source_class": "UNVERIFIED_FALLBACK_ESTIMATE",
                "fallback_method": "TERMINAL_SINGLE_LLM_VALUE_SEMANTIC_VETO",
                "source_title": "Qwen terminal single model-only exploratory estimate",
                "source_url": None,
                "source_geography": "Unverified model assumption; target context " + self.target_geography,
                "source_year": None,
                "search_tier": "FINAL_TERMINAL_SINGLE_LLM_VALUE",
                "search_query": None,
                "evidence_quote": None,
                "declared_unit_evidence": None,
                "boundary": "A1-A3 / cradle-to-gate model assumption",
                "indicator": parsed.get("indicator") or ("GWP-GHG" if _requires_no_biogenic_storage(material) else "GWP-total"),
                "declared_impact_value": central,
                "declared_impact_unit": "kg CO2e",
                "declared_quantity": 1.0,
                "declared_unit": preferred,
                "uncertainty_type": "TERMINAL_SINGLE_MODEL_RANGE",
                "uncertainty_lower_value": lower,
                "uncertainty_upper_value": upper,
                "uncertainty_gsd": None,
                "uncertainty_cv": None,
                "uncertainty_confidence_level": 95.0,
                "uncertainty_evidence_quote": None,
                "reason": (
                    "No source-supported or database-anchored proxy could be used. One fresh model-only A1-A3 estimate was retained solely for terminal exploratory coverage. "
                    "It passed unit/boundary/indicator checks and the deterministic material-family semantic veto. Repeated-value consensus was not used as validation."
                ),
                "raw_model_output": json.dumps(obj, ensure_ascii=False),
                "fallback_attempts_json": json.dumps(self._last_fallback_attempts, ensure_ascii=False),
                "guardrail_version": GUARDRAIL_VERSION,
                "guardrail_status": "TERMINAL_SINGLE_VALUE_INTERPRETATION_FIRST_IDENTITY_PASS",
                "guardrail_reason": "No repeated-value consensus used; product_interpretation independently passed deterministic family compatibility",
                "selected_result_id": None,
                "external_match_type": "UNVERIFIED_FALLBACK_ESTIMATE",
                "external_proxy_basis": None,
                "product_identity_status": "TERMINAL_UNVERIFIED_MODEL_ONLY_SEMANTIC_VETO_PASS",
                "product_identity_reason": parsed.get("interpretation"),
                "product_identity_title": material,
                "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
                "proxy_representativeness": "TERMINAL_MODEL_ONLY_NO_DATABASE_ANCHOR",
                "peer_values_json": json.dumps([central], ensure_ascii=False),
                "terminal_quality_flag": "LOWEST_CONFIDENCE_MODEL_ONLY",
            }

        # Parser rescue only: this does not broaden semantic acceptance. The same
        # deterministic product-family veto is applied by _parse_terminal_minimal_factor.
        working = {
            "material": material,
            "target_context": self.target_geography,
            "requested_reference_unit": preferred,
            "required_indicator": _required_indicator_policy(material),
            "deterministic_material_family": classify_material(material),
            "identity_validation_requirement": "product_interpretation must explicitly describe this family; do not merely echo or re-expand an acronym",
            "status": "Parser-recovery only for one terminal unverified A1-A3 value.",
            "repair_instruction": "Preserve the exact product identity; JSON only.",
        }
        minimal_attempts = []
        for attempt_no in range(1, 3):
            raw = _call_matcher(self.matcher, TERMINAL_MINIMAL_EF_SYSTEM_PROMPT, working, max_new_tokens=260)
            parsed_min, min_reason = self._parse_terminal_minimal_factor(raw, material, preferred)
            minimal_attempts.append({
                "attempt": attempt_no,
                "accepted": parsed_min is not None,
                "reason": min_reason,
                "raw_model_output": raw,
            })
            if parsed_min is not None:
                central = float(parsed_min["central"])
                self._last_fallback_attempts.append({
                    "stage": "CLASS5D_TERMINAL_MINIMAL_PARSER_RESCUE",
                    "accepted": True,
                    "attempts": minimal_attempts,
                    "repeated_value_consensus_used": False,
                })
                return {
                    "ef_value": central,
                    "reference_unit": preferred,
                    "lower_value": None,
                    "upper_value": None,
                    "verification": "UNVERIFIED_FALLBACK_ESTIMATE",
                    "source_class": "UNVERIFIED_FALLBACK_ESTIMATE",
                    "fallback_method": "TERMINAL_MINIMAL_SINGLE_LLM_VALUE_SEMANTIC_VETO",
                    "source_title": "Qwen terminal minimal-schema model-only estimate",
                    "source_url": None,
                    "source_geography": "Unverified model assumption; target context " + self.target_geography,
                    "source_year": None,
                    "search_tier": "FINAL_TERMINAL_MINIMAL_SINGLE_VALUE",
                    "search_query": None,
                    "evidence_quote": None,
                    "declared_unit_evidence": None,
                    "boundary": "A1-A3 / cradle-to-gate model assumption",
                    "indicator": parsed_min["indicator"],
                    "declared_impact_value": central,
                    "declared_impact_unit": "kg CO2e",
                    "declared_quantity": 1.0,
                    "declared_unit": preferred,
                    "uncertainty_type": "TERMINAL_MODEL_NO_VALID_RANGE",
                    "uncertainty_lower_value": None,
                    "uncertainty_upper_value": None,
                    "uncertainty_gsd": None,
                    "uncertainty_cv": None,
                    "uncertainty_confidence_level": None,
                    "uncertainty_evidence_quote": None,
                    "reason": (
                        "All anchored Class-4 routes failed and the richer terminal schema could not be validated. "
                        "A minimal single model-only value was retained after the same semantic identity veto. It is the lowest-confidence output and repeated-value consensus was not used."
                    ),
                    "raw_model_output": raw,
                    "fallback_attempts_json": json.dumps(self._last_fallback_attempts, ensure_ascii=False),
                    "guardrail_version": GUARDRAIL_VERSION,
                    "guardrail_status": "TERMINAL_MINIMAL_SINGLE_VALUE_INTERPRETATION_FIRST_IDENTITY_PASS",
                    "guardrail_reason": "LOWEST_CONFIDENCE_MODEL_ONLY_NO_RANGE",
                    "selected_result_id": None,
                    "external_match_type": "UNVERIFIED_FALLBACK_ESTIMATE",
                    "external_proxy_basis": None,
                    "product_identity_status": "TERMINAL_UNVERIFIED_MODEL_ONLY_SEMANTIC_VETO_PASS",
                    "product_identity_reason": parsed_min.get("interpretation"),
                    "product_identity_title": material,
                    "biogenic_carbon_policy": ("EXCLUDE_STORAGE_USE_GWP_GHG" if _requires_no_biogenic_storage(material) else None),
                    "proxy_representativeness": "TERMINAL_MINIMAL_MODEL_ONLY_NO_DATABASE_ANCHOR",
                    "peer_values_json": json.dumps([central]),
                    "terminal_quality_flag": "LOWEST_CONFIDENCE_MODEL_ONLY_NO_RANGE",
                }
            working = {
                **working,
                "previous_validation_error": min_reason,
                "previous_output": (raw or "")[-1000:],
                "repair_instruction": "Return the exact minimal JSON schema only and preserve the requested product family.",
            }

        self._last_fallback_attempts.append({
            "stage": "CLASS5D_TERMINAL_MINIMAL_PARSER_RESCUE",
            "accepted": False,
            "attempts": minimal_attempts,
            "reason": "no_semantically_valid_parseable_positive_terminal_value",
        })
        return None

    def resolve_record(self, rec: dict[str, Any]):
        item_id = _clean(rec.get("item_id") or rec.get("ID")) or "ITEM"
        original_material = _clean(rec.get("original_material") or rec.get("Material"))
        normalized_material = _clean(rec.get("normalized_material"))
        material, material_basis = choose_resolution_material(original_material, normalized_material)
        approved = _as_bool(rec.get("production_approved"))
        structured_valid = _as_bool(rec.get("structured_output_valid"))
        result = {
            "external_ef_resolution_status": "NOT_APPLICABLE_ELCD_APPROVED" if approved else "UNRESOLVED",
            "external_ef_value": None,
            "external_ef_reference_unit": None,
            "external_ef_lower_value": None,
            "external_ef_upper_value": None,
            "external_ef_uncertainty_type": None,
            "external_ef_uncertainty_lower_value": None,
            "external_ef_uncertainty_upper_value": None,
            "external_ef_uncertainty_gsd": None,
            "external_ef_uncertainty_cv": None,
            "external_ef_uncertainty_confidence_level": None,
            "external_ef_uncertainty_evidence_quote": None,
            "external_ef_peer_values_json": None,
            "external_ef_peer_sources_json": None,
            "external_ef_unit_basis_records_json": None,
            "external_ef_relaxed_sources_json": None,
            "external_ef_relaxed_source_count": None,
            "external_ef_relaxed_retained_count": None,
            "external_ef_relaxed_outlier_count": None,
            "external_ef_relaxed_consensus_method": None,
            "external_ef_relaxed_consensus_version": None,
            "external_ef_verification": "NOT_APPLICABLE" if approved else "UNRESOLVED",
            "external_ef_source_class": None,
            "external_ef_source_title": None,
            "external_ef_source_url": None,
            "external_ef_source_geography": None,
            "external_ef_source_year": None,
            "external_ef_search_tier": None,
            "external_ef_search_query": None,
            "external_ef_evidence_quote": None,
            "external_ef_declared_unit_evidence": None,
            "external_ef_boundary": None,
            "external_ef_indicator": None,
            "external_ef_reason": None,
            "external_ef_match_type": None,
            "external_ef_proxy_basis": None,
            "external_ef_product_identity_status": None,
            "external_ef_product_identity_reason": None,
            "external_ef_product_identity_title": None,
            "external_ef_material_query": material,
            "external_ef_material_selection_basis": material_basis,
            "external_ef_trigger": ("BIOGENIC_POLICY_EXTERNAL_REQUIRED" if _requires_no_biogenic_storage(material) else ("ELCD_REVIEW_REQUIRED" if structured_valid else "INVALID_QWEN_OUTPUT")),
            "external_ef_material_family": classify_material(material),
            "external_ef_material_taxonomy_version": MATERIAL_TAXONOMY_VERSION,
            "external_ef_biogenic_carbon_policy": None,
            "external_ef_proxy_representativeness": None,
            "external_ef_fallback_method": None,
            "external_ef_terminal_quality_flag": None,
            "external_ef_details_json": None,
            "external_ef_fallback_attempts_json": None,
            "external_ef_guardrail_status": None,
            "external_ef_guardrail_reason": None,
            "external_ef_resolver_version": EXTERNAL_EF_RESOLVER_VERSION,
            "external_ef_guardrail_version": GUARDRAIL_VERSION,
            "external_ef_evidence_cache_version": CACHE_VERSION,
            "external_ef_resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if approved and not _requires_no_biogenic_storage(material):
            return result, []
        self._last_fallback_attempts = []
        # A malformed Qwen response is still a Review Required outcome.  The
        # external resolver is independent of the model selection stage, so it may
        # continue from the original BOM description instead of discarding the row.
        resolved, evidence = self._resolve_one(item_id, material)
        source_unit = rec.get("unit") or rec.get("Unit")

        # Class 4 fallback is database-anchored first. Qwen may expand terminology or
        # rank supplied catalog processes, but it does not generate the numerical
        # factor for these preferred fallback routes. This specifically prevents
        # repeated model hallucinations from being treated as numerical validation.
        if resolved is None and (self.allow_llm_unverified_estimate or self.allow_conservative_analog_estimate):
            self._progress(f"EF Class 4 · Method A · dynamic same-family frozen-database proxy · {material}")
            resolved = self._estimate_dynamic_frozen_database_proxy(material, source_unit)
        if resolved is None and (self.allow_llm_unverified_estimate or self.allow_conservative_analog_estimate):
            self._progress(f"EF Class 4 · Method B · dynamic semantic-analog frozen-database proxy · {material}")
            resolved = self._estimate_dynamic_frozen_analog_proxy(material, source_unit)

        # Only after source-supported and both database-anchored Class-4 routes
        # fail is a free-form model number attempted. It is explicitly terminal,
        # excluded from verified subtotal, and must pass the
        # deterministic product-family semantic veto in addition to unit/boundary
        # validation. No repeated-value consensus is treated as verification.
        if resolved is None and self.allow_llm_unverified_estimate:
            self._progress(f"EF Class 4 · Method C · terminal model-only fallback with semantic veto · {material}")
            resolved = self._estimate_terminal_llm_only_value(material, source_unit)
        if resolved is None:
            # Ordinary evidence scarcity should not reach this branch. It is reserved
            # for unsupported/invalid units or complete inability to obtain any
            # parseable finite positive terminal model value after all retries.
            result["external_ef_resolution_status"] = "INPUT_OR_MODEL_FAILURE"
            result["external_ef_verification"] = "INPUT_OR_MODEL_FAILURE"
            result["external_ef_reason"] = "No usable source-supported factor, dynamic frozen-database proxy, or semantically valid terminal model-only value could be obtained; see fallback_attempts_json."
            result["external_ef_fallback_attempts_json"] = json.dumps(self._last_fallback_attempts, ensure_ascii=False)
            return result, evidence

        result.update({
            "external_ef_value": resolved.get("ef_value"),
            "external_ef_reference_unit": resolved.get("reference_unit"),
            "external_ef_lower_value": resolved.get("lower_value"),
            "external_ef_upper_value": resolved.get("upper_value"),
            "external_ef_uncertainty_type": resolved.get("uncertainty_type"),
            "external_ef_uncertainty_lower_value": resolved.get("uncertainty_lower_value"),
            "external_ef_uncertainty_upper_value": resolved.get("uncertainty_upper_value"),
            "external_ef_uncertainty_gsd": resolved.get("uncertainty_gsd"),
            "external_ef_uncertainty_cv": resolved.get("uncertainty_cv"),
            "external_ef_uncertainty_confidence_level": resolved.get("uncertainty_confidence_level"),
            "external_ef_uncertainty_evidence_quote": resolved.get("uncertainty_evidence_quote"),
            "external_ef_peer_values_json": resolved.get("peer_values_json"),
            "external_ef_peer_sources_json": resolved.get("peer_sources_json"),
            "external_ef_unit_basis_records_json": resolved.get("unit_basis_records_json"),
            "external_ef_relaxed_sources_json": resolved.get("relaxed_sources_json"),
            "external_ef_relaxed_source_count": resolved.get("relaxed_source_count"),
            "external_ef_relaxed_retained_count": resolved.get("relaxed_retained_count"),
            "external_ef_relaxed_outlier_count": resolved.get("relaxed_outlier_count"),
            "external_ef_relaxed_consensus_method": resolved.get("relaxed_consensus_method"),
            "external_ef_relaxed_consensus_version": resolved.get("relaxed_consensus_version"),
            "external_ef_verification": ("EXTERNAL_VERIFIED" if resolved.get("verification") in {"EXTERNAL_VERIFIED", "PROVISIONAL_SOURCE_SUPPORTED"} else resolved.get("verification")),
            "external_ef_verification_tier": resolved.get("verification_tier") or ("RELAXED" if resolved.get("verification") == "PROVISIONAL_SOURCE_SUPPORTED" else None),
            "external_ef_source_class": ("EXTERNAL_VERIFIED_RELAXED" if resolved.get("verification") == "PROVISIONAL_SOURCE_SUPPORTED" else resolved.get("source_class")),
            "external_ef_source_title": resolved.get("source_title"),
            "external_ef_source_url": resolved.get("source_url"),
            "external_ef_source_geography": resolved.get("source_geography"),
            "external_ef_source_year": resolved.get("source_year"),
            "external_ef_search_tier": resolved.get("search_tier"),
            "external_ef_search_query": resolved.get("search_query"),
            "external_ef_evidence_quote": resolved.get("evidence_quote"),
            "external_ef_declared_unit_evidence": resolved.get("declared_unit_evidence"),
            "external_ef_boundary": resolved.get("boundary"),
            "external_ef_indicator": resolved.get("indicator"),
            "external_ef_reason": resolved.get("reason"),
            "external_ef_match_type": resolved.get("external_match_type"),
            "external_ef_proxy_basis": resolved.get("external_proxy_basis"),
            "external_ef_product_identity_status": resolved.get("product_identity_status"),
            "external_ef_product_identity_reason": resolved.get("product_identity_reason"),
            "external_ef_product_identity_title": resolved.get("product_identity_title"),
            "external_ef_material_family": resolved.get("material_family") or classify_material(material),
            "external_ef_material_taxonomy_version": resolved.get("material_taxonomy_version") or MATERIAL_TAXONOMY_VERSION,
            "external_ef_biogenic_carbon_policy": resolved.get("biogenic_carbon_policy"),
            "external_ef_proxy_representativeness": resolved.get("proxy_representativeness"),
            "external_ef_fallback_method": resolved.get("fallback_method"),
            "external_ef_terminal_quality_flag": resolved.get("terminal_quality_flag"),
            "external_ef_details_json": json.dumps(resolved, ensure_ascii=False),
            "external_ef_fallback_attempts_json": resolved.get("fallback_attempts_json"),
            "external_ef_guardrail_status": resolved.get("guardrail_status"),
            "external_ef_guardrail_reason": resolved.get("guardrail_reason"),
        })
        if resolved.get("verification") in {"EXTERNAL_VERIFIED", "PROVISIONAL_SOURCE_SUPPORTED"}:
            result["external_ef_resolution_status"] = "RESOLVED_EXTERNAL_VERIFIED"
        elif resolved.get("verification") == "UNVERIFIED_FALLBACK_ESTIMATE":
            result["external_ef_resolution_status"] = "RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE"
        else:
            result["external_ef_resolution_status"] = "INPUT_OR_MODEL_FAILURE"
            result["external_ef_verification"] = "INPUT_OR_MODEL_FAILURE"
        return result, evidence

    def resolve_batch(self, df: pd.DataFrame):
        rows = []
        evidence_rows = []
        records = df.to_dict(orient="records")
        total = len(records)
        for i, rec in enumerate(records, start=1):
            material = _clean(rec.get("normalized_material") or rec.get("original_material") or rec.get("Material"))
            self._progress(f"EF resolver · row {i}/{total} · {material}")
            resolved, evidence = self.resolve_record(rec)
            merged = dict(rec)
            merged.update(resolved)
            rows.append(merged)
            evidence_rows.extend(evidence)
            status = resolved.get("external_ef_verification") or resolved.get("external_ef_resolution_status")
            self._progress(f"EF resolver · row {i}/{total} complete · {material} · {status}")
        return pd.DataFrame(rows), pd.DataFrame(evidence_rows)
