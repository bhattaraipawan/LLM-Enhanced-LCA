"""Automatic physical-property resolution for standardized BOM quantities.

The resolver is used only when a deterministic unit conversion needs a physical
property that the user did not provide (for example density for m3 -> kg).

Resolution hierarchy:
1. User/project value already present in the BOM.
2. Value explicitly encoded in the BOM description (currently thickness in mm).
3. Live traceable web retrieval, searched Nepal first, then South Asia, Asia, global.
4. If strict verification fails, a source-supported relaxed-phase property may be
   formed only from explicit numerical values in retrieved evidence.
5. If no retrieved value survives and the conversion still requires a property,
   a final UNVERIFIED_FALLBACK_ESTIMATE may be used with an explicit broad model-estimated
   range. It is never treated as traceable evidence.

No case-study-specific density, thickness, mass-per-item value, source URL, or
preselected reference is stored in this resolver.

For verified and source-supported relaxed-phase properties, Qwen is an evidence
extractor only and every value must be supported by retrieved text. The final
LLM-only property fallback is isolated and explicitly marked unverified. Search
evidence and the final fallback assumptions are retained for auditability.
"""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
import re
import time
from typing import Any

import pandas as pd

from unit_conversion import convert_quantity, norm_unit
from evidence_consensus import (
    robust_positive_consensus, CONSENSUS_METHOD_VERSION, robust_model_ensemble_consensus,
    MODEL_ENSEMBLE_SIZE, MODEL_ENSEMBLE_METHOD_VERSION,
)
from semantic_analog import infer_analog_plan, SEMANTIC_ANALOG_VERSION
from technical_equivalence import infer_technical_equivalents, TECHNICAL_EQUIVALENCE_VERSION
from evidence_cache import RuntimeEvidenceCache, CACHE_VERSION
from material_taxonomy import MATERIAL_TAXONOMY_VERSION, classify_material, choose_resolution_material, external_title_compatibility
from guardrails import GUARDRAIL_VERSION, property_plausible, property_guardrail_context

PROPERTY_RESOLVER_VERSION = "11.0-four-class-strict-relaxed-external"
TARGET_GEOGRAPHY = "Nepal"

TRACEABLE_SYSTEM_PROMPT = """You are a technical evidence extractor supporting an LCA unit-conversion workflow.

You are NOT allowed to invent a physical-property value, source, citation, URL, year, geography, or evidence quote.
You receive web-search evidence that has already been retrieved for one material property.

Rules:
1. Use ONLY the supplied evidence candidates.
2. A value is acceptable only when the evidence explicitly supports the requested material/product AND the requested physical property.
3. Physical condition is mandatory. If the requested property is BULK DENSITY for aggregate/stone/soil, reject particle density, true density, specific gravity, solid-rock density, or generic material density unless the evidence explicitly states bulk/unit-weight conditions consistent with a bulk BOM volume.
4. Do not treat a downstream or different product as equivalent.
5. source_result_id must exactly match one supplied result ID.
6. evidence_quote MUST be copied verbatim from the supplied snippet or extracted text. Never reconstruct or paraphrase the quote.
7. Do not infer geography from the search query. Geography must be supported by the source itself.
8. If no supplied evidence gives a defensible explicit value, return found=false. Do not estimate.
9. For density return kg/m3. For mass-per-item return kg/item. For thickness return mm.
10. If a source provides a defensible range, return lower_value and upper_value. Otherwise use null.
11. Return JSON only with exactly these keys:
found, value, lower_value, upper_value, unit, source_result_id, source_geography, source_year, source_class, evidence_quote, reason
"""

PROVISIONAL_PROPERTY_SYSTEM_PROMPT = """You are a numerical engineering evidence extractor.

Strict source verification has already failed. Extract only physical-property
values that are explicitly present in the supplied retrieved evidence. Never
estimate from memory.

Rules:
1. Use ONLY supplied evidence candidates.
2. Evaluate every result_id independently.
3. The source text must explicitly support the requested material/product and
   requested property.
4. For bulk density, reject particle density, true density, specific gravity, or
   solid-material density unless the evidence explicitly describes bulk/unit-weight
   conditions consistent with a bulk BOM volume.
5. Return the value only in the requested standardized unit. Do not invent a unit
   conversion; if the supplied text does not provide a directly interpretable value
   in that unit, return found=false for that candidate.
6. evidence_quote must be copied verbatim and include/support the numerical value.
7. Do not average or choose a final value. Python will combine independently
   supported values with a robust median/outlier procedure.
8. Return JSON only with exactly one top-level key, records. Each record must have
   exactly: source_result_id, found, value, lower_value, upper_value, unit,
   source_year, evidence_quote, reason
"""

LLM_UNVERIFIED_PROPERTY_SYSTEM_PROMPT = """You are the model-only exploratory physical-property fallback in a construction LCA unit-conversion workflow.

Traceable and source-supported property evidence failed. Produce exactly FIVE independent candidate estimates for the requested material itself. Python will perform material-agnostic robust consensus; no material-specific density, thickness, or item-mass range is supplied.

Rules:
1. Do not claim a source, URL, citation, standard, year, or evidence quote.
2. Estimate ONLY requested_property.
3. Every candidate must use required_standardized_unit EXACTLY.
4. Each candidate must have finite positive lower_value < central_value < upper_value.
5. Use the supplied estimation_lenses to make genuinely distinct engineering estimates; do not clone one number five times merely to force agreement.
6. For bulk density estimate bulk/unit-weight conditions for the BOM volume, not particle density, true density, or specific gravity.
7. No expected numerical range is supplied. Python rejects disagreement after generation.
8. Return JSON only with exactly one top-level key: candidates.
9. candidates must contain exactly five objects, each with exactly these keys:
found, central_value, lower_value, upper_value, unit, material_interpretation, estimation_basis, rationale
"""

CONSERVATIVE_ANALOG_PROPERTY_SYSTEM_PROMPT = """You are the final analog-estimation stage for a construction LCA physical-property conversion.

The exact-product five-candidate estimate did not agree. You are given a semantic family and exactly five analog products inferred dynamically from the material description. Estimate ONE candidate for EACH supplied analog. Python will reject outliers and require at least three agreeing candidates.

Rules:
1. Do not claim a source, citation, URL, standard, or publication year.
2. Use only the supplied analogs; there is no hard-coded material-to-analog mapping.
3. Estimate only requested_property and use required_standardized_unit EXACTLY.
4. Each candidate must have finite positive lower_value < central_value < upper_value.
5. For bulk density use bulk/unit-weight conditions, not particle density or specific gravity.
6. Preserve each supplied analog_material name exactly.
7. Do not force agreement; Python performs the consensus test.
8. Return JSON only with exactly one top-level key: candidates.
9. candidates must contain exactly five objects, each with exactly these keys:
found, central_value, lower_value, upper_value, unit, analog_material, estimation_basis, rationale
"""

LLM_UNVERIFIED_PROPERTY_SINGLE_SYSTEM_PROMPT = """You are producing ONE independent exploratory physical-property estimate for a construction material.

No material-specific density, thickness, or mass-per-item value/range is supplied. Do not claim a source, citation, URL, standard, or publication year. Use only the single estimation_lens supplied.

Return exactly one JSON object with these keys:
found, central_value, lower_value, upper_value, unit, material_interpretation, estimation_basis, rationale

Rules:
- Estimate only requested_property.
- Use required_standardized_unit exactly.
- lower_value < central_value < upper_value and all values must be finite and positive.
- For bulk density, use bulk/unit-weight conditions appropriate to BOM volume, not particle density/specific gravity.
- Do not try to force agreement with other candidates.
- JSON only.
"""

CONSERVATIVE_ANALOG_PROPERTY_SINGLE_SYSTEM_PROMPT = """You are producing ONE independent exploratory physical-property estimate for ONE dynamically inferred analog product.

No material-specific numeric value/range or hard-coded material-to-analog mapping is supplied. Do not claim a source, citation, URL, standard, or year.

Return exactly one JSON object with these keys:
found, central_value, lower_value, upper_value, unit, analog_material, estimation_basis, rationale

Rules:
- Preserve analog_material exactly.
- Estimate only requested_property in required_standardized_unit exactly.
- lower_value < central_value < upper_value and all values must be finite and positive.
- For bulk density, use bulk/unit-weight conditions, not particle density/specific gravity.
- Do not force agreement with other analogs.
- JSON only.
"""
TERMINAL_MINIMAL_PROPERTY_SYSTEM_PROMPT = """You are the last numerical fallback for one physical property needed only to convert a construction BOM quantity.

No material-specific property value, density range, item mass, thickness, source, or expected magnitude is supplied. The result will be explicitly labeled UNVERIFIED and used only in the separated exploratory calculation. Do not claim a source.

Return JSON only with exactly these keys:
found, value, unit, material_interpretation, rationale

Rules:
- Estimate only requested_property.
- Use required_standardized_unit exactly.
- value must be finite and positive.
- For bulk density, estimate the bulk/unit-weight condition appropriate to the BOM volume rather than particle density or specific gravity.
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
    try:
        return matcher.generate_with_system(system_prompt, payload, max_new_tokens_override=max_new_tokens)
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


def extract_thickness_mm(material: str) -> float | None:
    """Extract an explicit thickness such as '19mm' or '0.45 mm' from BOM text."""
    text = _clean(material).lower().replace("millimetres", "mm").replace("millimeters", "mm")
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm\b", text)
    if not m:
        return None
    value = float(m.group(1))
    return value if value > 0 else None


_BULK_VOLUME_TERMS = (
    "gravel", "aggregate", "crushed stone", "stone", "soil", "earth",
    "mud", "backfill", "ballast", "sand",
)

_LOW_QUALITY_HOSTS = (
    "scribd.com", "pinterest.", "quora.", "reddit.",
    "wisdomanswer.com", "blog.welcu.com", "leukstekadotip.nl",
)

_RECOGNIZED_TECHNICAL_HOSTS = (
    "istructe.org", "inbar.int", "apawood.org", "astm.org",
    "iso.org", "bsigroup.com", "ccaa.com.au",
)

_PEER_REVIEWED_HOSTS = (
    "sciencedirect.com", "springer.com", "springeropen.com",
    "wiley.com", "tandfonline.com", "mdpi.com", "sagepub.com",
    "nature.com",
)


def _property_label(kind: str, material: str | None = None) -> tuple[str, str]:
    if kind == "density_kg_m3":
        m = _clean(material).lower()
        if any(term in m for term in _BULK_VOLUME_TERMS):
            return (
                "bulk density / bulk unit weight for the material volume as supplied in the BOM",
                "kg/m3",
            )
        return "material density", "kg/m3"
    if kind == "mass_per_item_kg":
        return "mass per item", "kg/item"
    if kind == "thickness_mm":
        return "product thickness", "mm"
    raise ValueError(f"Unsupported property kind: {kind}")


def _url_host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _source_quality(url: str, title: str = "") -> tuple[bool, str]:
    host = _url_host(url)
    if not host:
        return False, "UNKNOWN_SOURCE"
    if any(bad in host for bad in _LOW_QUALITY_HOSTS):
        return False, "LOW_QUALITY_WEB"
    if host.endswith(".gov.np") or host == "gov.np":
        return True, "NEPAL_GOVERNMENT"
    if ".gov." in host or host.endswith(".gov"):
        return True, "GOVERNMENT"
    if ".edu." in host or host.endswith(".edu") or ".ac." in host:
        return True, "ACADEMIC"
    if any(host.endswith(h) or host == h for h in _RECOGNIZED_TECHNICAL_HOSTS):
        return True, "RECOGNIZED_TECHNICAL_ORGANIZATION"
    if any(host.endswith(h) or host == h for h in _PEER_REVIEWED_HOSTS):
        return True, "PEER_REVIEWED_PUBLISHER"
    return False, "UNVERIFIED_WEB_SOURCE"




def _property_identity_ok(material: str, selected: dict[str, Any]) -> tuple[bool, str]:
    """Deterministically reject physical-property evidence for the wrong product family."""
    requested_family = classify_material(material)
    evidence = _clean(selected.get("title")) + " " + _clean(selected.get("snippet"))
    evidence_family = classify_material(evidence)

    if requested_family != "UNKNOWN":
        if evidence_family == requested_family:
            return True, "material_family_match"
        if evidence_family != "UNKNOWN":
            return False, f"wrong_material_family:{evidence_family}"

        # When the taxonomy cannot classify the source text, require at least
        # one meaningful token from the material description to be present.
        stop = {"mm", "kg", "m", "grade", "local", "natural", "ordinary", "for", "and", "the"}
        tokens = [t for t in re.findall(r"[a-z0-9]+", _clean(material).lower()) if len(t) >= 4 and t not in stop]
        low = evidence.lower()
        if any(t in low for t in tokens):
            return True, "material_token_match"
        return False, "material_identity_not_supported"

    tokens = [t for t in re.findall(r"[a-z0-9]+", _clean(material).lower()) if len(t) >= 4]
    low = evidence.lower()
    if tokens and any(t in low for t in tokens):
        return True, "unknown_family_token_match"
    return False, "unknown_family_identity_not_supported"

def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", _clean(text)).strip().lower()


def _quote_supported(quote: str, selected: dict[str, Any]) -> bool:
    q = _normalize_quote(quote)
    if not q:
        return False
    evidence = _normalize_quote(
        f"{selected.get('snippet', '')} {selected.get('excerpt', '')}"
    )
    return q in evidence


def _infer_geography(selected: dict[str, Any]) -> str:
    host = _url_host(selected.get("url", ""))
    evidence = _normalize_quote(
        f"{selected.get('title', '')} {selected.get('snippet', '')} "
        f"{selected.get('excerpt', '')}"
    )
    if host.endswith(".np") or "nepal" in evidence:
        return "Nepal"
    if host.endswith(".in") or "india" in evidence:
        return "India"
    if host.endswith(".bd") or "bangladesh" in evidence:
        return "Bangladesh"
    if host.endswith(".lk") or "sri lanka" in evidence:
        return "Sri Lanka"
    if host.endswith(".pk") or "pakistan" in evidence:
        return "Pakistan"
    if host.endswith(".bt") or "bhutan" in evidence:
        return "Bhutan"
    if host.endswith(".au") or "australia" in evidence:
        return "Australia"
    if host.endswith(".uk") or "united kingdom" in evidence:
        return "United Kingdom"
    return "Unspecified"


def _tier_geography_ok(tier: str, geography: str) -> bool:
    if tier == "NEPAL":
        return geography == "Nepal"
    if tier == "SOUTH_ASIA":
        return geography in {
            "Nepal", "India", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan"
        }
    if tier == "ASIA":
        # Accept only explicitly inferred Asian geography before the global tier.
        return geography in {
            "Nepal", "India", "Bangladesh", "Sri Lanka", "Pakistan", "Bhutan"
        }
    return True


def _target_geography_supported(selected: dict[str, Any], target_geography: str) -> bool:
    """Strict Class-3/traceable-property geography test based on source evidence.

    The search query is never used as geography evidence. relaxed External Verified
    properties intentionally bypass this exact-geography gate.
    """
    target = _normalize_quote(target_geography)
    if not target:
        return False
    inferred = _normalize_quote(_infer_geography(selected))
    if inferred and inferred != "unspecified" and inferred == target:
        return True
    evidence = _normalize_quote(
        f"{selected.get('title','')} {selected.get('snippet','')} {selected.get('excerpt','')}"
    )
    host = _url_host(selected.get('url',''))
    padded = f" {evidence} "
    aliases = {
        "nepal": ("nepal",), "india": ("india",),
        "bangladesh": ("bangladesh",), "sri lanka": ("sri lanka",),
        "pakistan": ("pakistan",), "bhutan": ("bhutan",),
        "australia": ("australia",),
        "united kingdom": ("united kingdom", " uk "),
        "united states": ("united states", " usa ", " u.s. "),
        "germany": ("germany",), "norway": ("norway",),
    }
    if any(a in padded for a in aliases.get(target, (target,))):
        return True
    suffixes = {
        "nepal": ".np", "india": ".in", "bangladesh": ".bd",
        "sri lanka": ".lk", "pakistan": ".pk", "bhutan": ".bt",
        "australia": ".au", "united kingdom": ".uk",
        "united states": ".us", "germany": ".de", "norway": ".no",
    }
    suffix = suffixes.get(target)
    return bool(suffix and host.endswith(suffix))


def _bulk_density_evidence_ok(material: str, selected: dict[str, Any]) -> bool:
    m = _clean(material).lower()
    if not any(term in m for term in _BULK_VOLUME_TERMS):
        return True
    evidence = _normalize_quote(
        f"{selected.get('snippet', '')} {selected.get('excerpt', '')}"
    )
    good_terms = (
        "bulk density", "bulk unit weight", "loose density",
        "rodded density", "compacted density", "unit weight",
    )
    bad_only_terms = (
        "particle density", "specific gravity", "true density",
        "solid density", "rock density",
    )
    if any(term in evidence for term in good_terms):
        return True
    if any(term in evidence for term in bad_only_terms):
        return False
    # For granular/earth materials, generic "density" without bulk condition
    # is not strong enough for a volume-to-mass BOM conversion.
    return False


def _expected_unit_ok(kind: str, unit: str) -> bool:
    u = _clean(unit).lower().replace("³", "3").replace("^3", "3").replace(" ", "")
    if kind == "density_kg_m3":
        return u in {"kg/m3", "kgperm3", "kgm-3"}
    if kind == "mass_per_item_kg":
        return u in {"kg/item", "kg/piece", "kg/each", "kg"}
    if kind == "thickness_mm":
        return u in {"mm", "millimeter", "millimetre"}
    return False


def _number_close_in_text(value: float, text: str) -> bool:
    for token in re.findall(r"[-+]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text or ""):
        s = token.replace("−", "-").replace("–", "-")
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            parsed = float(s)
        except Exception:
            continue
        tol = max(1e-9, abs(float(value)) * 1e-6)
        if abs(parsed - float(value)) <= tol:
            return True
    return False


def _property_unit_supported_in_quote(kind: str, quote: str) -> bool:
    q = _normalize_quote(quote).replace("³", "3").replace("^3", "3")
    if kind == "density_kg_m3":
        return any(x in q for x in ("kg/m3", "kg per m3", "kg m-3", "kgm-3"))
    if kind == "mass_per_item_kg":
        return "kg" in q and any(x in q for x in ("item", "piece", "each", "unit", "kg"))
    if kind == "thickness_mm":
        return "mm" in q or "millimeter" in q or "millimetre" in q
    return False


def _provisional_source_allowed(url: str, title: str = "") -> tuple[bool, str]:
    host = _url_host(url)
    if not host:
        return False, "UNKNOWN_SOURCE"
    if any(bad in host for bad in _LOW_QUALITY_HOSTS):
        return False, "LOW_QUALITY_WEB"
    strict_ok, strict_class = _source_quality(url, title)
    if strict_ok:
        return True, strict_class
    return True, "SOURCE_SUPPORTED_WEB"



def _promising_property_candidate(material: str, kind: str, candidate: dict[str, Any]) -> tuple[bool, str]:
    """Value-free trigger for a deeper relaxed-phase Class-3 property search."""
    allowed, source_class = _provisional_source_allowed(candidate.get("url", ""), candidate.get("title", ""))
    if not allowed:
        return False, f"source_not_allowed:{source_class}"
    identity_ok, identity_reason = external_title_compatibility(
        material, candidate.get("title"), match_type="DIRECT_PRODUCT"
    )
    if not identity_ok:
        return False, f"identity_not_supported:{identity_reason}"
    text = _normalize_quote(f"{candidate.get('snippet','')} {candidate.get('excerpt','')}")
    label, expected_unit = _property_label(kind, material)
    label_terms = [x for x in re.split(r"[^a-z0-9]+", _normalize_quote(label)) if len(x) > 2]
    has_label = any(x in text for x in label_terms)
    has_unit = _property_unit_supported_in_quote(kind, text) or _normalize_quote(expected_unit) in text
    return (has_label or has_unit), ("PROMISING_PROPERTY_SIGNAL" if (has_label or has_unit) else "no_property_signal")

def conversion_requirements(source_unit: str | None, ref_unit: str | None) -> list[str]:
    """Return physical properties required after pure deterministic conversions are tried."""
    src = norm_unit(source_unit)
    ref = norm_unit(ref_unit)
    if not src or not ref:
        return []

    probe = convert_quantity(1.0, src, ref)
    if probe.ok:
        return []

    mass_units = {"kg", "g", "t"}
    if src == "m3" and ref in mass_units:
        return ["density_kg_m3"]
    if src == "m2" and ref in mass_units:
        return ["thickness_mm", "density_kg_m3"]
    if src == "m2" and ref == "m3":
        return ["thickness_mm"]
    if src == "m3" and ref == "m2":
        return ["thickness_mm"]
    if src == "item" and ref in mass_units:
        return ["mass_per_item_kg"]
    if src in mass_units and ref == "m3":
        return ["density_kg_m3"]
    if src in mass_units and ref == "m2":
        return ["thickness_mm", "density_kg_m3"]
    if src in mass_units and ref == "item":
        return ["mass_per_item_kg"]
    return []


def _query_sets(material: str, kind: str, target_geography: str) -> list[tuple[str, list[str]]]:
    label, unit = _property_label(kind, material)
    m = material.strip()
    family = classify_material(material)
    family_term = family.lower().replace("_", " ") if family != "UNKNOWN" else None

    south_asia = [
        f'"{m}" {label} {unit} India Nepal construction standard',
        f'"{m}" {label} {unit} South Asia filetype:pdf',
    ]
    global_queries = [
        f'"{m}" {label} {unit} construction material technical data',
        f'"{m}" {label} {unit} filetype:pdf',
    ]
    if family_term and family_term.lower() != m.lower():
        # Same-family terminology broadens retrieval only; it contributes no
        # physical-property value and every extracted number still has to pass
        # product/property/unit/source checks.
        south_asia.insert(0, f'"{family_term}" {label} {unit} India construction technical data')
        global_queries.insert(0, f'"{family_term}" {label} {unit} construction material technical data')

    return [
        (
            "NEPAL",
            [
                f'"{m}" {label} {unit} {target_geography} construction',
                f'"{m}" {label} {unit} {target_geography} filetype:pdf',
                f'{m} {label} {unit} site:gov.np',
                f'{m} {label} {unit} Nepal engineering',
            ],
        ),
        ("SOUTH_ASIA", south_asia),
        ("ASIA", [f'"{m}" {label} {unit} Asia construction technical data']),
        ("GLOBAL", global_queries),
    ]



class WebPropertyResolver:
    def __init__(
        self,
        matcher,
        *,
        target_geography: str = TARGET_GEOGRAPHY,
        allow_source_supported_provisional: bool = True,
        allow_llm_unverified_estimate: bool = True,
        allow_conservative_analog_estimate: bool = True,
        search_results_per_query: int = 5,
        extract_top_n: int = 5,
        max_evidence_candidates: int = 5,
        excerpt_chars: int = 4500,
        timeout: int = 8,
        class3_source_budget: int = 3,
        class4_total_source_budget: int = 10,
        max_search_queries_per_property: int = 5,
        max_external_seconds_per_property: float = 60.0,
        adaptive_max_search_queries_per_property: int = 10,
        adaptive_total_source_budget: int = 9,
        adaptive_external_seconds_per_property: float = 120.0,
        pdf_max_pages: int = 20,
        adaptive_pdf_max_pages: int = 50,
        max_download_bytes: int = 12 * 1024 * 1024,
        progress_callback=None,
        evidence_cache_path: str | None = None,
    ):
        self.matcher = matcher
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
        self.max_search_queries_per_property = max(1, int(max_search_queries_per_property))
        self.max_external_seconds_per_property = max(5.0, float(max_external_seconds_per_property))
        self.adaptive_max_search_queries_per_property = max(self.max_search_queries_per_property, int(adaptive_max_search_queries_per_property))
        self.adaptive_total_source_budget = max(self.class4_total_source_budget, int(adaptive_total_source_budget))
        self.adaptive_external_seconds_per_property = max(self.max_external_seconds_per_property, float(adaptive_external_seconds_per_property))
        self.pdf_max_pages = max(1, int(pdf_max_pages))
        self.adaptive_pdf_max_pages = max(self.pdf_max_pages, int(adaptive_pdf_max_pages))
        self.max_download_bytes = max(1024 * 1024, int(max_download_bytes))
        self.progress_callback = progress_callback
        self.evidence_cache = RuntimeEvidenceCache(evidence_cache_path)
        self._search_cache: dict[str, list[dict[str, Any]]] = {}
        self._resolution_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._last_fallback_attempts: list[dict[str, Any]] = []
        self._relaxed_property_pool: list[dict[str, Any]] = []
        self._anchored_exact_property_pool: list[dict[str, Any]] = []

    def _progress(self, message: str) -> None:
        if callable(self.progress_callback):
            try:
                self.progress_callback(message)
            except Exception:
                pass

    def _ddgs(self):
        from ddgs import DDGS
        return DDGS(timeout=self.timeout)

    def _search(self, query: str) -> list[dict[str, Any]]:
        """Search defensively.

        Search-engine backends can legitimately return no results, time out,
        rate-limit, or fail transiently. None of those conditions should abort
        a building-LCA run. We try a small sequence of region settings and, if
        all attempts fail, cache an empty result so the resolver can continue
        to the next query/geography tier and ultimately an unresolved result or the explicit LLM
        fallback when the user has enabled it.
        """
        if query in self._search_cache:
            return [dict(x) for x in self._search_cache[query]]

        rows = []
        for region in ("np-en", "us-en"):
            try:
                result = self._ddgs().text(
                    query,
                    region=region,
                    safesearch="moderate",
                    max_results=self.search_results_per_query,
                    backend="auto",
                )
                rows = list(result or [])
                if rows:
                    break
            except Exception:
                # "No results found", timeout, backend errors, rate limits,
                # unsupported region values, etc. are treated as no evidence
                # for this attempt rather than fatal pipeline errors.
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

    def _extract(self, url: str) -> str:
        """Extract one bounded source document safely.

        pypdf is used only when the downloaded bytes begin with the PDF magic
        header, preventing HTML/PNG/access-denied bodies from generating PDF
        parser storms. No recursive crawling is performed for property sources.
        """
        request_timeout = (min(5, self.timeout), self.timeout)
        try:
            import requests
            resp = requests.get(
                url, timeout=request_timeout, allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 LCA-research-resolver/4.0"},
            )
            resp.raise_for_status()
            if len(resp.content) > self.max_download_bytes:
                return ""
            ctype = (resp.headers.get("content-type") or "").lower()
            if resp.content.lstrip().startswith(b"%PDF"):
                try:
                    import logging
                    logging.getLogger("pypdf").setLevel(logging.ERROR)
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(resp.content), strict=False)
                    parts = []
                    for page in reader.pages[: self.pdf_max_pages]:
                        try:
                            txt = page.extract_text() or ""
                        except Exception:
                            txt = ""
                        if txt:
                            parts.append(txt)
                    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
                    if text:
                        return text[: max(self.excerpt_chars, 5000)]
                except Exception:
                    return ""

            is_html = (
                "html" in ctype
                or resp.content.lstrip()[:100].lower().startswith((b"<!doctype", b"<html"))
                or b"<html" in resp.content[:2500].lower()
            )
            if is_html:
                try:
                    raw = resp.text
                except Exception:
                    raw = resp.content.decode("utf-8", errors="ignore")
                raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
                raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
                raw = re.sub(r"(?s)<[^>]+>", " ", raw)
                text = re.sub(r"\s+", " ", raw).strip()
                if text:
                    return text[: max(self.excerpt_chars, 5000)]
        except Exception:
            pass

        # One compact search-engine extraction fallback, then stop.
        try:
            extracted = self._ddgs().extract(url, fmt="text_rich")
            if isinstance(extracted, dict):
                text = extracted.get("content") or extracted.get("text") or json.dumps(extracted, ensure_ascii=False)
            else:
                text = str(extracted)
            text = re.sub(r"\s+", " ", text).strip()
            return text[: max(self.excerpt_chars, 5000)]
        except Exception:
            return ""

    def _collect_evidence(
        self, queries: list[str], tier: str, item_id: str, material: str, kind: str,
        *, max_candidates: int | None = None, max_queries: int | None = None,
        seen_urls: set[str] | None = None,
    ):
        seen = seen_urls if seen_urls is not None else set()
        limit = max(1, int(max_candidates or self.max_evidence_candidates))
        qlimit = max(1, int(max_queries or len(queries)))
        pool: list[dict[str, Any]] = []
        queries_used = 0
        for query in queries[:qlimit]:
            queries_used += 1
            for r in self._search(query):
                if r["url"] in seen or any(x["url"] == r["url"] for x in pool):
                    continue
                pool.append({**r, "query": query, "tier": tier})
            if len(pool) >= limit * 2:
                break

        def rank(c):
            ok, cls = _source_quality(c.get("url", ""), c.get("title", ""))
            class_rank = {
                "NEPAL_GOVERNMENT": 0, "GOVERNMENT": 1, "ACADEMIC": 2,
                "RECOGNIZED_TECHNICAL_ORGANIZATION": 3, "PEER_REVIEWED_PUBLISHER": 4,
            }.get(cls, 20)
            return (0 if ok else 1, class_rank)

        pool.sort(key=rank)
        candidates = pool[:limit]
        evidence_rows: list[dict[str, Any]] = []
        for i, c in enumerate(candidates):
            seen.add(c["url"])
            c["result_id"] = f"P{i+1}"
            self._progress(f"External property evidence · inspecting source {i+1}/{len(candidates)} · {material} · {kind}")
            c["full_text"] = self._extract(c["url"])
            c["excerpt"] = c["full_text"][: max(self.excerpt_chars, 2500)]
            identity_ok, identity_reason = _property_identity_ok(material, c)
            evidence_rows.append({
                "item_id": item_id, "material": material, "property_kind": kind,
                "search_tier": tier, "query": c["query"], "result_id": c["result_id"],
                "title": c["title"], "url": c["url"], "snippet": c["snippet"],
                "excerpt": c["excerpt"], "product_identity_ok": identity_ok,
                "product_identity_reason": identity_reason, "selected": False,
                "relaxed_candidate": False, "relaxed_retained": False,
                "relaxed_outlier": False, "relaxed_value": None,
                "relaxed_reason": None,
            })
        return candidates, evidence_rows, queries_used

    def _choose_traceable(self, material: str, kind: str, tier: str, candidates: list[dict[str, Any]]):
        if not candidates:
            return None, "no_candidates"
        label, expected_unit = _property_label(kind, material)
        payload = {
            "material": material,
            "target_geography": self.target_geography,
            "search_tier": tier,
            "requested_property": label,
            "required_standardized_unit": expected_unit,
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
        raw = _call_matcher(self.matcher, TRACEABLE_SYSTEM_PROMPT, payload, max_new_tokens=640)
        try:
            obj = json.loads(raw.strip())
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                return None, f"parse_error:{raw[:120]}"
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None, "parse_error"

        required_keys = {
            "found", "value", "lower_value", "upper_value", "unit",
            "source_result_id", "source_geography", "source_year",
            "source_class", "evidence_quote", "reason",
        }
        if set(obj.keys()) != required_keys:
            return None, "schema_invalid"
        if not bool(obj.get("found")):
            return None, _clean(obj.get("reason")) or "not_found"

        sid = _clean(obj.get("source_result_id"))
        by_id = {c["result_id"]: c for c in candidates}
        if sid not in by_id:
            return None, "source_id_invalid"
        value = _num(obj.get("value"))
        if value is None or value <= 0 or not _expected_unit_ok(kind, _clean(obj.get("unit"))):
            return None, "value_or_unit_invalid"

        selected = by_id[sid]

        # Deterministic QA: the LLM can extract, but it cannot promote weak
        # web pages, invent evidence, or assign geography from the query.
        source_ok, source_class = _source_quality(
            selected.get("url", ""),
            selected.get("title", ""),
        )
        if not source_ok:
            return None, f"source_quality_rejected:{source_class}"

        identity_ok, identity_reason = _property_identity_ok(material, selected)
        if not identity_ok:
            return None, f"product_identity_rejected:{identity_reason}"

        quote = _clean(obj.get("evidence_quote"))
        if not _quote_supported(quote, selected):
            return None, "evidence_quote_not_in_retrieved_source"

        geography = _infer_geography(selected)
        # A property can support a Class-3 row only when the property source
        # itself is explicitly representative of the target study geography.
        # Foreign/global/unspecified sources remain eligible in the relaxed External Verified phase.
        if not _target_geography_supported(selected, self.target_geography):
            return None, f"target_geography_not_verified:{geography or 'Unspecified'}"

        if kind == "density_kg_m3" and not _bulk_density_evidence_ok(
            material, selected
        ):
            return None, "density_condition_not_bulk_compatible"

        return {
            "kind": kind,
            "value": value,
            "lower_value": _num(obj.get("lower_value")),
            "upper_value": _num(obj.get("upper_value")),
            "unit": _clean(obj.get("unit")),
            "source_class": source_class,
            "verification": "TRACEABLE_WEB",
            "source_title": selected["title"],
            "source_url": selected["url"],
            "source_geography": geography,
            "source_year": _clean(obj.get("source_year")),
            "search_tier": tier,
            "search_query": selected["query"],
            "evidence_quote": quote,
            "reason": _clean(obj.get("reason")),
            "raw_model_output": raw,
            "selected_result_id": sid,
            "product_identity_status": "PASS",
            "product_identity_reason": identity_reason,
        }, "ok"

    def _extract_relaxed_candidates(self, material: str, kind: str, tier: str, candidates: list[dict[str, Any]]):
        """Extract relaxed-phase Class-3 physical properties one source at a time.

        This reduces JSON truncation and keeps every accepted value tied to one
        explicit retrieved quote. No material-specific property value is stored.
        """
        if not candidates:
            return []
        label, expected_unit = _property_label(kind, material)
        required = {
            "source_result_id", "found", "value", "lower_value", "upper_value",
            "unit", "source_year", "evidence_quote", "reason",
        }
        accepted = []
        for selected in candidates:
            payload = {
                "material": material, "target_geography": self.target_geography,
                "search_tier": tier, "requested_property": label,
                "required_standardized_unit": expected_unit,
                "evidence_candidates": [{
                    "result_id": selected["result_id"], "title": selected["title"],
                    "url": selected["url"], "search_snippet": selected["snippet"][:900],
                    "extracted_text": selected["excerpt"][: self.excerpt_chars],
                }],
            }
            raw = _call_matcher(
                self.matcher, PROVISIONAL_PROPERTY_SYSTEM_PROMPT, payload,
                max_new_tokens=480,
            )
            try:
                obj = json.loads(raw.strip())
            except Exception:
                m = re.search(r"\{.*\}", raw or "", flags=re.S)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    continue
            if set(obj.keys()) != {"records"} or not isinstance(obj.get("records"), list) or len(obj["records"]) != 1:
                continue
            rec = obj["records"][0]
            if not isinstance(rec, dict) or set(rec.keys()) != required:
                continue
            sid = _clean(rec.get("source_result_id"))
            if sid != selected.get("result_id") or not bool(rec.get("found")):
                continue
            value = _num(rec.get("value"))
            if value is None or value <= 0 or not _expected_unit_ok(kind, _clean(rec.get("unit"))):
                continue
            source_ok, source_class = _provisional_source_allowed(selected.get("url", ""), selected.get("title", ""))
            if not source_ok:
                continue
            identity_ok, identity_reason = _property_identity_ok(material, selected)
            if not identity_ok:
                continue
            quote = _clean(rec.get("evidence_quote"))
            if not _quote_supported(quote, selected) or not _number_close_in_text(value, quote):
                continue
            if not _property_unit_supported_in_quote(kind, quote):
                continue
            if kind == "density_kg_m3" and not _bulk_density_evidence_ok(material, selected):
                continue
            geography = _infer_geography(selected)
            accepted.append({
                "kind": kind, "value": float(value),
                "lower_value": _num(rec.get("lower_value")),
                "upper_value": _num(rec.get("upper_value")),
                "unit": _clean(rec.get("unit")), "source_class": source_class,
                "verification": "TRACEABLE_WEB_RELAXED",
                "source_title": selected.get("title"), "source_url": selected.get("url"),
                "source_geography": geography,
                "geography_representativeness": (
                    "TARGET_GEOGRAPHY" if _target_geography_supported(selected, self.target_geography)
                    else "NON_TARGET_OR_UNSPECIFIED_ACCEPTED_AS_CLASS4"
                ),
                "source_year": _clean(rec.get("source_year")), "search_tier": tier,
                "search_query": selected.get("query"), "evidence_quote": quote,
                "reason": _clean(rec.get("reason")), "raw_model_output": raw,
                "selected_result_id": sid, "product_identity_status": "PASS_SOURCE_SUPPORTED",
                "product_identity_reason": identity_reason,
            })
        return accepted

    def _build_provisional_consensus(self, kind: str, factors: list[dict[str, Any]]):
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
        consensus = robust_positive_consensus([float(x["value"]) for x in unique])
        if consensus is None:
            return None
        retained = [unique[i] for i in consensus.retained_indices]
        outliers = [unique[i] for i in consensus.outlier_indices]
        if not retained:
            return None
        values = [float(x["value"]) for x in retained]
        geos = sorted({_clean(x.get("source_geography")) for x in retained if _clean(x.get("source_geography"))})
        years = [int(x["source_year"]) for x in retained if str(x.get("source_year") or "").isdigit()]
        source_details = []
        retained_urls = {_clean(x.get("source_url")) for x in retained}
        outlier_urls = {_clean(x.get("source_url")) for x in outliers}
        for f in unique:
            source_details.append({
                "value": float(f["value"]),
                "unit": f.get("unit"),
                "source_title": f.get("source_title"),
                "source_url": f.get("source_url"),
                "source_class": f.get("source_class"),
                "source_geography": f.get("source_geography"),
                "geography_representativeness": f.get("geography_representativeness"),
                "source_year": f.get("source_year"),
                "search_tier": f.get("search_tier"),
                "evidence_quote": f.get("evidence_quote"),
                "retained_for_consensus": _clean(f.get("source_url")) in retained_urls,
                "outlier": _clean(f.get("source_url")) in outlier_urls,
            })
        one = retained[0]
        multi = len(retained) > 1
        return {
            "kind": kind,
            "value": float(consensus.central_value),
            "lower_value": min(values) if multi else None,
            "upper_value": max(values) if multi else None,
            "unit": one.get("unit"),
            "source_class": "EXTERNAL_VERIFIED_RELAXED",
            "verification": "TRACEABLE_WEB_RELAXED",
            "source_title": f"Median of {len(retained)} source-supported property values" if multi else one.get("source_title"),
            "source_url": None if multi else one.get("source_url"),
            "source_geography": ", ".join(geos) if geos else "Unspecified",
            "source_year": str(max(years)) if years else None,
            "search_tier": one.get("search_tier"),
            "search_query": None if multi else one.get("search_query"),
            "evidence_quote": None if multi else one.get("evidence_quote"),
            "reason": (
                f"Relaxed-phase External Verified property from {len(retained)} retained independent source(s) "
                f"using {consensus.method}; {len(outliers)} source-set outlier(s) excluded."
            ),
            "raw_model_output": None,
            "selected_result_id": one.get("selected_result_id") if not multi else None,
            "product_identity_status": "PASS_SOURCE_SUPPORTED",
            "product_identity_reason": "Every retained source passed deterministic material/property identity checks.",
            "peer_values_json": json.dumps(values, ensure_ascii=False),
            "peer_sources_json": json.dumps(source_details, ensure_ascii=False),
            "relaxed_source_count": len(unique),
            "relaxed_retained_count": len(retained),
            "relaxed_outlier_count": len(outliers),
            "relaxed_consensus_method": consensus.method,
            "relaxed_consensus_version": CONSENSUS_METHOD_VERSION,
        }

    def _parse_model_json(self, raw: str):
        try:
            return json.loads(raw.strip())
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None


    def _validated_model_property(self, obj, material: str, kind: str, expected_unit: str, required_keys: set[str]):
        if not isinstance(obj, dict) or set(obj.keys()) != required_keys or not bool(obj.get("found")):
            return None, "schema_or_found_invalid"
        central = _num(obj.get("central_value")); lower = _num(obj.get("lower_value")); upper = _num(obj.get("upper_value"))
        if central is None or lower is None or upper is None or not (0 < lower < central < upper):
            return None, "invalid_central_or_range"
        if _clean(obj.get("unit")).lower().replace("³", "3") != expected_unit.lower().replace("³", "3"):
            return None, "unit_mismatch"
        ok, reason = property_plausible(material, kind, central, lower, upper)
        if not ok:
            return None, reason
        return {
            "central": central, "lower": lower, "upper": upper,
            "interpretation": _clean(obj.get("analog_material") or obj.get("material_interpretation")),
            "basis": _clean(obj.get("estimation_basis")), "rationale": _clean(obj.get("rationale")),
        }, "ok"

    def _property_ensemble_from_output(self, raw: str, material: str, kind: str, expected_unit: str, *, analog: bool):
        obj = self._parse_model_json(raw)
        if not isinstance(obj, dict) or set(obj.keys()) != {"candidates"} or not isinstance(obj.get("candidates"), list):
            return None, {"accepted": False, "reason": "ensemble_schema_invalid", "raw_model_output": raw}
        if len(obj["candidates"]) != MODEL_ENSEMBLE_SIZE:
            return None, {"accepted": False, "reason": f"expected_{MODEL_ENSEMBLE_SIZE}_candidates", "raw_model_output": raw}
        required = {"found","central_value","lower_value","upper_value","unit","estimation_basis","rationale",("analog_material" if analog else "material_interpretation")}
        valid=[]; rejected=[]
        for i,candidate in enumerate(obj["candidates"]):
            parsed, reason = self._validated_model_property(candidate, material, kind, expected_unit, required)
            if parsed is None:
                rejected.append({"index":i,"reason":reason,"candidate":candidate})
            else:
                valid.append((i,candidate,parsed))
        values=[x[2]["central"] for x in valid]
        consensus=robust_model_ensemble_consensus(values)
        diag={"accepted":bool(consensus.accepted),"reason":consensus.reason,"method":consensus.method,
              "retained_ratio":consensus.retained_ratio,"valid_candidate_count":len(valid),
              "rejected_candidates":rejected,"all_valid_central_values":values,
              "valid_candidate_details":[x[2] for x in valid],
              "ensemble_method_version":MODEL_ENSEMBLE_METHOD_VERSION,"raw_model_output":raw}
        if not consensus.accepted:
            return None, diag
        retained_valid=[valid[i] for i in consensus.retained_indices]
        retained=[x[2] for x in retained_valid]
        central=float(consensus.central_value); lower=min(x["lower"] for x in retained); upper=max(x["upper"] for x in retained)
        if not (0 < lower < central < upper):
            return None, {**diag,"accepted":False,"reason":"consensus_interval_invalid"}
        diag["retained_candidate_indices"]=[x[0] for x in retained_valid]
        diag["retained_values"]=[x[2]["central"] for x in retained_valid]
        return {"central":central,"lower":lower,"upper":upper,
                "interpretations":[x["interpretation"] for x in retained],
                "bases":[x["basis"] for x in retained],"rationales":[x["rationale"] for x in retained],
                "all_values":values,"diagnostics":diag}, diag

    @staticmethod
    def _invalid_property_candidate(*, analog: bool, analog_material: str = "") -> dict[str, Any]:
        d={"found":False,"central_value":None,"lower_value":None,"upper_value":None,
           "unit":None,"estimation_basis":None,"rationale":None}
        if analog: d["analog_material"]=analog_material
        else: d["material_interpretation"]=None
        return d

    def _generate_one_property_candidate(self, *, system_prompt: str, payload: dict[str, Any],
                                         material: str, kind: str, expected_unit: str,
                                         analog: bool, analog_material: str = ""):
        required={"found","central_value","lower_value","upper_value","unit","estimation_basis","rationale",
                  ("analog_material" if analog else "material_interpretation")}
        attempts=[]; working=dict(payload)
        for attempt_no in range(1,3):
            raw=_call_matcher(self.matcher,system_prompt,working,max_new_tokens=360)
            obj=self._parse_model_json(raw)
            parsed=None; reason="parse_error"
            if isinstance(obj,dict):
                parsed,reason=self._validated_model_property(obj,material,kind,expected_unit,required)
                if analog and parsed is not None and _clean(obj.get("analog_material")) != _clean(analog_material):
                    parsed=None; reason="analog_name_not_preserved"
            attempts.append({"attempt":attempt_no,"accepted":parsed is not None,"reason":reason,"raw_model_output":raw})
            if parsed is not None:
                return obj,attempts
            working={**payload,"repair_instruction":"Return one complete JSON object only and correct the schema/unit/interval problem without using a prescribed numerical range.",
                     "previous_validation_error":reason,"previous_output":(raw or "")[-1200:]}
        return self._invalid_property_candidate(analog=analog,analog_material=analog_material),attempts


    @staticmethod
    def _property_anchor_warning(diag: dict[str, Any]) -> tuple[bool, str]:
        vals = []
        for v in (diag.get("all_valid_central_values") or []):
            try:
                vals.append(float(v))
            except Exception:
                pass
        if len(vals) < 4:
            return False, "insufficient_candidates_for_anchor_check"
        counts: dict[str, int] = {}
        for v in vals:
            key = format(v, ".12g")
            counts[key] = counts.get(key, 0) + 1
        most = max(counts.values(), default=0)
        distinct = len(counts)
        suspicious = most >= 3 and distinct > 1 and diag.get("method") in {"REPEATED_LOG_MEDIAN", "TIGHTEST_LOG_CLUSTER"}
        return suspicious, ("repeated_exact_model_value_with_disagreement" if suspicious else "no_anchor_warning")

    def _estimate_llm_unverified_property(self, material: str, kind: str):
        label, expected_unit = _property_label(kind, material)
        lenses=["composition and physical structure","manufacturing/product form","functional-product comparison",
                "unit-basis reconstruction","conservative engineering interpretation without a preset range"]
        candidates=[]; calls=[]
        self._progress(f"Property Class 4 · Method A · exact-material consensus · {material} · {kind}")
        for i,lens in enumerate(lenses,start=1):
            self._progress(f"Property Class 4 · Method A · candidate {i}/{len(lenses)} · {material} · {kind}")
            payload={"material":material,"target_context":self.target_geography,"requested_property":label,
                     "required_standardized_unit":expected_unit,"candidate_index":i,"estimation_lens":lens,
                     **property_guardrail_context(material,kind),
                     "status":"Evidence-backed property pathways failed; exploratory estimate only."}
            obj,attempts=self._generate_one_property_candidate(
                system_prompt=LLM_UNVERIFIED_PROPERTY_SINGLE_SYSTEM_PROMPT,payload=payload,material=material,
                kind=kind,expected_unit=expected_unit,analog=False)
            candidates.append(obj); calls.append({"candidate_index":i,"lens":lens,"attempts":attempts})
        raw=json.dumps({"candidates":candidates},ensure_ascii=False)
        ensemble,diag=self._property_ensemble_from_output(raw,material,kind,expected_unit,analog=False)
        diag["generation_calls"]=calls
        self._relaxed_property_pool.extend([{**x,"kind":kind,"unit":expected_unit} for x in diag.get("valid_candidate_details", [])])
        anchor_warning,anchor_reason=self._property_anchor_warning(diag)
        diag["anchor_warning"]=anchor_warning; diag["anchor_reason"]=anchor_reason
        self._last_fallback_attempts.append({"stage":"CLASS5A_EXACT_PROPERTY","diagnostics":diag})
        if ensemble is None: return None
        if anchor_warning:
            self._anchored_exact_property_pool = [{**x,"kind":kind,"unit":expected_unit} for x in diag.get("valid_candidate_details", [])]
            self._progress(f"Property Class 4 · repeated-value anchor detected; trying Method B before terminal fallback · {material} · {kind}")
            return None
        central,lower,upper=ensemble["central"],ensemble["lower"],ensemble["upper"]
        interp="; ".join(dict.fromkeys(x for x in ensemble["interpretations"] if x))
        return {
            "kind":kind,"value":central,"lower_value":lower,"upper_value":upper,"unit":expected_unit,
            "source_class":"UNVERIFIED_FALLBACK_ESTIMATE","verification":"UNVERIFIED_FALLBACK_ESTIMATE",
            "fallback_method":"EXACT_MATERIAL_LLM_CONSENSUS",
            "source_title":"Qwen five-independent-call property consensus estimate","source_url":None,
            "source_geography":"Unverified model assumption; target context "+self.target_geography,"source_year":None,
            "search_tier":"FINAL_LLM_PRODUCT_ENSEMBLE","search_query":None,"evidence_quote":None,
            "reason":f"Five independent material-agnostic property calls using {diag.get('method')}; retained {len(diag.get('retained_values',[]))} candidate(s).",
            "raw_model_output":raw,"selected_result_id":None,"product_identity_status":"UNVERIFIED_MODEL_ENSEMBLE",
            "product_identity_reason":interp,"peer_values_json":json.dumps(ensemble["all_values"],ensure_ascii=False),
            "peer_sources_json":None,"relaxed_source_count":None,"relaxed_retained_count":None,
            "relaxed_outlier_count":None,"relaxed_consensus_method":None,"relaxed_consensus_version":None,
            "guardrail_status":"PASS_ENSEMBLE_CONSENSUS","guardrail_reason":diag.get("method"),"guardrail_version":GUARDRAIL_VERSION,
            "fallback_attempts_json":json.dumps(self._last_fallback_attempts,ensure_ascii=False),
        }

    def _estimate_conservative_analog_property(self, material: str, kind: str):
        label, expected_unit = _property_label(kind, material)
        feedback=None; rounds=[]
        self._progress(f"Property Class 4 · Method B · dynamic analog fallback · {material} · {kind}")
        for round_no in range(1,3):
            self._progress(f"Property Class 4 · Method B · analog-plan round {round_no}/2 · {material} · {kind}")
            plan=infer_analog_plan(self.matcher,material=material,estimation_target=f"{label} in {expected_unit}",target_context=self.target_geography,feedback=feedback)
            if plan is None:
                rounds.append({"round":round_no,"accepted":False,"reason":"semantic_analog_plan_invalid"})
                feedback="Infer one broader but technically defensible family and five distinct analog products."
                continue
            candidates=[]; calls=[]
            for i,a in enumerate(plan["analogs"],start=1):
                analog_name=_clean(a.get("analog_material"))
                self._progress(f"Property Class 4 · Method B · analog {i}/{len(plan['analogs'])} · {material} → {analog_name} · {kind}")
                payload={"material":material,"semantic_family":plan["family_description"],"analog_material":analog_name,
                         "similarity_basis":a.get("similarity_basis"),"target_context":self.target_geography,
                         "requested_property":label,"required_standardized_unit":expected_unit,"candidate_index":i,
                         **property_guardrail_context(material,kind)}
                obj,attempts=self._generate_one_property_candidate(
                    system_prompt=CONSERVATIVE_ANALOG_PROPERTY_SINGLE_SYSTEM_PROMPT,payload=payload,material=material,
                    kind=kind,expected_unit=expected_unit,analog=True,analog_material=analog_name)
                candidates.append(obj); calls.append({"analog":analog_name,"attempts":attempts})
            raw=json.dumps({"candidates":candidates},ensure_ascii=False)
            ensemble,diag=self._property_ensemble_from_output(raw,material,kind,expected_unit,analog=True)
            diag["generation_calls"]=calls
            rounds.append({"round":round_no,"family":plan["family_description"],"analogs":plan["analogs"],"diagnostics":diag})
            self._relaxed_property_pool.extend([{**x,"kind":kind,"unit":expected_unit} for x in diag.get("valid_candidate_details", [])])
            if ensemble is None:
                feedback=f"Previous analog property estimates had no robust 3-of-5 consensus ({diag.get('reason')}). Move one semantic level broader while preserving composition, form and function."
                continue
            central,lower,upper=ensemble["central"],ensemble["lower"],ensemble["upper"]
            analogs="; ".join(dict.fromkeys(x for x in ensemble["interpretations"] if x))
            self._last_fallback_attempts.append({"stage":"CLASS5B_ANALOG_PROPERTY","rounds":rounds})
            return {
                "kind":kind,"value":central,"lower_value":lower,"upper_value":upper,"unit":expected_unit,
                "source_class":"UNVERIFIED_FALLBACK_ESTIMATE","verification":"UNVERIFIED_FALLBACK_ESTIMATE",
                "fallback_method":"DYNAMIC_ANALOG_CONSENSUS",
                "source_title":"Qwen dynamic semantic-family five-independent-analog property consensus","source_url":None,
                "source_geography":"Unverified analog assumption; target context "+self.target_geography,"source_year":None,
                "search_tier":"FINAL_DYNAMIC_ANALOG_ENSEMBLE","search_query":None,"evidence_quote":None,
                "reason":f"Dynamic semantic family: {plan['family_description']}; retained analogs: {analogs}.",
                "raw_model_output":raw,"selected_result_id":None,"product_identity_status":"DYNAMIC_SEMANTIC_ANALOG_CONSENSUS",
                "product_identity_reason":f"Family inferred at run time: {plan['family_description']}; analogs: {analogs}",
                "peer_values_json":json.dumps(ensemble["all_values"],ensure_ascii=False),"peer_sources_json":None,
                "relaxed_source_count":None,"relaxed_retained_count":None,"relaxed_outlier_count":None,
                "relaxed_consensus_method":None,"relaxed_consensus_version":None,
                "guardrail_status":"PASS_ENSEMBLE_CONSENSUS","guardrail_reason":diag.get("method"),"guardrail_version":GUARDRAIL_VERSION,
                "fallback_attempts_json":json.dumps(self._last_fallback_attempts,ensure_ascii=False),"semantic_analog_version":SEMANTIC_ANALOG_VERSION,
            }
        self._last_fallback_attempts.append({"stage":"CLASS5B_ANALOG_PROPERTY","rounds":rounds,"accepted":False})
        return None

    def _estimate_relaxed_valid_property_median(self, material: str, kind: str):
        """Final Class-4 property safety net from individually valid candidates."""
        label, expected_unit = _property_label(kind, material)
        anchored = [x for x in self._anchored_exact_property_pool if isinstance(x, dict) and x.get("kind") == kind and x.get("unit") == expected_unit]
        valid = anchored if anchored else [x for x in self._relaxed_property_pool if isinstance(x, dict) and x.get("kind") == kind and x.get("unit") == expected_unit]
        if not valid:
            self._last_fallback_attempts.append({"stage":"CLASS5C_RELAXED_PROPERTY","accepted":False,"reason":"no_individually_valid_candidates","kind":kind})
            return None
        values = sorted(float(x["central"]) for x in valid)
        n=len(values); central=values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2.0
        lower=min(float(x["lower"]) for x in valid); upper=max(float(x["upper"]) for x in valid)
        if not (0 < lower < central < upper):
            self._last_fallback_attempts.append({"stage":"CLASS5C_RELAXED_PROPERTY","accepted":False,"reason":"combined_interval_invalid","kind":kind,"values":values})
            return None
        interpretations="; ".join(dict.fromkeys(_clean(x.get("interpretation")) for x in valid if _clean(x.get("interpretation"))))
        terminal_method="TERMINAL_ROUNDED_EXACT_MATERIAL_MEDIAN" if anchored else "TERMINAL_VALID_CANDIDATE_MEDIAN"
        self._last_fallback_attempts.append({"stage":"CLASS5C_RELAXED_PROPERTY","accepted":True,"kind":kind,"valid_candidate_count":len(valid),"values":values,"terminal_method":terminal_method})
        return {
            "kind":kind,"value":central,"lower_value":lower,"upper_value":upper,"unit":expected_unit,
            "source_class":"UNVERIFIED_FALLBACK_ESTIMATE","verification":"UNVERIFIED_FALLBACK_ESTIMATE",
            "fallback_method":terminal_method,
            "source_title":("Qwen terminal median of anchored exact-material property candidates" if anchored else "Qwen terminal median of individually valid property candidates"),"source_url":None,
            "source_geography":"Unverified model/analog assumption; target context "+self.target_geography,"source_year":None,
            "search_tier":"FINAL_TERMINAL_VALID_CANDIDATE_MEDIAN","search_query":None,"evidence_quote":None,
            "reason":(f"Strict/independence-aware exact-material acceptance and analog property consensus failed. Used the median of {len(valid)} anchored exact-material candidates as the terminal numerical fallback; repeated/rounded values are permitted only at this final stage and the full exact-candidate envelope is retained as uncertainty." if anchored else f"Strict/independence-aware exact-material acceptance and analog property consensus failed. Used the median of {len(valid)} individually valid candidates as the terminal numerical fallback; the full envelope is retained as uncertainty."),
            "raw_model_output":None,"selected_result_id":None,"product_identity_status":"UNVERIFIED_RELAXED_MODEL_ANALOG_POOL",
            "product_identity_reason":interpretations or "Individually valid fallback property candidates",
            "peer_values_json":json.dumps(values,ensure_ascii=False),"peer_sources_json":None,
            "relaxed_source_count":None,"relaxed_retained_count":None,"relaxed_outlier_count":None,
            "relaxed_consensus_method":None,"relaxed_consensus_version":None,
            "guardrail_status":"PASS_INDIVIDUAL_VALIDATION_RELAXED_CONSENSUS","guardrail_reason":terminal_method,
            "guardrail_version":GUARDRAIL_VERSION,"fallback_attempts_json":json.dumps(self._last_fallback_attempts,ensure_ascii=False),
        }


    @staticmethod
    def _median_property_value(values: list[float]) -> float:
        vals = sorted(float(v) for v in values)
        n = len(vals)
        if not n:
            raise ValueError("median requires at least one value")
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    def _estimate_terminal_llm_only_property(self, material: str, kind: str):
        """Last Class-4 property fallback used to preserve exploratory coverage.

        This stage contains no material-specific property value/range. It runs
        only after evidence, exact-material consensus, analog consensus, and the
        existing valid-candidate median have failed. Fresh Qwen estimates are
        generated in the standardized property unit. Their median is retained
        even without strong consensus because the resulting row remains Class 4
        and is excluded from verified subtotal.
        """
        label, expected_unit = _property_label(kind, material)
        lenses = [
            "fresh physical-structure reconstruction",
            "independent product-form/manufacturing reconstruction",
            "independent functional engineering estimate",
        ]
        valid=[]; calls=[]
        self._progress(f"Property Class 4 · Method D · terminal LLM-only numerical fallback · {material} · {kind}")
        for i,lens in enumerate(lenses,start=1):
            payload={
                "material":material,"target_context":self.target_geography,
                "requested_property":label,"required_standardized_unit":expected_unit,
                "candidate_index":i,"estimation_lens":lens,
                **property_guardrail_context(material,kind),
                "status":"All stronger property checks failed. A terminal unverified numerical value is required only for the separated exploratory calculation.",
            }
            obj,attempts=self._generate_one_property_candidate(
                system_prompt=LLM_UNVERIFIED_PROPERTY_SINGLE_SYSTEM_PROMPT,
                payload=payload,material=material,kind=kind,expected_unit=expected_unit,analog=False)
            required={"found","central_value","lower_value","upper_value","unit","estimation_basis","rationale","material_interpretation"}
            parsed,reason=self._validated_model_property(obj,material,kind,expected_unit,required) if isinstance(obj,dict) else (None,"parse_error")
            calls.append({"candidate_index":i,"lens":lens,"attempts":attempts,"accepted":parsed is not None,"reason":reason})
            if parsed is not None:
                valid.append(parsed)
        if valid:
            values=[float(x["central"]) for x in valid]
            central=self._median_property_value(values)
            lower=min(float(x["lower"]) for x in valid)
            upper=max(float(x["upper"]) for x in valid)
            if not (0 < lower < central < upper):
                lower=None; upper=None
            method="TERMINAL_LLM_ONLY_PROPERTY_MEDIAN_NO_CONSENSUS" if len(valid)>1 else "TERMINAL_LLM_ONLY_PROPERTY_SINGLE_VALID_ESTIMATE"
            self._last_fallback_attempts.append({
                "stage":"CLASS5D_TERMINAL_PROPERTY","accepted":True,"method":method,
                "valid_candidate_count":len(valid),"values":values,"generation_calls":calls,
                "quality":"VERY_LOW_CONFIDENCE_MODEL_ONLY",
            })
            interp="; ".join(dict.fromkeys(_clean(x.get("interpretation")) for x in valid if _clean(x.get("interpretation"))))
            return {
                "kind":kind,"value":central,"lower_value":lower,"upper_value":upper,"unit":expected_unit,
                "source_class":"UNVERIFIED_FALLBACK_ESTIMATE","verification":"UNVERIFIED_FALLBACK_ESTIMATE",
                "fallback_method":method,"source_title":"Qwen terminal model-only exploratory property estimate","source_url":None,
                "source_geography":"Unverified model assumption; target context "+self.target_geography,"source_year":None,
                "search_tier":"FINAL_TERMINAL_LLM_ONLY_PROPERTY","search_query":None,"evidence_quote":None,
                "reason":f"Stronger property checks failed. Used the median of {len(valid)} fresh model-only estimate(s) in {expected_unit} solely to preserve exploratory numerical coverage. No source support or material-specific coded range is claimed.",
                "raw_model_output":None,"selected_result_id":None,"product_identity_status":"TERMINAL_UNVERIFIED_MODEL_ONLY_PROPERTY",
                "product_identity_reason":interp or "Fresh terminal property estimate",
                "peer_values_json":json.dumps(values,ensure_ascii=False),"peer_sources_json":None,
                "relaxed_source_count":None,"relaxed_retained_count":None,"relaxed_outlier_count":None,
                "relaxed_consensus_method":None,"relaxed_consensus_version":None,
                "guardrail_status":"TERMINAL_VALUE_PRESERVED_AFTER_FAILED_STRONG_CONSENSUS",
                "guardrail_reason":"VERY_LOW_CONFIDENCE_MODEL_ONLY","guardrail_version":GUARDRAIL_VERSION,
                "fallback_attempts_json":json.dumps(self._last_fallback_attempts,ensure_ascii=False),
                "terminal_quality_flag":"VERY_LOW_CONFIDENCE_MODEL_ONLY",
            }

        working={
            "material":material,"target_context":self.target_geography,
            "requested_property":label,"required_standardized_unit":expected_unit,
            "status":"Last numerical property fallback; return one unverified positive value.",
        }
        attempts=[]
        for attempt_no in range(1,3):
            raw=_call_matcher(self.matcher,TERMINAL_MINIMAL_PROPERTY_SYSTEM_PROMPT,working,max_new_tokens=220)
            obj=self._parse_model_json(raw)
            reason="minimal_schema_invalid"; value=None
            if isinstance(obj,dict) and set(obj.keys())=={"found","value","unit","material_interpretation","rationale"} and bool(obj.get("found")):
                value=_num(obj.get("value"))
                if value is None or value<=0:
                    reason="minimal_value_invalid"; value=None
                elif _clean(obj.get("unit")).lower().replace("³","3") != expected_unit.lower().replace("³","3"):
                    reason="minimal_unit_mismatch"; value=None
                else:
                    ok,why=property_plausible(material,kind,value)
                    if not ok:
                        reason=why; value=None
                    else:
                        reason="ok"
            attempts.append({"attempt":attempt_no,"accepted":value is not None,"reason":reason,"raw_model_output":raw})
            if value is not None:
                self._last_fallback_attempts.append({
                    "stage":"CLASS5E_TERMINAL_MINIMAL_PROPERTY","accepted":True,"attempts":attempts,
                    "quality":"LOWEST_CONFIDENCE_MODEL_ONLY_NO_RANGE",
                })
                return {
                    "kind":kind,"value":float(value),"lower_value":None,"upper_value":None,"unit":expected_unit,
                    "source_class":"UNVERIFIED_FALLBACK_ESTIMATE","verification":"UNVERIFIED_FALLBACK_ESTIMATE",
                    "fallback_method":"TERMINAL_MINIMAL_SINGLE_LLM_PROPERTY_VALUE",
                    "source_title":"Qwen terminal minimal-schema model-only property estimate","source_url":None,
                    "source_geography":"Unverified model assumption; target context "+self.target_geography,"source_year":None,
                    "search_tier":"FINAL_TERMINAL_MINIMAL_PROPERTY_VALUE","search_query":None,"evidence_quote":None,
                    "reason":"All stronger property paths failed. Retained one final positive model-only value in the required standardized unit to preserve exploratory coverage; no source-supported range is claimed.",
                    "raw_model_output":raw,"selected_result_id":None,"product_identity_status":"TERMINAL_UNVERIFIED_MODEL_ONLY_PROPERTY",
                    "product_identity_reason":_clean(obj.get("material_interpretation")),
                    "peer_values_json":json.dumps([float(value)]),"peer_sources_json":None,
                    "relaxed_source_count":None,"relaxed_retained_count":None,"relaxed_outlier_count":None,
                    "relaxed_consensus_method":None,"relaxed_consensus_version":None,
                    "guardrail_status":"TERMINAL_MINIMAL_VALUE_ONLY","guardrail_reason":"LOWEST_CONFIDENCE_MODEL_ONLY_NO_RANGE",
                    "guardrail_version":GUARDRAIL_VERSION,"fallback_attempts_json":json.dumps(self._last_fallback_attempts,ensure_ascii=False),
                    "terminal_quality_flag":"LOWEST_CONFIDENCE_MODEL_ONLY_NO_RANGE",
                }
            working={**working,"repair_instruction":"Return exactly the requested minimal JSON schema with one finite positive value in required_standardized_unit.","previous_validation_error":reason,"previous_output":(raw or "")[-800:]}
        self._last_fallback_attempts.append({"stage":"CLASS5E_TERMINAL_MINIMAL_PROPERTY","accepted":False,"attempts":attempts})
        return None

    def _resolve_one_property(self, item_id: str, material: str, kind: str):
        cache_key = (material.strip().lower(), kind)
        if cache_key in self._resolution_cache:
            return dict(self._resolution_cache[cache_key]), []

        cached = self.evidence_cache.get(
            category="physical_property", material=material,
            target_geography=self.target_geography, property_kind=kind,
        )
        if cached and cached.get("verification") in {"TRACEABLE_WEB", "TRACEABLE_WEB_RELAXED"}:
            cached = dict(cached)
            cached["reason"] = (_clean(cached.get("reason")) + " Runtime evidence-cache reuse.").strip()
            cached["evidence_cache_version"] = CACHE_VERSION
            self._progress(f"Property cache hit · {material} · {kind} · {cached.get('verification')}")
            self._resolution_cache[cache_key] = dict(cached)
            cache_evidence = [{
                "item_id": item_id, "material": material, "property_kind": kind,
                "search_tier": "RUNTIME_CACHE", "query": None, "result_id": "CACHEP1",
                "title": cached.get("source_title"), "url": cached.get("source_url"),
                "snippet": cached.get("evidence_quote"), "excerpt": cached.get("evidence_quote"),
                "product_identity_ok": True, "product_identity_reason": "Previously accepted retrieved evidence",
                "selected": True, "relaxed_candidate": cached.get("verification") == "TRACEABLE_WEB_RELAXED",
                "relaxed_retained": cached.get("verification") == "TRACEABLE_WEB_RELAXED",
                "relaxed_outlier": False, "relaxed_value": cached.get("value"),
                "relaxed_reason": "Runtime evidence-cache reuse",
            }]
            return cached, cache_evidence

        all_evidence: list[dict[str, Any]] = []
        provisional_factors: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        promising_reasons: list[str] = []
        promising_candidates: list[dict[str, Any]] = []
        plans = list(_query_sets(material, kind, self.target_geography))

        # CLASS 3 PROPERTY — independent target-geography clock.
        class3_started = time.monotonic()
        class3_inspected = 0
        class3_queries_used = 0
        nepal_plan = next((p for p in plans if p[0] == "NEPAL"), None)
        if nepal_plan is not None:
            tier, queries = nepal_plan
            self._progress(
                f"External property evidence · Phase A strict {self.target_geography} · {material} · {kind} · "
                f"independent budget {self.max_external_seconds_per_property:.0f} s · "
                f"max {self.class3_source_budget} sources"
            )
            candidates, evidence_rows, used = self._collect_evidence(
                queries, tier, item_id, material, kind,
                max_candidates=self.class3_source_budget,
                max_queries=min(3, self.max_search_queries_per_property),
                seen_urls=seen_urls,
            )
            class3_queries_used += used
            class3_inspected += len(candidates)
            all_evidence.extend(evidence_rows)
            for c in candidates:
                ok_promising, why_promising = _promising_property_candidate(material, kind, c)
                if ok_promising:
                    promising_reasons.append(why_promising)
                    if c.get("url") and not any(x.get("url") == c.get("url") for x in promising_candidates):
                        promising_candidates.append(dict(c))
            selected, _ = self._choose_traceable(material, kind, tier, candidates)
            if selected is not None:
                sid = selected.get("selected_result_id")
                for row in all_evidence:
                    if row["search_tier"] == tier and row["result_id"] == sid and row["url"] == selected["source_url"]:
                        row["selected"] = True
                selected = dict(selected)
                selected["external_verified_strict_elapsed_seconds"] = round(time.monotonic() - class3_started, 3)
                selected["evidence_cache_version"] = CACHE_VERSION
                self.evidence_cache.put(
                    category="physical_property", material=material, target_geography=self.target_geography,
                    property_kind=kind, resolved=selected,
                )
                self._resolution_cache[cache_key] = dict(selected)
                self._progress(f"External property evidence Phase A accepted · {material} · {kind}")
                return selected, all_evidence
            if self.allow_source_supported_provisional and candidates:
                provisional_factors.extend(self._extract_relaxed_candidates(material, kind, tier, candidates))

        self._progress(
            f"External property evidence Phase A finished · {material} · {kind} · {class3_inspected} source(s), "
            f"{class3_queries_used} query/queries · starting Phase B relaxed search with a fresh clock"
        )

        if not self.allow_source_supported_provisional:
            return None, all_evidence

        # CLASS 4 PROPERTY — independent geography-relaxed clock. This is used
        # only when deterministic unit conversion actually requires the property.
        class4_started = time.monotonic()
        class4_inspected = 0
        external_verified_relaxed_queries_used = 0
        class4_source_budget = self.adaptive_total_source_budget
        class4_query_budget = self.adaptive_max_search_queries_per_property
        class4_time_budget = self.adaptive_external_seconds_per_property
        self._progress(
            f"External property evidence · Phase B relaxed source-supported search · {material} · {kind} · "
            f"budget {class4_time_budget:.0f} s · up to {class4_query_budget} queries / "
            f"{class4_source_budget} unique sources · geography may differ"
        )

        equivalent_plan = infer_technical_equivalents(
            self.matcher, material=material, target_context=self.target_geography
        )
        label, unit = _property_label(kind, material)
        equivalent_plans = []
        if equivalent_plan:
            for rec in equivalent_plan.get("search_terms", []):
                subject = _clean(rec.get("term"))
                if not subject:
                    continue
                equivalent_plans.extend([
                    ("GLOBAL", [
                        f'"{subject}" {label} {unit} technical data',
                        f'"{subject}" {label} {unit} USA Canada',
                        f'"{subject}" {label} {unit} filetype:pdf',
                    ]),
                    ("SOUTH_ASIA", [
                        f'"{subject}" {label} {unit} India technical data',
                        f'"{subject}" {label} {unit} South Asia',
                    ]),
                ])

        standard_non_nepal = [p for p in plans if p[0] != "NEPAL"]
        search_plans = standard_non_nepal + equivalent_plans
        for tier, queries in search_plans:
            if time.monotonic() - class4_started >= class4_time_budget:
                self._progress(f"External property evidence Phase B time budget reached · {material} · {kind}; moving toward Class 4 fallback")
                break
            if class4_inspected >= class4_source_budget or external_verified_relaxed_queries_used >= class4_query_budget:
                break
            remaining_sources = class4_source_budget - class4_inspected
            remaining_queries = class4_query_budget - external_verified_relaxed_queries_used
            candidates, evidence_rows, used = self._collect_evidence(
                queries, tier, item_id, material, kind,
                max_candidates=min(2, remaining_sources),
                max_queries=min(2, remaining_queries),
                seen_urls=seen_urls,
            )
            external_verified_relaxed_queries_used += used
            class4_inspected += len(candidates)
            all_evidence.extend(evidence_rows)
            for c in candidates:
                ok_promising, why_promising = _promising_property_candidate(material, kind, c)
                if ok_promising:
                    promising_reasons.append(why_promising)
                    if c.get("url") and not any(x.get("url") == c.get("url") for x in promising_candidates):
                        promising_candidates.append(dict(c))
            if candidates:
                extracted = self._extract_relaxed_candidates(material, kind, tier, candidates)
                provisional_factors.extend(extracted)
                by_url = {_clean(x.get("source_url")): x for x in extracted}
                for row in all_evidence:
                    f = by_url.get(_clean(row.get("url")))
                    if f is not None:
                        row["relaxed_candidate"] = True
                        row["relaxed_value"] = f.get("value")
                        row["relaxed_reason"] = f.get("reason")
                provisional = self._build_provisional_consensus(kind, provisional_factors)
                if provisional is not None:
                    try:
                        details = json.loads(provisional.get("peer_sources_json") or "[]")
                    except Exception:
                        details = []
                    retained_urls = {_clean(x.get("source_url")) for x in details if x.get("retained_for_consensus")}
                    outlier_urls = {_clean(x.get("source_url")) for x in details if x.get("outlier")}
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
                        category="physical_property", material=material, target_geography=self.target_geography,
                        property_kind=kind, resolved=provisional,
                    )
                    self._resolution_cache[cache_key] = dict(provisional)
                    self._progress(f"External property evidence Phase B accepted · {material} · {kind}")
                    return provisional, all_evidence

        # Deep re-extraction of a small number of promising documents is allowed
        # within the same independent relaxed-phase Class-3 clock.
        if promising_candidates and time.monotonic() - class4_started < class4_time_budget:
            old_pages = self.pdf_max_pages
            try:
                self.pdf_max_pages = self.adaptive_pdf_max_pages
                deep_candidates = []
                for c0 in promising_candidates[:2]:
                    if time.monotonic() - class4_started >= class4_time_budget:
                        break
                    c = dict(c0)
                    self._progress(f"External property evidence · Phase B deeper extraction of promising source · {material} · {kind}")
                    c["full_text"] = self._extract(c.get("url"))
                    c["excerpt"] = c["full_text"][: max(self.excerpt_chars, 2200)]
                    deep_candidates.append(c)
                for c in deep_candidates:
                    provisional_factors.extend(
                        self._extract_relaxed_candidates(material, kind, c.get("tier") or "GLOBAL", [c])
                    )
                provisional = self._build_provisional_consensus(kind, provisional_factors) if provisional_factors else None
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
                        category="physical_property", material=material, target_geography=self.target_geography,
                        property_kind=kind, resolved=provisional,
                    )
                    self._resolution_cache[cache_key] = dict(provisional)
                    self._progress(f"External property evidence Phase B accepted after deeper extraction · {material} · {kind}")
                    return provisional, all_evidence
            finally:
                self.pdf_max_pages = old_pages

        self._progress(
            f"External property evidence Phase B finished without accepted value · {material} · {kind} · "
            f"{class4_inspected} source(s), {external_verified_relaxed_queries_used} query/queries · moving to Class 4 fallback"
        )
        return None, all_evidence

    def resolve_record(self, rec: dict[str, Any]):
        item_id = _clean(rec.get("item_id") or rec.get("ID")) or "ITEM"
        original_material = _clean(rec.get("original_material") or rec.get("Material"))
        normalized_material = _clean(rec.get("normalized_material"))
        material, material_basis = choose_resolution_material(original_material, normalized_material)
        source_unit = rec.get("unit") or rec.get("Unit")
        approved = _as_bool(rec.get("production_approved"))
        external_status = _clean(rec.get("external_ef_resolution_status"))
        external_ready = external_status in {
            "RESOLVED_EXTERNAL_VERIFIED",
            "RESOLVED_TRACEABLE_WEB_RELAXED",
            "RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE",
            "RESOLVED_LLM_UNVERIFIED_ESTIMATE",  # legacy payload compatibility
            "RESOLVED_CONSERVATIVE_ANALOG_ESTIMATE",  # legacy payload compatibility
        }
        # Approved ELCD rows use the Qwen-selected production process unit; rows
        # routed to an external factor use the external factor's reference unit.
        if approved:
            ref_unit = rec.get("production_selected_process_ref_unit")
        elif external_ready:
            ref_unit = rec.get("external_ef_reference_unit")
        else:
            ref_unit = None

        result = {
            "property_resolution_status": "NOT_NEEDED",
            "property_required": None,
            "property_lookup_needed": False,
            "property_lookup_reason": None,
            "property_fallback_method": None,
            "property_terminal_quality_flag": None,
            "resolved_density_kg_m3": None,
            "resolved_thickness_mm": None,
            "resolved_mass_per_item_kg": None,
            "property_verification": "NOT_NEEDED",
            "property_source_class": None,
            "property_source_title": None,
            "property_source_url": None,
            "property_source_geography": None,
            "property_source_year": None,
            "property_search_tier": None,
            "property_search_query": None,
            "property_evidence_quote": None,
            "property_lower_value": None,
            "property_upper_value": None,
            "property_peer_values_json": None,
            "property_peer_sources_json": None,
            "property_relaxed_source_count": None,
            "property_relaxed_retained_count": None,
            "property_relaxed_outlier_count": None,
            "property_relaxed_consensus_method": None,
            "property_relaxed_consensus_version": None,
            "property_details_json": None,
            "property_material_query": material,
            "property_material_selection_basis": material_basis,
            "property_material_family": classify_material(material),
            "property_material_taxonomy_version": MATERIAL_TAXONOMY_VERSION,
            "property_resolver_version": PROPERTY_RESOLVER_VERSION,
            "property_resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "property_guardrail_status": None,
            "property_guardrail_reason": None,
            "property_guardrail_version": GUARDRAIL_VERSION,
            "property_evidence_cache_version": CACHE_VERSION,
            "property_fallback_attempts_json": None,
        }
        evidence: list[dict[str, Any]] = []

        if not approved and not external_ready:
            result["property_resolution_status"] = "SKIPPED_NO_APPROVED_FACTOR"
            result["property_verification"] = "NOT_APPLICABLE"
            return result, evidence
        if not _clean(ref_unit):
            result["property_resolution_status"] = "DEFERRED_REFERENCE_UNIT_UNKNOWN"
            result["property_verification"] = "INPUT_OR_MODEL_FAILURE"
            return result, evidence

        explicit_factor = _num(rec.get("conversion_factor_to_ref_unit") or rec.get("Conversion_factor_to_ref_unit"))
        if explicit_factor is not None and explicit_factor > 0:
            result["property_lookup_reason"] = "Explicit Conversion_factor_to_ref_unit supplied; no physical-property lookup needed."
            return result, evidence

        requirements = conversion_requirements(source_unit, ref_unit)
        if not requirements:
            result["property_lookup_reason"] = f"BOM unit {norm_unit(source_unit) or source_unit} converts directly to EF reference unit {norm_unit(ref_unit) or ref_unit}; no physical property needed."
            return result, evidence

        result["property_required"] = ";".join(requirements)
        result["property_lookup_needed"] = True
        result["property_lookup_reason"] = f"Physical conversion required from BOM unit {norm_unit(source_unit) or source_unit} to EF reference unit {norm_unit(ref_unit) or ref_unit}: {', '.join(requirements)}."
        self._last_fallback_attempts = []
        self._relaxed_property_pool = []
        self._anchored_exact_property_pool = []
        details: list[dict[str, Any]] = []
        any_provisional = False
        any_web = False
        any_llm = False
        any_analog = False
        unresolved = False

        for kind in requirements:
            # User/project value has highest priority.
            source_field = kind
            user_value = _num(rec.get(source_field))
            if user_value is not None and user_value > 0:
                d = {
                    "kind": kind,
                    "value": user_value,
                    "verification": "PROJECT_INPUT",
                    "source_class": "USER_PROJECT_INPUT",
                    "source_title": None,
                    "source_url": None,
                    "source_geography": self.target_geography,
                    "source_year": None,
                    "search_tier": None,
                    "search_query": None,
                    "evidence_quote": None,
                    "lower_value": None,
                    "upper_value": None,
                    "reason": "Physical property supplied directly in the BOM.",
                }
                details.append(d)
                continue

            # Thickness explicitly present in material name is an extraction, not an LLM estimate.
            if kind == "thickness_mm":
                thickness = extract_thickness_mm(
                    _clean(rec.get("original_material") or material)
                )
                if thickness is not None:
                    d = {
                        "kind": kind,
                        "value": thickness,
                        "verification": "BOM_EXTRACTED",
                        "source_class": "BOM_DESCRIPTION",
                        "source_title": None,
                        "source_url": None,
                        "source_geography": self.target_geography,
                        "source_year": None,
                        "search_tier": None,
                        "search_query": None,
                        "evidence_quote": _clean(rec.get("original_material") or material),
                        "lower_value": None,
                        "upper_value": None,
                        "reason": "Thickness explicitly encoded in BOM material description.",
                    }
                    details.append(d)
                    continue

            d, ev = self._resolve_one_property(item_id, material, kind)
            evidence.extend(ev)
            if d is None and self.allow_llm_unverified_estimate:
                d = self._estimate_llm_unverified_property(material, kind)
            if d is None and self.allow_conservative_analog_estimate:
                d = self._estimate_conservative_analog_property(material, kind)
            if d is None and (self.allow_llm_unverified_estimate or self.allow_conservative_analog_estimate):
                self._progress(f"Property Class 4 · Method C · terminal valid-candidate median · {material} · {kind}")
                d = self._estimate_relaxed_valid_property_median(material, kind)
            if d is None and (self.allow_llm_unverified_estimate or self.allow_conservative_analog_estimate):
                self._progress(f"Property Class 4 · Method D/E · value-preserving terminal fallback · {material} · {kind}")
                d = self._estimate_terminal_llm_only_property(material, kind)
            if d is None:
                unresolved = True
                details.append({
                    "kind": kind,
                    "value": None,
                    "verification": "INPUT_OR_MODEL_FAILURE",
                    "source_class": None,
                    "reason": "No traceable strict/relaxed external property or Class-4 unverified fallback property estimate passed deterministic checks.",
                    "fallback_attempts_json": json.dumps(self._last_fallback_attempts, ensure_ascii=False),
                })
                continue
            details.append(d)
            if d.get("verification") == "TRACEABLE_WEB":
                any_web = True
            if d.get("verification") == "TRACEABLE_WEB_RELAXED":
                any_provisional = True
            if d.get("verification") == "UNVERIFIED_FALLBACK_ESTIMATE":
                any_llm = True
                if d.get("fallback_method") == "DYNAMIC_ANALOG_CONSENSUS":
                    any_analog = True

        by_kind = {d["kind"]: d for d in details if d.get("value") is not None}
        if "density_kg_m3" in by_kind:
            result["resolved_density_kg_m3"] = by_kind["density_kg_m3"]["value"]
        if "thickness_mm" in by_kind:
            result["resolved_thickness_mm"] = by_kind["thickness_mm"]["value"]
        if "mass_per_item_kg" in by_kind:
            result["resolved_mass_per_item_kg"] = by_kind["mass_per_item_kg"]["value"]

        if unresolved:
            result["property_resolution_status"] = "INPUT_OR_MODEL_FAILURE"
            result["property_verification"] = "INPUT_OR_MODEL_FAILURE"
            result["property_fallback_attempts_json"] = json.dumps(self._last_fallback_attempts, ensure_ascii=False)
        elif any_llm or any_analog:
            result["property_resolution_status"] = "RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE"
            result["property_verification"] = "UNVERIFIED_FALLBACK_ESTIMATE"
        elif any_provisional:
            result["property_resolution_status"] = "RESOLVED_TRACEABLE_WEB_RELAXED"
            result["property_verification"] = "TRACEABLE_WEB_RELAXED"
        elif any_web:
            result["property_resolution_status"] = "RESOLVED_TRACEABLE_WEB"
            result["property_verification"] = "TRACEABLE_WEB"
        else:
            result["property_resolution_status"] = "RESOLVED_FROM_BOM_OR_PROJECT_INPUT"
            result["property_verification"] = "PROJECT_OR_BOM"

        # Surface the first external property in flat audit columns; all details remain in JSON.
        primary = next(
            (d for d in details if d.get("verification") in {"TRACEABLE_WEB", "TRACEABLE_WEB_RELAXED", "UNVERIFIED_FALLBACK_ESTIMATE"}),
            details[0] if details else None,
        )
        if primary:
            result["property_source_class"] = primary.get("source_class")
            result["property_source_title"] = primary.get("source_title")
            result["property_source_url"] = primary.get("source_url")
            result["property_source_geography"] = primary.get("source_geography")
            result["property_source_year"] = primary.get("source_year")
            result["property_search_tier"] = primary.get("search_tier")
            result["property_search_query"] = primary.get("search_query")
            result["property_evidence_quote"] = primary.get("evidence_quote")
            result["property_lower_value"] = primary.get("lower_value")
            result["property_upper_value"] = primary.get("upper_value")
            result["property_peer_values_json"] = primary.get("peer_values_json")
            result["property_peer_sources_json"] = primary.get("peer_sources_json")
            result["property_relaxed_source_count"] = primary.get("relaxed_source_count")
            result["property_relaxed_retained_count"] = primary.get("relaxed_retained_count")
            result["property_relaxed_outlier_count"] = primary.get("relaxed_outlier_count")
            result["property_relaxed_consensus_method"] = primary.get("relaxed_consensus_method")
            result["property_relaxed_consensus_version"] = primary.get("relaxed_consensus_version")
            result["property_fallback_method"] = primary.get("fallback_method")
            result["property_terminal_quality_flag"] = primary.get("terminal_quality_flag")
            result["property_guardrail_status"] = primary.get("guardrail_status")
            result["property_guardrail_reason"] = primary.get("guardrail_reason")
            result["property_fallback_attempts_json"] = primary.get("fallback_attempts_json")
        result["property_details_json"] = json.dumps(details, ensure_ascii=False)
        return result, evidence

    def resolve_batch(self, df: pd.DataFrame):
        rows = []
        evidence_rows: list[dict[str, Any]] = []
        records = df.to_dict(orient="records")
        total = len(records)
        for i, rec in enumerate(records, start=1):
            material = _clean(rec.get("normalized_material") or rec.get("original_material") or rec.get("Material"))
            self._progress(f"Property resolver · row {i}/{total} · {material}")
            resolved, evidence = self.resolve_record(rec)
            merged = dict(rec)
            merged.update(resolved)
            rows.append(merged)
            evidence_rows.extend(evidence)
            status = resolved.get("property_verification") or resolved.get("property_resolution_status")
            self._progress(f"Property resolver · row {i}/{total} complete · {material} · {status}")
        return pd.DataFrame(rows), pd.DataFrame(evidence_rows)
