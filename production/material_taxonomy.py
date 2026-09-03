"""Deterministic construction-material taxonomy used by the production safety layers.

The taxonomy is intentionally small and engineering-oriented.  It is not an LCA
factor database.  It provides reusable family identity rules so that obvious
semantic false positives (for example, earthen soil -> PVC soil pipe) are
rejected before they can enter the calculation.
"""
from __future__ import annotations

import re
from typing import Any

MATERIAL_TAXONOMY_VERSION = "1.6"


def _norm(value: Any) -> str:
    s = "" if value is None else str(value).lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _has_any(text: str, terms) -> bool:
    """Loose substring helper for intentionally broad/stem-based rules."""
    return any(t in text for t in terms)


def _has_term(text: str, term: str) -> bool:
    """Match a normalized whole word or multi-word phrase, never a substring.

    This prevents false positives such as ``cement`` matching the tail of
    ``reinforcement`` and also handles hyphenated source labels because both
    inputs are normalized before comparison.
    """
    normalized_text = f" {_norm(text)} "
    normalized_term = _norm(term)
    if not normalized_term:
        return False
    return f" {normalized_term} " in normalized_text


def _has_any_term(text: str, terms) -> bool:
    return any(_has_term(text, term) for term in terms)


def classify_material(material: Any) -> str:
    """Classify a BOM description into a reusable construction product family."""
    m = _norm(material)

    # Specific manufactured products before their constituents.
    if "stonecrete" in m:
        return "STONECRETE_BLOCK"
    if _has_any(m, ("compressed earth block", "compressed soil block", "soil block", "earth block", "ceb")):
        return "SOIL_BLOCK"
    if "plywood" in m:
        return "PLYWOOD"
    if _has_term(m, "bamboo"):
        return "BAMBOO"
    if _has_any(m, ("plain cement concrete", "pcc m10", "pcc 1 3 6", "blinding concrete", "blinding layer")):
        return "PLAIN_CONCRETE"
    # Generic concrete/RMC descriptions must be classified before the word
    # ``cement`` is considered.  This keeps company names such as "... Cement"
    # from masking the fact that the declared product itself is concrete.
    if _has_any_term(m, ("ready mix concrete", "ready mixed concrete", "precast concrete", "pre cast concrete", "concrete")):
        return "PLAIN_CONCRETE"
    if _has_any(m, ("plaster", "render")):
        return "PLASTER"
    if _has_any(m, (
        "cgi", "corrugated galvanized", "corrugated galvanised", "corrugated iron",
        "galvanized sheet", "galvanised sheet", "galvanized steel sheet", "galvanised steel sheet",
        "galvanized steel coil", "galvanised steel coil", "galvanized flat steel",
        "galvanised flat steel", "zinc coated steel", "zinc-coated steel",
        "hot dip coated steel", "hot-dip coated steel",
    )):
        return "GALVANIZED_FLAT_STEEL"
    if _has_any(m, ("binding wire", "tie wire", "annealed wire", "steel wire")):
        return "STEEL_WIRE"
    if _has_any(m, ("nail", "fastener")):
        return "STEEL_FASTENER"
    if _has_any(m, (
        "rebar", "reinforcing steel", "reinforcement steel", "reinforcing bar",
        "reinforcement bar", "tmt bar", "tmt steel", "tmt fe",
    )):
        return "REBAR"
    if _has_term(m, "cement"):
        return "CEMENT"
    if _has_term(m, "sand"):
        return "SAND"
    if _has_any_term(m, ("gravel", "aggregate")):
        return "GRAVEL_AGGREGATE"
    if _has_any_term(m, ("soil", "earth", "mud", "backfill")):
        return "SOIL_EARTH"
    if _has_any_term(m, ("timber", "wood", "lumber", "sawnwood")):
        return "TIMBER"
    if _has_any_term(m, ("stone", "rock", "quarry")):
        return "NATURAL_STONE"
    return "UNKNOWN"

def choose_resolution_material(original_material: Any, normalized_material: Any) -> tuple[str, str]:
    """Choose the safest material text for downstream external/property resolution.

    Qwen normalization is useful when it preserves the BOM product family.  When
    the original BOM contains a recognizable construction family but the normalized
    text changes that family (for example, binding wire -> rebar), the original BOM
    description takes precedence.  This prevents a valid safety-gate rejection from
    being undone by downstream source searches.
    """
    original = "" if original_material is None else str(original_material).strip()
    normalized = "" if normalized_material is None else str(normalized_material).strip()
    ofam = classify_material(original) if original else "UNKNOWN"
    nfam = classify_material(normalized) if normalized else "UNKNOWN"

    if original and ofam != "UNKNOWN":
        if not normalized or nfam == "UNKNOWN" or nfam != ofam:
            return original, "ORIGINAL_BOM_FAMILY_OVERRIDE"
    if normalized:
        return normalized, "QWEN_NORMALIZED_SAME_FAMILY" if ofam == nfam and nfam != "UNKNOWN" else "QWEN_NORMALIZED"
    if original:
        return original, "ORIGINAL_BOM_ONLY"
    return "", "NO_MATERIAL_TEXT"


BIOGENIC_STORAGE_EXCLUDED_FAMILIES = {"TIMBER", "PLYWOOD", "BAMBOO"}


def excludes_biogenic_storage(material: Any) -> bool:
    """Return True for bio-based product families reported without storage credits."""
    return classify_material(material) in BIOGENIC_STORAGE_EXCLUDED_FAMILIES


def process_compatibility(material: Any, process_name: Any) -> tuple[bool, str]:
    """Conservative deterministic compatibility check for an ELCD process title.

    It only vetoes clear family contradictions.  It never selects a replacement.
    """
    family = classify_material(material)
    p = _norm(process_name)
    if not p:
        return False, "selected_process_name_blank"

    # Cross-cutting false positives.
    if family in {"REBAR", "STEEL_WIRE", "STEEL_FASTENER", "GALVANIZED_FLAT_STEEL"} and _has_any(p, ("copper", "aluminium", "aluminum")):
        return False, "ferrous_product_mapped_to_nonferrous_process"

    rules = {
        "CEMENT": (
            lambda: _has_term(p, "cement")
            and not _has_any_term(
                p,
                (
                    "concrete",
                    "mortar",
                    "concrete block",
                    "precast concrete",
                    "pre cast concrete",
                    "ready mixed concrete",
                    "ready mix concrete",
                ),
            ),
            "cement_requires_cement_process_not_downstream_product",
        ),
        "SAND": (lambda: "sand" in p, "sand_requires_sand_process"),
        "GRAVEL_AGGREGATE": (
            lambda: _has_any(p, ("gravel", "aggregate", "crushed stone")),
            "gravel_requires_aggregate_gravel_or_crushed_stone_process",
        ),
        "NATURAL_STONE": (
            lambda: _has_any(p, ("stone", "rock", "quarry", "aggregate")) and not _has_any(p, ("concrete block", "brick")),
            "stone_requires_stone_rock_or_quarry_process",
        ),
        "PLAIN_CONCRETE": (
            lambda: _has_term(p, "concrete") and not _has_any_term(p, ("concrete block", "aerated concrete", "autoclaved", "aac")),
            "plain_concrete_requires_concrete_process_not_cement_or_block",
        ),
        "STONECRETE_BLOCK": (
            lambda: _has_any(p, ("concrete block", "masonry block")) and not _has_any(p, ("aerated", "autoclaved", "aac")),
            "stonecrete_block_rejects_aac_and_non_block_processes",
        ),
        "REBAR": (
            lambda: _has_any(p, (
                "rebar", "reinforcing steel", "reinforcement steel",
                "reinforcing bar", "reinforcement bar", "tmt bar", "tmt steel",
            )),
            "rebar_requires_reinforcing_steel_process",
        ),
        "STEEL_WIRE": (
            lambda: _has_any(p, ("steel wire", "wire drawing", "wire rod", "annealed wire")),
            "binding_wire_requires_steel_wire_family_process",
        ),
        "STEEL_FASTENER": (
            lambda: _has_any(p, ("nail", "fastener", "steel screw", "screw")),
            "nail_requires_steel_fastener_family_process",
        ),
        "GALVANIZED_FLAT_STEEL": (
            lambda: _has_any(p, ("galvanized", "galvanised", "zinc coated", "zinc-coated", "coated steel", "steel sheet", "steel coil")) and not _has_any(p, ("floor panel", "access floor", "sandwich panel")),
            "cgi_requires_galvanized_flat_steel_family_process",
        ),
        "TIMBER": (
            lambda: _has_any(p, ("wood", "timber", "lumber", "sawn", "roundwood")) and "plywood" not in p,
            "timber_requires_wood_or_timber_process",
        ),
        "PLASTER": (
            lambda: _has_any_term(p, ("plaster", "render", "mortar")),
            "plaster_requires_plaster_render_or_mortar_process",
        ),
        "SOIL_EARTH": (
            lambda: _has_any(p, ("soil", "earth", "rammed earth")) and not _has_any(p, ("pipe", "pvc", "waste", "drain", "sewer", "plumbing", "conduit")),
            "earth_material_rejects_soil_pipe_and_non_earth_products",
        ),
        "SOIL_BLOCK": (
            lambda: _has_any(p, ("soil block", "earth block", "compressed earth", "compressed soil", "ceb")) and not _has_any(p, ("aerated concrete", "autoclaved", "aac")),
            "soil_block_requires_earthen_block_process",
        ),
        "PLYWOOD": (
            lambda: _has_any(p, ("plywood", "veneer", "wood panel", "wood based panel", "wood-based panel")),
            "plywood_requires_manufactured_wood_panel_process",
        ),
        "BAMBOO": (lambda: _has_term(p, "bamboo"), "bamboo_requires_bamboo_product_process"),
    }
    if family == "UNKNOWN":
        return True, "unknown_family_no_high_specificity_veto"
    rule = rules.get(family)
    if rule is None:
        return True, "family_without_specific_veto"
    ok = bool(rule[0]())
    return ok, "compatible" if ok else rule[1]


def external_title_compatibility(material: Any, title: Any, *, match_type: str = "DIRECT_PRODUCT", proxy_subject: str | None = None) -> tuple[bool, str]:
    """Check whether an external EPD/literature title belongs to the requested family."""
    family = classify_material(material)
    t = _norm(title)
    s = _norm(proxy_subject)

    if not t:
        return False, "external_title_blank"

    # Negative context always wins for earthen materials.
    if family in {"SOIL_EARTH", "SOIL_BLOCK"} and _has_any(t, ("pipe", "pvc", "waste pipe", "drain", "sewer", "plumbing", "conduit")):
        return False, "earthen_material_rejects_plumbing_pipe_context"

    if match_type == "DIRECT_PRODUCT":
        checks = {
            "CEMENT": lambda: _has_term(t, "cement")
            and not _has_any_term(
                t,
                (
                    "ready mix concrete", "ready mixed concrete", "ready-mix concrete",
                    "ready-mixed concrete", "precast concrete", "pre cast concrete",
                    "concrete block", "masonry block", "mortar", "clinker",
                ),
            ),
            "SAND": lambda: _has_term(t, "sand"),
            "GRAVEL_AGGREGATE": lambda: _has_any_term(t, ("gravel", "aggregate", "crushed stone")),
            "NATURAL_STONE": lambda: _has_any_term(t, ("stone", "rock", "quarry")),
            "REBAR": lambda: _has_any(t, (
                "rebar", "reinforcing steel", "reinforcement steel",
                "reinforcing bar", "reinforcement bar", "tmt bar", "tmt steel",
            )),
            "TIMBER": lambda: _has_any_term(t, ("timber", "wood", "lumber", "sawnwood")) and "plywood" not in t,
            "SOIL_EARTH": lambda: _has_any(t, ("rammed earth", "earth material", "soil material", "earthen")),
            "SOIL_BLOCK": lambda: _has_any(t, ("compressed earth block", "earth block", "soil block", "ceb")),
            "PLYWOOD": lambda: "plywood" in t,
            "BAMBOO": lambda: "bamboo" in t,
            "STEEL_WIRE": lambda: _has_any(t, ("binding wire", "tie wire", "steel wire", "annealed wire")),
            "STEEL_FASTENER": lambda: "nail" in t,
            "GALVANIZED_FLAT_STEEL": lambda: _has_any(t, ("galvan", "zinc coat", "hot dip coat")) and _has_any(t, ("steel", "sheet", "coil")),
            "PLAIN_CONCRETE": lambda: "concrete" in t,
            "STONECRETE_BLOCK": lambda: _has_any(t, ("masonry block", "concrete block")),
            "PLASTER": lambda: _has_any(t, ("plaster", "render")),
        }
        fn = checks.get(family)
        if fn is None:
            return True, "direct_title_no_specific_rule"
        return (True, "direct_title_family_match") if fn() else (False, "direct_title_wrong_family")

    # For a documented product-family proxy, require both the proxy query
    # subject and the retrieved title to resolve to the same deterministic
    # material family as the BOM. This is a validation rule only: it contains
    # no source URL, factor, manufacturer, or predetermined product mapping.
    if match_type == "PRODUCT_PROXY":
        proxy_family = classify_material(proxy_subject)
        title_family = classify_material(title)
        if family == "UNKNOWN":
            return False, "proxy_unknown_material_family"
        if proxy_family != family:
            return False, "proxy_query_wrong_family"
        if title_family != family:
            return False, "proxy_title_wrong_family"
        return True, "proxy_title_family_match"

    return False, "unknown_external_match_type"
