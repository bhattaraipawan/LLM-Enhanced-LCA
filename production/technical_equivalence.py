"""Dynamic technical-equivalence search planning.

This module helps the relaxed phase of Class 3 broaden live searches when exact wording does not find
usable evidence.  It never returns GWP factors, densities, thicknesses, item
masses, source URLs, EPD IDs, or other numerical environmental/property values.
All returned search terms are checked against the deterministic material family
before they are allowed to drive a relaxed Class-3 product-proxy search.
"""
from __future__ import annotations

import json
import re
from typing import Any

from material_taxonomy import classify_material, process_compatibility

TECHNICAL_EQUIVALENCE_VERSION = "1.1-production-compatibility-filter"

TECHNICAL_EQUIVALENCE_SYSTEM_PROMPT = """You are a construction-product terminology planner for evidence retrieval.

Your task is ONLY to propose alternative technical names that may be used by EPDs,
LCA studies, standards, or manufacturers for the SAME construction-product family.
Do not provide any environmental factor, density, thickness, mass, unit conversion,
price, source, citation, URL, publication, manufacturer, or numerical property.

Rules:
1. Preserve the requested product's dominant material chemistry, manufacturing route,
   physical form, and construction function.
2. Different wording is allowed. A broader product description is allowed only when
   it remains in the same technical product family and is a defensible relaxed Class-3 proxy.
3. Do not jump to a constituent, downstream product, or unrelated material family.
4. Return concise search terminology, not explanations or sources.
5. Return JSON only with exactly these keys: normalized_product, search_terms.
6. search_terms must be a list of objects with exactly: term, equivalence_basis.
"""


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads((raw or "").strip())
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


def infer_technical_equivalents(matcher, *, material: str, target_context: str) -> dict[str, Any] | None:
    payload = {
        "material": material,
        "target_context": target_context,
        "purpose": "relaxed Class-3 external-verified EPD/LCA search terminology only",
    }
    try:
        raw = matcher.generate_with_system(
            TECHNICAL_EQUIVALENCE_SYSTEM_PROMPT, payload, max_new_tokens_override=420
        )
    except TypeError:
        raw = matcher.generate_with_system(TECHNICAL_EQUIVALENCE_SYSTEM_PROMPT, payload)
    obj = _parse_json(raw)
    if not isinstance(obj, dict) or set(obj.keys()) != {"normalized_product", "search_terms"}:
        return None
    normalized = str(obj.get("normalized_product") or "").strip()
    terms = obj.get("search_terms")
    if not normalized or not isinstance(terms, list):
        return None

    requested_family = classify_material(material)
    cleaned = []
    seen = set()
    for rec in terms:
        if not isinstance(rec, dict) or set(rec.keys()) != {"term", "equivalence_basis"}:
            continue
        term = str(rec.get("term") or "").strip()
        basis = str(rec.get("equivalence_basis") or "").strip()
        key = term.lower()
        if not term or not basis or key in seen:
            continue
        # Known materials may use a terminology/proxy phrase that is broader than
        # the taxonomy label as long as the existing deterministic production
        # compatibility gate still considers it technically defensible.  This is
        # semantic validation only; no emission factor or property is encoded.
        if requested_family == "UNKNOWN":
            continue
        compatible, _ = process_compatibility(material, term)
        if not compatible:
            continue
        seen.add(key)
        cleaned.append({"term": term, "equivalence_basis": basis})
    if not cleaned:
        return None
    return {
        "normalized_product": normalized,
        "search_terms": cleaned,
        "raw_model_output": raw,
        "version": TECHNICAL_EQUIVALENCE_VERSION,
    }
