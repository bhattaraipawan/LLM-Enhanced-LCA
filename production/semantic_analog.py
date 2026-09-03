"""Dynamic semantic analog planning with no material lookup table.

The LLM derives the technical family and nearby analog products from the BOM text
at run time.  This module contains no material names, family mappings, densities,
or emission factors.
"""
from __future__ import annotations

import json
import re
from typing import Any

SEMANTIC_ANALOG_VERSION = "1.0"
ANALOG_COUNT = 5

ANALOG_PLAN_SYSTEM_PROMPT = """You are a construction-material semantic analog planner.

Do NOT provide any GWP, density, mass, thickness, or other numerical property.
Given the requested material and the estimation target, infer its broad technical
family from composition, manufacturing process, physical form, and construction
function. Then propose exactly five distinct nearby analog products, ordered from
most technically similar to broader-but-still-defensible.

Important rules:
1. Derive the family from the supplied material description at run time. Do not
   assume a fixed lookup table.
2. Prefer the exact product family first. If exact-product knowledge is sparse,
   move only one semantic level broader at a time.
3. Analog choices should preserve the dominant material chemistry/manufacturing
   route and construction function as far as possible.
4. Do not invent sources or numerical values.
5. Return JSON only with exactly these keys:
family_description, analogs
6. analogs must contain exactly five objects, each with exactly:
analog_material, similarity_basis
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


def infer_analog_plan(matcher, *, material: str, estimation_target: str, target_context: str, feedback: str | None = None) -> dict[str, Any] | None:
    payload = {
        "material": material,
        "estimation_target": estimation_target,
        "target_context": target_context,
        "required_analog_count": ANALOG_COUNT,
        "previous_consensus_feedback": feedback,
    }
    try:
        raw = matcher.generate_with_system(
            ANALOG_PLAN_SYSTEM_PROMPT, payload, max_new_tokens_override=512
        )
    except TypeError:
        # Compatibility with lightweight test/mocked matchers.
        raw = matcher.generate_with_system(ANALOG_PLAN_SYSTEM_PROMPT, payload)
    obj = _parse_json(raw)
    if not isinstance(obj, dict) or set(obj.keys()) != {"family_description", "analogs"}:
        return None
    family = str(obj.get("family_description") or "").strip()
    analogs = obj.get("analogs")
    if not family or not isinstance(analogs, list) or len(analogs) != ANALOG_COUNT:
        return None
    cleaned = []
    seen = set()
    for a in analogs:
        if not isinstance(a, dict) or set(a.keys()) != {"analog_material", "similarity_basis"}:
            return None
        name = str(a.get("analog_material") or "").strip()
        basis = str(a.get("similarity_basis") or "").strip()
        key = name.lower()
        if not name or not basis or key in seen:
            return None
        seen.add(key)
        cleaned.append({"analog_material": name, "similarity_basis": basis})
    return {"family_description": family, "analogs": cleaned, "raw_model_output": raw}
