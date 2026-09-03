"""Material-agnostic structural validation for exploratory LCA fallbacks.

This module deliberately contains NO material names, material-family GWP caps,
density ranges, item-mass ranges, or prescribed environmental/property values.
It validates only universal structure: finite positive numbers, supported units,
and internally ordered uncertainty intervals.

No material-specific numerical magnitude prior is used for terminal model-only
estimates. Evidence-supported rows are controlled by retrieved-source validation,
and database-anchored Class-4 rows use frozen snapshot factors. A terminal model
number, when all anchored routes fail, remains explicitly unverified and is gated
only by universal numeric/unit structure plus deterministic product identity.
"""
from __future__ import annotations

import math
from typing import Any

from unit_conversion import norm_unit

GUARDRAIL_VERSION = "2.1-material-agnostic-no-terminal-consensus"

_SUPPORTED_FACTOR_UNITS = {"kg", "g", "t", "m3", "l", "cm3", "m2", "cm2", "mm2", "item", "kwh", "mj"}


def emission_factor_cap(material: Any, reference_unit: Any) -> None:
    """Compatibility shim: no material-specific numerical cap exists."""
    return None


def emission_factor_plausible(
    material: Any,
    value: Any,
    reference_unit: Any,
    lower: Any = None,
    upper: Any = None,
) -> tuple[bool, str]:
    try:
        value = float(value)
    except Exception:
        return False, "factor_not_numeric"
    if not math.isfinite(value) or value <= 0:
        return False, "factor_must_be_finite_and_positive"
    ref = norm_unit(reference_unit)
    if ref not in _SUPPORTED_FACTOR_UNITS:
        return False, f"unsupported_reference_unit:{ref or 'blank'}"

    lo = None if lower is None else float(lower)
    hi = None if upper is None else float(upper)
    if lo is not None and (not math.isfinite(lo) or lo <= 0 or lo >= value):
        return False, "invalid_lower_bound"
    if hi is not None and (not math.isfinite(hi) or hi <= value):
        return False, "invalid_upper_bound"
    if lo is not None and hi is not None and hi <= lo:
        return False, "invalid_uncertainty_interval"
    return True, "ok"


def emission_guardrail_context(material: Any, reference_unit: Any) -> dict[str, Any]:
    return {
        "reference_unit": norm_unit(reference_unit),
        "material_specific_numeric_limits_used": False,
        "validation_role": "structure/unit validation only; no material-specific numerical magnitude prior",
    }


def property_range(material: Any, kind: str) -> None:
    """Compatibility shim: no material-specific physical-property range exists."""
    return None


def property_plausible(
    material: Any,
    kind: str,
    value: Any,
    lower: Any = None,
    upper: Any = None,
) -> tuple[bool, str]:
    try:
        value = float(value)
    except Exception:
        return False, "property_not_numeric"
    if not math.isfinite(value) or value <= 0:
        return False, "property_must_be_finite_and_positive"
    lo = None if lower is None else float(lower)
    hi = None if upper is None else float(upper)
    if lo is not None and (not math.isfinite(lo) or lo <= 0 or lo >= value):
        return False, "invalid_property_lower_bound"
    if hi is not None and (not math.isfinite(hi) or hi <= value):
        return False, "invalid_property_upper_bound"
    if lo is not None and hi is not None and hi <= lo:
        return False, "invalid_property_uncertainty_interval"
    return True, "ok"


def property_guardrail_context(material: Any, kind: str) -> dict[str, Any]:
    return {
        "requested_property": kind,
        "material_specific_numeric_limits_used": False,
        "validation_role": "structure/unit validation only; no material-specific numerical magnitude prior",
    }
