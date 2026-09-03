"""Deterministic quantity conversion for openLCA reference units.

This module never invents density, thickness, or item mass. Required physical
properties can come from the user/project BOM or from the separate audited
property_resolver. Python performs only deterministic arithmetic here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversionResult:
    ok: bool
    quantity_in_ref_unit: float | None
    normalized_ref_unit: str | None
    method: str
    message: str | None = None


def norm_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    u = str(unit).strip().lower().replace("³", "3").replace("²", "2")
    aliases = {
        "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
        "gram": "g", "grams": "g",
        "ton": "t", "tons": "t", "tonne": "t", "tonnes": "t", "metric ton": "t", "metric tonne": "t",
        "m^3": "m3", "cu.m.": "m3", "cu.m": "m3", "cubic meter": "m3", "cubic metre": "m3",
        "liter": "l", "liters": "l", "litre": "l", "litres": "l", "dm3": "l", "dm^3": "l",
        "cm^3": "cm3", "cubic centimeter": "cm3", "cubic centimeters": "cm3",
        "cubic centimetre": "cm3", "cubic centimetres": "cm3",
        "m^2": "m2", "sq.m.": "m2", "sq.m": "m2", "square meter": "m2", "square metre": "m2",
        "cm^2": "cm2", "square centimeter": "cm2", "square centimeters": "cm2",
        "square centimetre": "cm2", "square centimetres": "cm2",
        "mm^2": "mm2", "square millimeter": "mm2", "square millimeters": "mm2",
        "square millimetre": "mm2", "square millimetres": "mm2",
        "nos.": "item", "nos": "item", "no.": "item", "number": "item", "piece": "item", "pieces": "item", "pcs": "item",
        "kilowatt hour": "kwh", "kilowatt-hour": "kwh",
        "megajoule": "mj",
    }
    return aliases.get(u, u)


def _mass_to_kg(q: float, unit: str) -> float | None:
    return {"kg": q, "g": q / 1000.0, "t": q * 1000.0}.get(unit)


def _kg_to_mass(qkg: float, ref: str) -> float | None:
    return {"kg": qkg, "g": qkg * 1000.0, "t": qkg / 1000.0}.get(ref)


def _volume_to_m3(q: float, unit: str) -> float | None:
    return {"m3": q, "l": q / 1000.0, "cm3": q / 1_000_000.0}.get(unit)


def _m3_to_volume(qm3: float, ref: str) -> float | None:
    return {"m3": qm3, "l": qm3 * 1000.0, "cm3": qm3 * 1_000_000.0}.get(ref)


def _area_to_m2(q: float, unit: str) -> float | None:
    return {"m2": q, "cm2": q / 10_000.0, "mm2": q / 1_000_000.0}.get(unit)


def _m2_to_area(qm2: float, ref: str) -> float | None:
    return {"m2": qm2, "cm2": qm2 * 10_000.0, "mm2": qm2 * 1_000_000.0}.get(ref)


def convert_quantity(
    quantity: float,
    source_unit: str,
    ref_unit: str,
    *,
    density_kg_m3: Optional[float] = None,
    thickness_mm: Optional[float] = None,
    mass_per_item_kg: Optional[float] = None,
    conversion_factor_to_ref_unit: Optional[float] = None,
) -> ConversionResult:
    q = float(quantity)
    src = norm_unit(source_unit)
    ref = norm_unit(ref_unit)

    if q < 0:
        return ConversionResult(False, None, ref, "invalid", "Quantity cannot be negative.")
    if not src or not ref:
        return ConversionResult(False, None, ref or None, "invalid", "Source or reference unit is blank.")
    if src == ref:
        return ConversionResult(True, q, ref, "same_unit")

    if conversion_factor_to_ref_unit is not None:
        f = float(conversion_factor_to_ref_unit)
        if f <= 0:
            return ConversionResult(False, None, ref, "explicit_factor", "Conversion factor must be > 0.")
        return ConversionResult(True, q * f, ref, "explicit_factor")

    # Pure mass conversions.
    qkg = _mass_to_kg(q, src)
    if qkg is not None and ref in {"kg", "g", "t"}:
        return ConversionResult(True, _kg_to_mass(qkg, ref), ref, "mass_unit_conversion")

    # Pure volume conversions. These are universal SI relationships and require
    # no material density or other product-specific property.
    qm3 = _volume_to_m3(q, src)
    if qm3 is not None and ref in {"m3", "l", "cm3"}:
        return ConversionResult(True, _m3_to_volume(qm3, ref), ref, "volume_unit_conversion")

    # Pure area conversions. These are universal SI relationships and require
    # no thickness, density, or product-specific property.
    qm2 = _area_to_m2(q, src)
    if qm2 is not None and ref in {"m2", "cm2", "mm2"}:
        return ConversionResult(True, _m2_to_area(qm2, ref), ref, "area_unit_conversion")

    # Volume to mass using user-supplied density.
    if src == "m3" and ref in {"kg", "g", "t"}:
        if density_kg_m3 is None or float(density_kg_m3) <= 0:
            return ConversionResult(False, None, ref, "volume_to_mass", "density_kg_m3 is required.")
        qkg = q * float(density_kg_m3)
        return ConversionResult(True, _kg_to_mass(qkg, ref), ref, "volume_to_mass")

    # Mass to volume using density.
    if src in {"kg", "g", "t"} and ref == "m3":
        if density_kg_m3 is None or float(density_kg_m3) <= 0:
            return ConversionResult(False, None, ref, "mass_to_volume", "density_kg_m3 is required.")
        qkg = _mass_to_kg(q, src)
        return ConversionResult(True, qkg / float(density_kg_m3), ref, "mass_to_volume")

    # Mass to area using thickness and density.
    if src in {"kg", "g", "t"} and ref == "m2":
        if thickness_mm is None or float(thickness_mm) <= 0:
            return ConversionResult(False, None, ref, "mass_to_area", "thickness_mm is required.")
        if density_kg_m3 is None or float(density_kg_m3) <= 0:
            return ConversionResult(False, None, ref, "mass_to_area", "density_kg_m3 is required.")
        qkg = _mass_to_kg(q, src)
        volume_m3 = qkg / float(density_kg_m3)
        area_m2 = volume_m3 / (float(thickness_mm) / 1000.0)
        return ConversionResult(True, area_m2, ref, "mass_to_area")

    # Mass to item count using a resolved unit mass.
    if src in {"kg", "g", "t"} and ref == "item":
        if mass_per_item_kg is None or float(mass_per_item_kg) <= 0:
            return ConversionResult(False, None, ref, "mass_to_count", "mass_per_item_kg is required.")
        qkg = _mass_to_kg(q, src)
        return ConversionResult(True, qkg / float(mass_per_item_kg), ref, "mass_to_count")

    # Area <-> volume using an explicit/BOM-resolved thickness.
    if src == "m2" and ref == "m3":
        if thickness_mm is None or float(thickness_mm) <= 0:
            return ConversionResult(False, None, ref, "area_to_volume", "thickness_mm is required.")
        return ConversionResult(True, q * float(thickness_mm) / 1000.0, ref, "area_to_volume")

    if src == "m3" and ref == "m2":
        if thickness_mm is None or float(thickness_mm) <= 0:
            return ConversionResult(False, None, ref, "volume_to_area", "thickness_mm is required.")
        return ConversionResult(True, q / (float(thickness_mm) / 1000.0), ref, "volume_to_area")

    # Area to mass using thickness and density.
    if src == "m2" and ref in {"kg", "g", "t"}:
        if thickness_mm is None or float(thickness_mm) <= 0:
            return ConversionResult(False, None, ref, "area_to_mass", "thickness_mm is required.")
        if density_kg_m3 is None or float(density_kg_m3) <= 0:
            return ConversionResult(False, None, ref, "area_to_mass", "density_kg_m3 is required.")
        volume_m3 = q * float(thickness_mm) / 1000.0
        qkg = volume_m3 * float(density_kg_m3)
        return ConversionResult(True, _kg_to_mass(qkg, ref), ref, "area_to_mass")

    # Count to mass using user-supplied unit mass.
    if src == "item" and ref in {"kg", "g", "t"}:
        if mass_per_item_kg is None or float(mass_per_item_kg) <= 0:
            return ConversionResult(False, None, ref, "count_to_mass", "mass_per_item_kg is required.")
        qkg = q * float(mass_per_item_kg)
        return ConversionResult(True, _kg_to_mass(qkg, ref), ref, "count_to_mass")

    # Simple energy conversion.
    if src == "mj" and ref == "kwh":
        return ConversionResult(True, q / 3.6, ref, "energy_unit_conversion")
    if src == "kwh" and ref == "mj":
        return ConversionResult(True, q * 3.6, ref, "energy_unit_conversion")

    return ConversionResult(
        False, None, ref, "unsupported",
        f"No deterministic conversion rule from '{source_unit}' to '{ref_unit}'. Supply Conversion_factor_to_ref_unit if justified."
    )


def equivalent_reference_units(reference_unit: str) -> tuple[str, ...]:
    """Return dimensionally equivalent units that require no material property.

    The mapping contains only universal unit relationships. It intentionally
    contains no material-specific factor, density, thickness, item mass, or
    plausibility value. The requested unit is always returned first.
    """
    ref = norm_unit(reference_unit)
    groups = (
        ("kg", "g", "t"),
        ("m3", "l", "cm3"),
        ("m2", "cm2", "mm2"),
        ("kwh", "mj"),
    )
    for group in groups:
        if ref in group:
            return (ref,) + tuple(u for u in group if u != ref)
    return (ref,) if ref else tuple()


def convert_factor_reference_basis(
    value: float,
    from_reference_unit: str,
    to_reference_unit: str,
) -> float | None:
    """Convert an intensity denominator using deterministic unit arithmetic.

    ``value`` is interpreted as kg CO2e per ``from_reference_unit``. The
    returned value is kg CO2e per ``to_reference_unit``. Conversion is allowed
    only when :func:`convert_quantity` can convert one target reference unit to
    the source reference unit without any material-specific physical property.

    Example logic (not material data): if a factor is expressed per tonne and
    the calculation needs per kilogram, Python converts the denominator using
    the same universal mass-unit conversion already used for BOM quantities.
    """
    try:
        v = float(value)
    except Exception:
        return None
    src = norm_unit(from_reference_unit)
    dst = norm_unit(to_reference_unit)
    if not src or not dst:
        return None
    if src == dst:
        return v
    # For an intensity F / src, obtain how many src units are contained in one
    # dst unit, then scale F by that quantity. No density or product property is
    # supplied here; conversions needing such information fail deterministically.
    q = convert_quantity(1.0, dst, src)
    if not q.ok or q.quantity_in_ref_unit is None:
        return None
    return v * float(q.quantity_in_ref_unit)


def reference_unit_schedule(reference_unit: str, count: int) -> tuple[str, ...]:
    """Build a deterministic multi-unit schedule for independent estimates.

    For dimensions with multiple directly convertible units, calls are rotated
    across those units so a denominator-scale mistake cannot create artificial
    agreement merely because every call was requested in the same basis.
    Non-convertible reference units remain unchanged.
    """
    n = max(0, int(count))
    if n == 0:
        return tuple()
    variants = equivalent_reference_units(reference_unit)
    if not variants:
        return tuple()
    return tuple(variants[i % len(variants)] for i in range(n))
