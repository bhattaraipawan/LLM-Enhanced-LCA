"""Convert bill-of-material quantities to kilograms."""

from __future__ import annotations

import math
import re
from typing import Any

from app.core.exceptions import LlamaUnavailableError
from app.models import QuantityConversion
from app.utils.json_helpers import parse_json_object


class UnitConversionService:
    def __init__(self, llama: Any) -> None:
        self.llama = llama

    @staticmethod
    def _valid_amount(quantity: Any) -> float | None:
        try:
            amount = float(quantity)
        except (TypeError, ValueError):
            return None
        return amount if math.isfinite(amount) and amount >= 0 else None

    @classmethod
    def deterministic(cls, quantity: Any, unit: str) -> QuantityConversion | None:
        amount = cls._valid_amount(quantity)
        if amount is None:
            return None
        normalized = re.sub(r"\s+", " ", str(unit or "").lower().strip().rstrip("."))
        factors = {
            "kg": 1.0,
            "kgs": 1.0,
            "kilogram": 1.0,
            "kilograms": 1.0,
            "g": 0.001,
            "gram": 0.001,
            "grams": 0.001,
            "ton": 1000.0,
            "tons": 1000.0,
            "tonne": 1000.0,
            "tonnes": 1000.0,
            "metric ton": 1000.0,
            "metric tons": 1000.0,
            "lb": 0.45359237,
            "lbs": 0.45359237,
            "pound": 0.45359237,
            "pounds": 0.45359237,
            "oz": 0.028349523125,
            "ounce": 0.028349523125,
            "ounces": 0.028349523125,
        }
        factor = factors.get(normalized)
        if factor is None:
            return None
        return QuantityConversion(
            quantity_kg=amount * factor,
            source="deterministic",
            notes=f"Converted from {quantity} {unit}.",
        )

    @staticmethod
    def material_density_kg_per_m3(material: str) -> float | None:
        text = str(material or "").lower()
        for keyword, density in (
            ("sand", 1600.0),
            ("aggregate", 1500.0),
            ("aggregates", 1500.0),
            ("gravel", 1500.0),
            ("stone", 1600.0),
            ("mud", 1600.0),
            ("soil", 1500.0),
            ("earth", 1500.0),
            ("clay", 1600.0),
        ):
            if keyword in text:
                return density
        return None

    @classmethod
    def construction_load(
        cls, material: str, quantity: Any, unit: str
    ) -> QuantityConversion | None:
        amount = cls._valid_amount(quantity)
        if amount is None:
            return None
        normalized = re.sub(r"\s+", " ", str(unit or "").lower().strip().rstrip("."))
        if normalized not in {
            "tractor",
            "tracter",
            "tractor load",
            "tracter load",
            "tractor trolley",
            "tracter trolley",
        }:
            return None
        density = cls.material_density_kg_per_m3(material)
        if density is None:
            return None
        trolley_m3 = 2.83
        return QuantityConversion(
            quantity_kg=amount * trolley_m3 * density,
            source="estimated_volume_density",
            notes=(
                f"Assumed 1 tractor trolley/load = {trolley_m3} m3 and "
                f"{density} kg/m3 bulk density for {material}."
            ),
        )

    @classmethod
    def volume_or_area(
        cls, material: str, quantity: Any, unit: str
    ) -> QuantityConversion | None:
        amount = cls._valid_amount(quantity)
        if amount is None:
            return None
        normalized = str(unit or "").lower().strip().rstrip(".")
        normalized = re.sub(
            r"\s+", " ", normalized.replace("³", "3").replace("²", "2")
        )
        material_text = str(material or "").lower()
        volume_units = {
            "m3",
            "m^3",
            "cum",
            "cubic meter",
            "cubic meters",
            "cubic metre",
            "cubic metres",
        }
        area_units = {
            "m2",
            "m^2",
            "sqm",
            "sq m",
            "square meter",
            "square meters",
            "square metre",
            "square metres",
        }
        if normalized in volume_units:
            density = cls.material_density_kg_per_m3(material)
            if density is None and ("wood" in material_text or "timber" in material_text):
                density = 600.0
            if density is None:
                return None
            return QuantityConversion(
                quantity_kg=amount * density,
                source="estimated_volume_density",
                notes=f"Converted {quantity} {unit} using {density} kg/m3 density.",
            )
        if normalized not in area_units:
            return None

        thickness_match = re.search(r"(\d+(?:\.\d+)?)\s*mm", material_text)
        thickness_mm: float | None = None
        density: float | None = None
        if "plywood" in material_text or "ply" in material_text:
            density = 600.0
            thickness_mm = (
                float(thickness_match.group(1)) if thickness_match else None
            )
        elif any(
            term in material_text
            for term in (
                "gi sheet",
                "galvanized",
                "galvanised",
                "galvanise",
                "corrugated",
                "steel sheet",
            )
        ):
            density = 7850.0
            thickness_mm = (
                float(thickness_match.group(1)) if thickness_match else 0.5
            )
        elif "plaster" in material_text:
            density = 1800.0
            thickness_mm = (
                float(thickness_match.group(1)) if thickness_match else 12.0
            )
        if density is None or thickness_mm is None:
            return None
        return QuantityConversion(
            quantity_kg=amount * (thickness_mm / 1000.0) * density,
            source="estimated_area_density",
            notes=(
                f"Converted {quantity} {unit} using {thickness_mm:g} mm thickness "
                f"and {density} kg/m3 density."
            ),
        )

    @classmethod
    def count(
        cls, material: str, quantity: Any, unit: str
    ) -> QuantityConversion | None:
        amount = cls._valid_amount(quantity)
        if amount is None:
            return None
        normalized = re.sub(r"\s+", " ", str(unit or "").lower().strip().rstrip("."))
        if normalized not in {
            "no",
            "nos",
            "number",
            "numbers",
            "piece",
            "pieces",
            "pc",
            "pcs",
            "each",
            "ea",
        }:
            return None

        material_text = str(material or "").lower()
        per_piece_kg: float | None = None
        notes = ""
        bamboo_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches)", material_text
        )
        if "bamboo" in material_text:
            diameter_in = float(bamboo_match.group(1)) if bamboo_match else 2.0
            diameter_m = diameter_in * 0.0254
            wall_thickness_m = min(0.012, diameter_m * 0.18)
            length_m = 3.0
            density = 700.0
            outer_area = 3.141592653589793 * (diameter_m / 2) ** 2
            inner_radius = max((diameter_m / 2) - wall_thickness_m, 0)
            inner_area = 3.141592653589793 * inner_radius**2
            per_piece_kg = (outer_area - inner_area) * length_m * density
            notes = (
                f"Assumed each bamboo is {length_m} m long, {diameter_in} inch "
                f"diameter, {wall_thickness_m:.3f} m wall thickness, and "
                f"{density} kg/m3 density."
            )
        elif "washer" in material_text:
            per_piece_kg = 0.00143
            notes = "Assumed one small steel/bitumen washer weighs 0.00143 kg."
        if per_piece_kg is None:
            return None
        return QuantityConversion(
            quantity_kg=amount * per_piece_kg,
            source="estimated_count_weight",
            notes=notes,
        )

    def _llama_conversion(
        self, material: str, quantity: Any, unit: str, *, retry: bool = False
    ) -> QuantityConversion:
        retry_instruction = (
            "A previous conversion failed. Make a reasonable construction assumption."
            if retry
            else "Use typical construction/material assumptions when needed."
        )
        prompt = f"""
Convert this bill-of-quantities item to kilograms for openLCA.

Material description: {material}
Quantity: {quantity}
Unit: {unit}

{retry_instruction}
Return ONLY JSON like:
{{
  "quantity_kg": 123.45,
  "normalized_unit": "kg",
  "conversion_notes": "brief assumption"
}}
Use null only if conversion is impossible.
"""
        try:
            generated = self.llama.generate(prompt=prompt, max_new_tokens=256)
        except LlamaUnavailableError as exc:
            return QuantityConversion(
                quantity_kg=None,
                source="unavailable",
                notes=str(exc),
                message=str(exc),
            )
        except Exception as exc:
            message = f"Llama is not available: {exc}"
            return QuantityConversion(
                quantity_kg=None,
                source="unavailable",
                notes=message,
                message=message,
            )

        if not getattr(generated, "available", True):
            message = (
                getattr(generated, "message", None)
                or "Llama is not available on this device."
            )
            return QuantityConversion(
                quantity_kg=None,
                source="unavailable",
                notes=message,
                message=message,
            )
        data = parse_json_object(getattr(generated, "raw_output", ""))
        numeric_result = getattr(generated, "result", {}) or {}
        if isinstance(numeric_result, dict):
            data.update(numeric_result)
        try:
            value = float(data.get("quantity_kg"))
        except (TypeError, ValueError):
            value = None
        if value is not None and (not math.isfinite(value) or value < 0):
            value = None
        return QuantityConversion(
            quantity_kg=value,
            normalized_unit=str(data.get("normalized_unit") or "kg"),
            source="llama_unit_conversion_retry" if retry else "llama_unit_conversion",
            notes=str(data.get("conversion_notes") or ""),
        )

    def convert(self, material: str, quantity: Any, unit: str) -> QuantityConversion:
        amount = self._valid_amount(quantity)
        if amount is None:
            message = "Quantity must be a finite, non-negative number."
            return QuantityConversion(
                quantity_kg=None,
                source="invalid_quantity",
                notes=message,
                message=message,
            )
        if amount == 0:
            return QuantityConversion(
                quantity_kg=0.0,
                source="deterministic",
                notes=f"Zero {unit or 'units'} converts to zero kg.",
            )
        for converter in (
            lambda: self.deterministic(quantity, unit),
            lambda: self.construction_load(material, quantity, unit),
            lambda: self.volume_or_area(material, quantity, unit),
            lambda: self.count(material, quantity, unit),
        ):
            converted = converter()
            if converted is not None:
                return converted
        return self._llama_conversion(material, quantity, unit)

    def retry(
        self, material: str, quantity: Any, unit: str
    ) -> QuantityConversion:
        return self._llama_conversion(material, quantity, unit, retry=True)
