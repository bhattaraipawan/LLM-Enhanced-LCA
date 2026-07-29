"""Bill-of-materials processing orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.services.material_service import MaterialService
from app.services.unit_conversion_service import UnitConversionService
from app.utils.spreadsheet import bom_items_from_xlsx
from app.utils.text import material_name_from_query, normalize_process_name


def _join_messages(*messages: str | None) -> str:
    output: list[str] = []
    for message in messages:
        cleaned = str(message or "").strip()
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return " ".join(output)


@dataclass(frozen=True, slots=True)
class BomProcessingResult:
    rows: list[list[Any]]
    messages: tuple[str, ...]


class BomService:
    def __init__(
        self,
        material_service: MaterialService,
        conversion_service: UnitConversionService,
    ) -> None:
        self.material_service = material_service
        self.conversion_service = conversion_service

    def process_xlsx(self, content: bytes) -> BomProcessingResult:
        return self.process_items(bom_items_from_xlsx(content))

    def process_items(self, items: list[dict[str, Any]]) -> BomProcessingResult:
        rows: list[list[Any]] = [
            [
                "row_number",
                "material name",
                "quantity",
                "unit",
                "quantity_kg",
                "kg_co2e_per_kg",
                "total_kg_co2e",
                "source",
                "conversion_source",
                "conversion_notes",
                "process_name",
                "message",
            ]
        ]
        calculation_cache: dict[str, dict[str, Any]] = {}
        messages: list[str] = []

        for item in items:
            material = str(item["description"])
            conversion = self.conversion_service.convert(
                material, item["quantity"], item["unit"]
            )
            if conversion.quantity_kg is None and conversion.message is None:
                conversion = self.conversion_service.retry(
                    material, item["quantity"], item["unit"]
                )

            cache_key = normalize_process_name(material)
            result = calculation_cache.get(cache_key)
            if result is None:
                result = self.material_service.calculate_material(material)
                calculation_cache[cache_key] = result

            quantity_kg = conversion.quantity_kg
            kg_co2e_per_kg = result["kg_co2e_per_kg"]
            valid_quantity = (
                isinstance(quantity_kg, (int, float))
                and math.isfinite(quantity_kg)
                and quantity_kg >= 0
            )
            valid_gwp = (
                isinstance(kg_co2e_per_kg, (int, float))
                and math.isfinite(kg_co2e_per_kg)
                and kg_co2e_per_kg >= 0
            )
            if not valid_quantity or not valid_gwp:
                total_kg_co2e: float | str = "MISSING"
            else:
                total_kg_co2e = quantity_kg * kg_co2e_per_kg

            message = _join_messages(conversion.message, result.get("message"))
            if message and message not in messages:
                messages.append(message)
            rows.append(
                [
                    item["row_number"],
                    material_name_from_query(material),
                    item["quantity"],
                    item["unit"],
                    quantity_kg,
                    kg_co2e_per_kg,
                    total_kg_co2e,
                    result["source"],
                    conversion.source,
                    conversion.notes,
                    result["process_name"],
                    message,
                ]
            )
        return BomProcessingResult(rows=rows, messages=tuple(messages))
