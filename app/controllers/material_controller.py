"""Controller for single-material calculations."""

from __future__ import annotations

from app.core import InvalidInputError
from app.models import CalculateResponse
from app.services.material_service import MaterialService


class MaterialController:
    def __init__(self, service: MaterialService) -> None:
        self.service = service

    def calculate(self, material_query: str) -> CalculateResponse:
        cleaned = material_query.strip()
        if not cleaned:
            raise InvalidInputError("'input' cannot be empty")
        return CalculateResponse.model_validate(
            self.service.calculate_material(cleaned)
        )
