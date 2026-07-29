"""Typed bill-of-materials values used by the service layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BomItem:
    row_number: int
    description: str
    quantity: Any
    unit: str


@dataclass(frozen=True, slots=True)
class QuantityConversion:
    quantity_kg: float | None
    normalized_unit: str = "kg"
    source: str = "unavailable"
    notes: str = ""
    message: str | None = None
