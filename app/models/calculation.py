"""HTTP and domain models for material calculations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CalculateRequest(BaseModel):
    input: str = Field(min_length=1, description="Natural-language material description")


class CalculateResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    input: str
    source: str
    kg_co2e_per_kg: float | None
    unit: str
    gwp_per_reference_unit: float | None = None
    reference_unit: str | None = None
    process_name: str
    product_system: str | None
    impact_method: str | None
    reference_exchanges: list[dict[str, Any]]
    impacts: list[dict[str, Any]]
    top_flows: list[dict[str, Any]]
    message: str | None = None


class CapabilityStatus(BaseModel):
    status: str
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    openlca: CapabilityStatus
    llama: CapabilityStatus
