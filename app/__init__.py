"""Application factory for the LLM-enhanced life-cycle assessment GUI."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.controllers import BomController, MaterialController, SystemController
from app.core import LlamaEngine
from app.routes import bom_router, material_router, system_router, ui_router
from app.services.bom_service import BomService
from app.services.material_service import MaterialService
from app.services.openlca_service import OpenLCAService
from app.services.unit_conversion_service import UnitConversionService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    llama = LlamaEngine(
        model_name=settings.llama_model_id,
        allow_mps=settings.llama_allow_mps,
        local_files_only=settings.llama_local_files_only,
        default_max_new_tokens=min(settings.llama_max_new_tokens, 512),
        max_new_tokens_limit=max(settings.llama_max_new_tokens, 512),
    )
    openlca = OpenLCAService(settings)
    material_service = MaterialService(openlca, llama)
    conversion_service = UnitConversionService(llama)
    bom_service = BomService(material_service, conversion_service)

    application = FastAPI(
        title="LLM-Enhanced WBLCA",
        description=(
            "AI-assisted whole-building life cycle assessment with openLCA "
            "process matching and an optional, in-process local Llama fallback."
        ),
        version="2.0.0",
    )
    application.state.settings = settings
    application.state.llama = llama
    application.state.openlca = openlca
    application.state.material_controller = MaterialController(material_service)
    application.state.bom_controller = BomController(bom_service)
    application.state.system_controller = SystemController(openlca, llama)

    application.include_router(ui_router)
    application.include_router(material_router)
    application.include_router(bom_router)
    application.include_router(system_router)
    return application


__all__ = ["create_app"]
