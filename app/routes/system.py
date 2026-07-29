"""Health and diagnostic routes."""

from fastapi import APIRouter, HTTPException, Request

from app.core.exceptions import OpenLCAUnavailableError

router = APIRouter(tags=["System"])


@router.get("/health")
def health(request: Request) -> dict:
    return request.app.state.system_controller.health()


@router.get("/llama_status")
def llama_status(request: Request) -> dict:
    return request.app.state.system_controller.llama_status()


@router.get("/openlca_debug")
def openlca_debug(request: Request) -> dict:
    try:
        return request.app.state.system_controller.openlca_debug()
    except OpenLCAUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
