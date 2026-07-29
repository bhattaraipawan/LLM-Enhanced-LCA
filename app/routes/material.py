"""Routes for single-material impact calculations."""

from fastapi import APIRouter, HTTPException, Request

from app.core import InvalidInputError
from app.models import CalculateRequest, CalculateResponse

router = APIRouter(tags=["WBLCA impact assessment"])


@router.post(
    "/calculate",
    response_model=CalculateResponse,
    summary="Calculate material GWP from a natural-language description",
)
def calculate(
    payload: CalculateRequest,
    request: Request,
) -> CalculateResponse:
    try:
        return request.app.state.material_controller.calculate(payload.input)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
