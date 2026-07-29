"""Routes for Excel bill-of-materials uploads."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import CalculationError, OpenLCAUnavailableError

router = APIRouter(tags=["WBLCA impact assessment"])


def _header_message(message: str | None) -> str | None:
    if not message:
        return None
    cleaned = re.sub(r"[\r\n]+", " ", message).strip()
    return cleaned.encode("ascii", "replace").decode("ascii")[:1500]


async def _limited_request_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Workbook upload exceeds the {limit} byte limit.",
            )

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Workbook upload exceeds the {limit} byte limit.",
            )
    return bytes(body)


@router.post(
    "/calculate_bom_excel",
    summary="Calculate GWP for a bill-of-materials workbook",
)
async def calculate_bom_excel(
    request: Request,
    filename: str = "bom.xlsx",
) -> Response:
    content = await _limited_request_body(
        request,
        request.app.state.settings.bom_max_upload_bytes,
    )
    try:
        download = await run_in_threadpool(
            request.app.state.bom_controller.process,
            content,
            filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenLCAUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CalculationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {
        "Content-Disposition": f'attachment; filename="{download.filename}"',
    }
    status_message = _header_message(download.message)
    if status_message:
        headers["X-Status-Message"] = status_message
    return Response(
        content=download.content,
        media_type="application/zip",
        headers=headers,
    )
