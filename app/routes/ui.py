"""Dependency-independent HTML GUI routes."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)
_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "index.html"


def _gui() -> HTMLResponse:
    return HTMLResponse(_TEMPLATE_PATH.read_text(encoding="utf-8"))


@router.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return _gui()


@router.get("/upload_bom", response_class=HTMLResponse)
def upload_bom() -> HTMLResponse:
    return _gui()
