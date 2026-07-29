"""Controller for uploaded BOM workbooks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.bom_service import BomService
from app.utils.spreadsheet import bom_download_zip


@dataclass(frozen=True, slots=True)
class BomDownload:
    content: bytes
    filename: str
    message: str | None


class BomController:
    def __init__(self, service: BomService) -> None:
        self.service = service

    def process(self, content: bytes, filename: str) -> BomDownload:
        if not content:
            raise ValueError("Upload a .xlsx Excel file in the request body.")
        base_name = re.sub(r"\.xlsx$", "", filename, flags=re.IGNORECASE)
        base_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name).strip("_") or "bom"
        result = self.service.process_xlsx(content)
        return BomDownload(
            content=bom_download_zip(result.rows, base_name),
            filename=f"{base_name}_gwp_results.zip",
            message=" ".join(result.messages) or None,
        )
