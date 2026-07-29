"""Controller for health and diagnostics endpoints."""

from __future__ import annotations

from typing import Any

from app.services.openlca_service import OpenLCAService


class SystemController:
    def __init__(self, openlca: OpenLCAService, llama: Any) -> None:
        self.openlca = openlca
        self.llama = llama

    def health(self) -> dict[str, Any]:
        openlca_status = self.openlca.capability(probe=True)
        llama_snapshot = self.llama.status()
        llama_state = str(llama_snapshot.get("state") or "uninitialized")
        if llama_state == "ready":
            llama_status = "available"
            llama_message = (
                f"Loaded on {llama_snapshot.get('device')}."
                if llama_snapshot.get("device")
                else "The local model is ready."
            )
        elif llama_state in {"unavailable", "error"}:
            llama_status = "unavailable"
            llama_message = (
                llama_snapshot.get("message") or "Llama is not available."
            )
        else:
            llama_status = "not_loaded"
            llama_message = (
                "Llama is not loaded. It is checked and loaded lazily only when "
                "an LLM fallback is needed."
            )

        capability_states = {
            openlca_status.get("status"),
            llama_status,
        }
        app_status = (
            "degraded"
            if "unavailable" in capability_states
            else "ok"
        )
        return {
            "status": app_status,
            "message": "The GUI and API are ready.",
            "openlca": openlca_status,
            "llama": {
                "status": llama_status,
                "available": llama_snapshot.get("available", False),
                "message": llama_message,
                "device": llama_snapshot.get("device"),
                "model": llama_snapshot.get("model_name"),
            },
        }

    def openlca_debug(self) -> dict[str, Any]:
        return self.openlca.debug_snapshot()

    def llama_status(self) -> dict[str, Any]:
        return self.llama.status()
