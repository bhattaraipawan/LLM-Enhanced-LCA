import unittest
import math
from dataclasses import replace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from app import create_app
from app.config import Settings
from app.core import LlamaGenerationResult
from app.core.exceptions import CalculationError, OpenLCAUnavailableError
from app.services.material_service import MaterialService
from app.services.openlca_service import OpenLCAService
from app.services.unit_conversion_service import UnitConversionService
from app.routes.bom import _limited_request_body


def test_settings() -> Settings:
    return Settings(
        app_host="127.0.0.1",
        app_port=8000,
        openlca_host="localhost",
        openlca_port=8080,
        openlca_calculation_timeout_seconds=30,
        recreate_product_systems=False,
        show_top_flows=True,
        bom_max_upload_bytes=1024 * 1024,
        llama_model_id="test/model",
        llama_allow_mps=True,
        llama_local_files_only=True,
        llama_max_new_tokens=64,
    )


class _UnavailableOpenLCA:
    @property
    def schema(self):
        raise OpenLCAUnavailableError("openLCA is not available")

    def capability(self):
        return {
            "status": "unavailable",
            "message": "openLCA is not available",
        }


class _UnavailableLlama:
    def generate(self, **_kwargs):
        return LlamaGenerationResult.unavailable(
            "no supported GPU was detected"
        )


class ApplicationStartupTests(unittest.TestCase):
    def test_app_creation_does_not_import_or_load_llama_dependencies(self):
        with patch("app.core.llama.importlib.import_module") as importer:
            application = create_app(test_settings())
        importer.assert_not_called()
        self.assertEqual(application.title, "LLM-Enhanced WBLCA")

    def test_health_does_not_load_llama(self):
        application = create_app(test_settings())
        payload = application.state.system_controller.health()
        self.assertIn(payload["status"], {"ok", "degraded"})
        self.assertEqual(payload["llama"]["status"], "not_loaded")
        self.assertIn("GUI and API are ready", payload["message"])


class GracefulFallbackTests(unittest.TestCase):
    def test_material_result_reports_llama_unavailable(self):
        service = MaterialService(_UnavailableOpenLCA(), _UnavailableLlama())
        result = service.calculate_material("unknown composite")
        self.assertEqual(result["source"], "unavailable")
        self.assertIsNone(result["kg_co2e_per_kg"])
        self.assertIn("Llama is not available", result["message"])

    def test_known_mass_unit_does_not_need_llama(self):
        conversion = UnitConversionService(_UnavailableLlama()).convert(
            "cement", 2, "tonnes"
        )
        self.assertEqual(conversion.quantity_kg, 2000)
        self.assertEqual(conversion.source, "deterministic")

    def test_non_finite_and_negative_quantities_are_rejected(self):
        service = UnitConversionService(_UnavailableLlama())
        for value in ("NaN", math.inf, -1):
            with self.subTest(value=value):
                conversion = service.convert("cement", value, "kg")
                self.assertIsNone(conversion.quantity_kg)
                self.assertEqual(conversion.source, "invalid_quantity")

    def test_openlca_mass_normalization_and_gwp_100a_selection(self):
        self.assertEqual(OpenLCAService._mass_unit_kg_factor("tonne"), 1000)
        self.assertIsNone(OpenLCAService._mass_unit_kg_factor("m3"))
        value = OpenLCAService._gwp_value(
            [
                {"name": "GWP 20a", "amount": 8.0, "unit": "kg CO2e"},
                {"name": "GWP 100a", "amount": 5.0, "unit": "kg CO2e"},
                {"name": "GWP 500a", "amount": 3.0, "unit": "kg CO2e"},
            ]
        )
        self.assertEqual(value, 5.0)

    def test_openlca_wait_has_a_deadline(self):
        service = OpenLCAService(
            replace(test_settings(), openlca_calculation_timeout_seconds=0)
        )

        class NeverReady:
            @staticmethod
            def get_state():
                return type("State", (), {"is_scheduled": True})()

        with self.assertRaisesRegex(CalculationError, "timed out"):
            service._wait_until_ready(NeverReady())


class UploadLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_upload_stops_at_limit(self):
        chunks = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"5678", "more_body": False},
            ]
        )

        async def receive():
            return next(chunks)

        request = Request(
            {"type": "http", "method": "POST", "path": "/", "headers": []},
            receive,
        )
        with self.assertRaises(HTTPException) as raised:
            await _limited_request_body(request, 6)
        self.assertEqual(raised.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
