"""Lazy openLCA IPC integration.

Importing this module does not require an openLCA server to be running.  A
client is created only when an openLCA operation is requested.
"""

from __future__ import annotations

import re
import socket
import threading
import time
from typing import Any

from app.config import IMPACT_METHOD_CANDIDATES, Settings
from app.core.exceptions import CalculationError, OpenLCAUnavailableError


class OpenLCAService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._olca: Any | None = None
        self._schema: Any | None = None
        self._client: Any | None = None
        self._load_lock = threading.Lock()
        self._calculation_lock = threading.Lock()
        self._impact_units: dict[str, str] = {}
        self._last_error: str | None = None
        self._last_ok = False

    @property
    def schema(self) -> Any:
        self._ensure_runtime()
        return self._schema

    @property
    def client(self) -> Any:
        self._ensure_runtime()
        if self._client is None:
            try:
                self._client = self._olca.Client(self.settings.openlca_url)
            except Exception as exc:  # pragma: no cover - version-specific constructor
                self._remember_error(exc)
                raise OpenLCAUnavailableError(self._unavailable_message()) from exc
        return self._client

    def _ensure_runtime(self) -> None:
        if self._olca is not None and self._schema is not None:
            return
        with self._load_lock:
            if self._olca is not None and self._schema is not None:
                return
            try:
                import olca_ipc
                import olca_schema
            except ImportError as exc:
                self._last_error = (
                    "openLCA Python packages are not installed. "
                    "Install the base requirements first."
                )
                raise OpenLCAUnavailableError(self._unavailable_message()) from exc
            self._olca = olca_ipc
            self._schema = olca_schema

    def _remember_error(self, exc: BaseException) -> None:
        self._last_ok = False
        detail = str(exc).strip() or exc.__class__.__name__
        self._last_error = detail

    def _remember_success(self) -> None:
        self._last_ok = True
        self._last_error = None

    def _unavailable_message(self) -> str:
        suffix = f" Details: {self._last_error}" if self._last_error else ""
        return (
            f"openLCA is not available at {self.settings.openlca_url}. "
            "Start openLCA, open the database, and enable the IPC server."
            f"{suffix}"
        )

    def capability(self, *, probe: bool = False) -> dict[str, str | None]:
        try:
            self._ensure_runtime()
        except OpenLCAUnavailableError as exc:
            return {"status": "unavailable", "message": str(exc)}
        if probe:
            try:
                with socket.create_connection(
                    (self.settings.openlca_host, self.settings.openlca_port),
                    timeout=0.75,
                ):
                    pass
            except OSError as exc:
                self._remember_error(exc)
            else:
                self._remember_success()
        if self._last_ok:
            return {
                "status": "available",
                "message": f"IPC port is reachable at {self.settings.openlca_url}",
            }
        if self._last_error:
            return {"status": "unavailable", "message": self._unavailable_message()}
        return {
            "status": "not_checked",
            "message": f"Configured for {self.settings.openlca_url}",
        }

    def descriptors(self, model_type: Any, *, strict: bool = False) -> list[Any]:
        try:
            rows = list(self.client.get_descriptors(model_type) or [])
        except Exception as exc:
            self._remember_error(exc)
            if strict:
                raise OpenLCAUnavailableError(self._unavailable_message()) from exc
            return []
        self._remember_success()
        return rows

    def names(self, model_type: Any) -> list[str]:
        return sorted(
            descriptor.name
            for descriptor in self.descriptors(model_type)
            if getattr(descriptor, "name", None)
        )

    def find(self, model_type: Any, name: str, *, strict: bool = False) -> Any | None:
        try:
            value = self.client.find(model_type, name)
        except Exception as exc:
            self._remember_error(exc)
            if strict:
                raise OpenLCAUnavailableError(self._unavailable_message()) from exc
            return None
        self._remember_success()
        return value

    def get(self, model_type: Any, uid: str) -> Any | None:
        try:
            value = self.client.get(model_type, uid=uid)
        except Exception as exc:
            self._remember_error(exc)
            raise OpenLCAUnavailableError(self._unavailable_message()) from exc
        self._remember_success()
        return value

    def debug_snapshot(self) -> dict[str, Any]:
        schema = self.schema
        process_rows = self.descriptors(schema.Process, strict=True)
        product_system_rows = self.descriptors(schema.ProductSystem, strict=True)
        impact_method_rows = self.descriptors(schema.ImpactMethod, strict=True)
        return {
            "ipc_url": self.settings.openlca_url,
            "target_database": "BAFU:2025 Version 2",
            "process_count": len(process_rows),
            "product_system_count": len(product_system_rows),
            "impact_method_count": len(impact_method_rows),
            "sample_processes": [
                row.name for row in process_rows[:20] if getattr(row, "name", None)
            ],
            "sample_impact_methods": [
                row.name for row in impact_method_rows[:20] if getattr(row, "name", None)
            ],
        }

    def reference_exchanges(self, process: Any) -> list[dict[str, Any]]:
        schema = self.schema
        process_full = self.get(schema.Process, process.id)
        if process_full is None or not getattr(process_full, "exchanges", None):
            return []
        rows: list[dict[str, Any]] = []
        for exchange in process_full.exchanges:
            if not getattr(exchange, "is_quantitative_reference", False):
                continue
            flow = getattr(exchange, "flow", None)
            unit = getattr(exchange, "unit", None)
            rows.append(
                {
                    "reference_flow": getattr(flow, "name", None),
                    "amount": getattr(exchange, "amount", None),
                    "unit": getattr(unit, "name", None),
                }
            )
        return rows

    def related_product_systems(self, name: str) -> list[Any]:
        schema = self.schema
        return [
            descriptor
            for descriptor in self.descriptors(schema.ProductSystem, strict=True)
            if getattr(descriptor, "name", None) == name
            or str(getattr(descriptor, "name", "")).startswith(f"{name} - ")
        ]

    def _create_product_system(self, process: Any) -> Any:
        schema = self.schema
        config = schema.LinkingConfig(
            prefer_unit_processes=True,
            provider_linking=schema.ProviderLinking.ONLY_DEFAULTS,
        )
        try:
            return self.client.create_product_system(process, config)
        except TypeError:
            process_id = process.id if hasattr(process, "id") else process
            try:
                return self.client.create_product_system(process_id)
            except TypeError:
                return self.client.create_product_system(process_id, "prefer")

    def _product_system_for(self, process: Any) -> Any:
        schema = self.schema
        process_name = process.name
        product_system = self.find(schema.ProductSystem, process_name, strict=True)

        if self.settings.recreate_product_systems:
            for old_system in self.related_product_systems(process_name):
                self.client.delete(old_system)
            product_system = None

        if product_system is None:
            try:
                product_system = self._create_product_system(process)
            except Exception as exc:
                self._remember_error(exc)
                raise CalculationError(
                    f"Could not create a product system from process '{process_name}': {exc}"
                ) from exc
        if product_system is None:
            raise CalculationError(
                f"Could not create a product system from process '{process_name}'."
            )
        return product_system

    def _find_impact_method(self) -> Any | None:
        schema = self.schema
        for method_name in IMPACT_METHOD_CANDIDATES:
            method = self.find(schema.ImpactMethod, method_name)
            if method is not None:
                return method

        target_tokens = {"ipcc", "2013", "gwp", "100a"}
        climate_tokens = {"climate", "change"}
        best_match = None
        best_score = 0
        for descriptor in self.descriptors(schema.ImpactMethod):
            name = getattr(descriptor, "name", None)
            if not name:
                continue
            method_name = name.lower()
            tokens = set(re.findall(r"[a-z0-9]+", method_name))
            score = sum(20 for token in target_tokens if token in tokens)
            score += sum(8 for token in climate_tokens if token in tokens)
            if "100" in method_name and "a" in tokens:
                score += 10
            if "gwp 100a" in method_name or "gwp100" in method_name.replace(" ", ""):
                score += 30
            if "ipcc 2013" in method_name:
                score += 30
            if "carbon feedback" in method_name or "climate-carbon" in method_name:
                score += 5
            if score > best_score:
                best_score = score
                best_match = descriptor
        return best_match if best_score >= 60 else None

    def _impact_unit(self, impact_category: Any) -> str:
        if impact_category is None:
            return ""
        ref_unit = getattr(impact_category, "ref_unit", None)
        if ref_unit:
            return ref_unit
        category_id = impact_category.id
        if category_id in self._impact_units:
            return self._impact_units[category_id]
        schema = self.schema
        category = self.get(schema.ImpactCategory, category_id)
        unit = getattr(category, "ref_unit", "") if category else ""
        self._impact_units[category_id] = unit or ""
        return unit or ""

    @staticmethod
    def _top_flows(result: Any, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        flows = result.get_total_flows()
        rows: list[dict[str, Any]] = []
        for item in sorted(
            flows, key=lambda value: abs(getattr(value, "amount", 0) or 0), reverse=True
        )[:limit]:
            envi_flow = getattr(item, "envi_flow", None)
            flow = getattr(envi_flow, "flow", None)
            rows.append(
                {
                    "direction": "input"
                    if envi_flow and getattr(envi_flow, "is_input", False)
                    else "output",
                    "name": getattr(flow, "name", "unknown flow"),
                    "amount": getattr(item, "amount", None),
                    "unit": getattr(flow, "ref_unit", "") if flow else "",
                }
            )
        return rows

    @staticmethod
    def _gwp_value(impact_rows: list[dict[str, Any]]) -> float | None:
        candidates: list[tuple[int, float]] = []
        for row in impact_rows:
            name = str(row.get("name") or "").lower()
            unit = str(row.get("unit") or "").lower()
            amount = row.get("amount")
            if not isinstance(amount, (int, float)):
                continue
            score = 0
            if "100a" in name or "100 year" in name or "100-year" in name:
                score += 100
            if "gwp" in name or "global warming" in name:
                score += 40
            if "climate change" in name:
                score += 25
            if "co2" in unit:
                score += 10
            if ("20a" in name or "20 year" in name or "500a" in name) and "100a" not in name:
                score -= 50
            if score > 0:
                candidates.append((score, float(amount)))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        value = impact_rows[0].get("amount") if impact_rows else None
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _mass_unit_kg_factor(reference_unit: str | None) -> float | None:
        unit = re.sub(r"\s+", " ", str(reference_unit or "").strip().lower())
        return {
            "kg": 1.0,
            "kilogram": 1.0,
            "kilograms": 1.0,
            "g": 0.001,
            "gram": 0.001,
            "grams": 0.001,
            "mg": 0.000001,
            "milligram": 0.000001,
            "milligrams": 0.000001,
            "t": 1000.0,
            "tonne": 1000.0,
            "tonnes": 1000.0,
            "metric ton": 1000.0,
            "metric tons": 1000.0,
            "lb": 0.45359237,
            "lbs": 0.45359237,
            "pound": 0.45359237,
            "pounds": 0.45359237,
            "oz": 0.028349523125,
            "ounce": 0.028349523125,
            "ounces": 0.028349523125,
        }.get(unit)

    def _wait_until_ready(self, result: Any) -> Any:
        deadline = (
            time.monotonic()
            + self.settings.openlca_calculation_timeout_seconds
        )
        state = result.get_state()
        while getattr(state, "is_scheduled", False):
            if time.monotonic() >= deadline:
                raise CalculationError(
                    "openLCA calculation timed out after "
                    f"{self.settings.openlca_calculation_timeout_seconds} seconds."
                )
            time.sleep(0.5)
            state = result.get_state()
        return state

    def calculate(self, process: Any, material_query: str) -> dict[str, Any]:
        """Calculate impacts for one openLCA process.

        openLCA IPC calculations are serialized because a shared client/result
        handle is not guaranteed to be thread-safe.
        """

        with self._calculation_lock:
            return self._calculate_locked(process, material_query)

    def _calculate_locked(self, process: Any, material_query: str) -> dict[str, Any]:
        schema = self.schema
        reference_rows = self.reference_exchanges(process)
        product_system = self._product_system_for(process)
        setup = schema.CalculationSetup()
        setup.target = product_system
        setup.amount = 1.0
        setup.allocation = schema.AllocationType.PHYSICAL_ALLOCATION
        impact_method = self._find_impact_method()
        if impact_method is not None:
            setup.impact_method = impact_method

        result = None
        try:
            try:
                result = self.client.calculate(setup)
            except Exception as exc:
                self._remember_error(exc)
                raise OpenLCAUnavailableError(self._unavailable_message()) from exc
            if result is None or getattr(result, "error", None) is not None:
                error = getattr(getattr(result, "error", None), "error", None)
                raise CalculationError(f"Calculation failed: {error or 'unknown error'}")

            state = self._wait_until_ready(result)
            if getattr(state, "error", None):
                raise CalculationError(f"Calculation failed: {state.error}")
            if getattr(state, "is_ready", None) is False:
                raise CalculationError("Calculation did not finish.")

            impact_rows: list[dict[str, Any]] = []
            if impact_method is not None:
                for item in result.get_total_impacts():
                    category = item.impact_category
                    impact_rows.append(
                        {
                            "name": category.name,
                            "amount": item.amount,
                            "unit": self._impact_unit(category),
                        }
                    )
            flow_rows = self._top_flows(
                result, 10 if self.settings.show_top_flows else 0
            )
            raw_gwp = self._gwp_value(impact_rows)
            reference_unit = (
                str(reference_rows[0].get("unit") or "")
                if reference_rows
                else ""
            )
            mass_factor = self._mass_unit_kg_factor(reference_unit)
            messages: list[str] = []
            if impact_method is None:
                messages.append(
                    "No compatible IPCC 2013 GWP 100a impact method was found."
                )
            if raw_gwp is not None and mass_factor is None:
                messages.append(
                    "The openLCA process reference unit is "
                    f"'{reference_unit or 'unknown'}', so its GWP cannot be "
                    "reported or multiplied as kg CO2e per kg."
                )
            normalized_gwp = (
                raw_gwp / mass_factor
                if raw_gwp is not None and mass_factor is not None
                else None
            )
            self._remember_success()
            return {
                "input": material_query,
                "source": (
                    "openlca"
                    if normalized_gwp is not None
                    else "openlca_non_mass_reference"
                    if raw_gwp is not None
                    else "openlca_process_no_lcia"
                ),
                "kg_co2e_per_kg": normalized_gwp,
                "unit": "kg CO2e/kg",
                "gwp_per_reference_unit": raw_gwp,
                "reference_unit": reference_unit or None,
                "process_name": process.name,
                "product_system": product_system.name or product_system.id,
                "impact_method": impact_method.name if impact_method else None,
                "reference_exchanges": reference_rows,
                "impacts": impact_rows,
                "top_flows": flow_rows,
                "message": " ".join(messages) or None,
            }
        except (OpenLCAUnavailableError, CalculationError):
            raise
        except Exception as exc:
            self._remember_error(exc)
            raise CalculationError(f"openLCA calculation failed: {exc}") from exc
        finally:
            if result is not None:
                try:
                    if hasattr(result, "dispose"):
                        result.dispose()
                    else:
                        self.client.dispose(result)
                except Exception:
                    pass
