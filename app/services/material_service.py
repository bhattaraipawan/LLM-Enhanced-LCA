"""Material resolution and calculation orchestration."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.config import MATERIALS
from app.core.exceptions import CalculationError, LlamaUnavailableError, OpenLCAUnavailableError
from app.services.openlca_service import OpenLCAService
from app.utils.json_helpers import parse_json_object
from app.utils.text import normalize_process_name, search_tokens


@dataclass(frozen=True, slots=True)
class ProcessResolution:
    process_name: str
    message: str | None = None


def _join_messages(*messages: str | None) -> str | None:
    unique: list[str] = []
    for message in messages:
        cleaned = str(message or "").strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return " ".join(unique) or None


class MaterialService:
    def __init__(self, openlca: OpenLCAService, llama: Any) -> None:
        self.openlca = openlca
        self.llama = llama

    def _ask_llama(
        self, prompt: str, *, max_new_tokens: int = 128
    ) -> tuple[dict[str, Any], str | None]:
        try:
            generated = self.llama.generate(
                prompt=prompt, max_new_tokens=max_new_tokens
            )
        except LlamaUnavailableError as exc:
            return {}, str(exc)
        except Exception as exc:  # the engine should normalize failures; keep UI safe
            return {}, f"Llama is not available: {exc}"

        if not getattr(generated, "available", True):
            return {}, (
                getattr(generated, "message", None)
                or "Llama is not available on this device."
            )
        parsed = parse_json_object(getattr(generated, "raw_output", ""))
        numeric_result = getattr(generated, "result", {}) or {}
        if isinstance(numeric_result, dict):
            parsed.update(numeric_result)
        return parsed, None

    @staticmethod
    def direct_material_match(material_query: str) -> tuple[str | None, str | None]:
        query = normalize_process_name(material_query)
        for key, process_name in sorted(
            MATERIALS.items(), key=lambda item: len(item[0]), reverse=True
        ):
            normalized_key = normalize_process_name(key)
            if query == normalized_key or re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_key)}(?![a-z0-9])", query
            ):
                return key, process_name
        if "steel" in query or "reinforcement" in query:
            return "rebar", MATERIALS["rebar"]
        if "block" in query or "masonry" in query:
            return "brick", MATERIALS["brick"]
        return None, None

    @staticmethod
    def process_search_terms(material_query: str) -> set[str]:
        query = normalize_process_name(material_query)
        terms = set(search_tokens(query))
        synonym_groups = {
            "gravel": ("gravel", "round"),
            "natural gravel": ("gravel", "round"),
            "stone": ("gravel", "crushed", "stone"),
            "sand": ("sand",),
            "soil": ("soil", "earth", "excavation"),
            "wood": ("sawnwood", "softwood", "hardwood", "wood"),
            "local wood": ("sawnwood", "softwood", "wood"),
            "ply": ("plywood",),
            "plywood": ("plywood",),
            "rebar": ("reinforcing", "steel"),
            "reinforcement": ("reinforcing", "steel"),
            "bindingwire": ("steel", "wire"),
            "binding wire": ("steel", "wire"),
            "bamboo": ("bamboo",),
            "cgi": ("steel", "sheet", "hot", "rolled"),
            "galvanized": ("steel", "sheet"),
            "galvanised": ("steel", "sheet"),
            "nail": ("nail", "steel"),
            "cement": ("cement", "portland"),
        }
        for key, values in synonym_groups.items():
            if key in query:
                terms.update(values)
        return terms

    @staticmethod
    def process_match_adjustment(
        material_query: str,
        process_name: str,
        candidate: str,
        candidate_tokens: set[str],
    ) -> int:
        query = normalize_process_name(material_query or process_name)
        target = normalize_process_name(process_name)
        score = 0
        if "soil" in query or target == "excavated soil":
            bad_terms = {
                "anchor",
                "concrete",
                "nailing",
                "notfall",
                "reinforced",
                "shotcrete",
                "sprayed",
                "waidspital",
            }
            if bad_terms.intersection(candidate_tokens):
                return -500
            good_terms = {"earth", "excavated", "excavation", "topsoil", "soil"}
            score += sum(18 for term in good_terms if term in candidate_tokens)
            if "excavat" in candidate:
                score += 30
        if "bamboo" in query and "flooring" in candidate_tokens:
            score -= 30
        if ("cgi" in query or "galvan" in query) and "steel" not in candidate_tokens:
            score -= 40
        return score

    def process_candidates_for_material(
        self, material_query: str, limit: int = 20
    ) -> list[str]:
        terms = self.process_search_terms(material_query)
        if not terms:
            return []
        try:
            schema = self.openlca.schema
            descriptors = self.openlca.descriptors(schema.Process)
        except OpenLCAUnavailableError:
            return []

        scored: list[tuple[int, str]] = []
        query = normalize_process_name(material_query)
        for descriptor in descriptors:
            name = getattr(descriptor, "name", None)
            if not name:
                continue
            candidate = normalize_process_name(name)
            candidate_tokens = set(search_tokens(name))
            score = sum(10 for term in terms if term in candidate_tokens)
            score += sum(4 for term in terms if term in candidate)
            if "market for" in candidate:
                score += 1
            if "bafu" in candidate or "ch" in candidate_tokens or "switzerland" in candidate:
                score += 6
            if "construction" in candidate or "building" in candidate:
                score += 4
            if "cutoff" in candidate or "apos" in candidate:
                score += 2
            if "production" in candidate and "market for" not in candidate:
                score += 2
            if query and query in candidate:
                score += 40
            score += self.process_match_adjustment(
                material_query, material_query, candidate, candidate_tokens
            )
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [name for _score, name in scored[:limit]]

    def resolve_process_name(self, material_query: str) -> ProcessResolution:
        _key, direct_name = self.direct_material_match(material_query)
        if direct_name is not None:
            return ProcessResolution(direct_name)

        candidates = self.process_candidates_for_material(material_query)
        if candidates:
            candidate_text = "\n".join(
                f"{index}: {name}" for index, name in enumerate(candidates)
            )
            data, message = self._ask_llama(
                f"""
Choose the best openLCA/BAFU process for this construction material.

Material name: {material_query}

Available openLCA process candidates:
{candidate_text}

Return ONLY JSON like {{"process_index": 0}}.
Use -1 only if none are usable.
""",
                max_new_tokens=64,
            )
            try:
                selected_index = int(data.get("process_index"))
            except (TypeError, ValueError):
                selected_index = 0
            if not 0 <= selected_index < len(candidates):
                selected_index = 0
            return ProcessResolution(candidates[selected_index], message)

        data, message = self._ask_llama(
            f"""
Suggest the best exact openLCA process name for this construction material:
{material_query}

Return ONLY JSON like {{"process_name": "exact process name"}}.
""",
            max_new_tokens=128,
        )
        suggested = data.get("process_name")
        if isinstance(suggested, str) and suggested.strip():
            return ProcessResolution(suggested.strip(), message)
        return ProcessResolution(material_query, message)

    def find_process(self, process_name: str, material_query: str) -> Any | None:
        try:
            schema = self.openlca.schema
        except OpenLCAUnavailableError:
            return None
        process = self.openlca.find(schema.Process, process_name)
        if process is not None:
            return process

        all_processes = self.openlca.descriptors(schema.Process)
        target = normalize_process_name(process_name)
        target_tokens = search_tokens(process_name)
        best_match = None
        best_score = 0
        for descriptor in all_processes:
            name = getattr(descriptor, "name", None)
            if not name:
                continue
            candidate = normalize_process_name(name)
            if candidate == target:
                return descriptor
            score = 0
            if target and target in candidate:
                score += 100
            if candidate.startswith(target):
                score += 50
            candidate_tokens = set(search_tokens(name))
            score += sum(10 for token in target_tokens if token in candidate_tokens)
            score += sum(4 for token in target_tokens if token in candidate)
            if "market for" in candidate:
                score += 1
            if "bafu" in candidate or "ch" in candidate_tokens or "switzerland" in candidate:
                score += 6
            if "construction" in candidate or "building" in candidate:
                score += 4
            if "production" in candidate:
                score += 3
            if "cutoff" in candidate or "apos" in candidate:
                score += 2
            score += self.process_match_adjustment(
                material_query, process_name, candidate, candidate_tokens
            )
            if score > best_score:
                best_score = score
                best_match = descriptor
        minimum_score = 12 if len(target_tokens) <= 1 else max(18, len(target_tokens) * 7)
        return best_match if best_match is not None and best_score >= minimum_score else None

    @staticmethod
    def fallback_kg_co2e_per_kg(
        material_query: str, process_name: str | None = None
    ) -> float | None:
        text = f"{material_query} {process_name or ''}".lower()
        if "soil" in text or "earth" in text:
            return 0.005
        return None

    def estimate_gwp(
        self, material_query: str, process_name: str
    ) -> tuple[float | None, str | None, bool]:
        fallback = self.fallback_kg_co2e_per_kg(material_query, process_name)
        data, message = self._ask_llama(
            f"""
Estimate a typical cradle-to-gate global warming potential for this material.

Material: {material_query}
Unavailable openLCA process: {process_name}

Return ONLY JSON like {{"kg_co2e_per_kg": 0.25}}.
The value must be numeric, positive, and in kg CO2e per kg.
""",
            max_new_tokens=256,
        )
        try:
            value = float(data.get("kg_co2e_per_kg"))
        except (TypeError, ValueError):
            return fallback, message, False
        if not math.isfinite(value) or value <= 0:
            return fallback, message, False
        return value, message, True

    def calculate_material(self, material_query: str) -> dict[str, Any]:
        resolution = self.resolve_process_name(material_query)
        process = self.find_process(resolution.process_name, material_query)
        openlca_message: str | None = None

        if process is not None:
            try:
                result = self.openlca.calculate(process, material_query)
                result["message"] = _join_messages(
                    resolution.message, result.get("message")
                )
                return result
            except (OpenLCAUnavailableError, CalculationError) as exc:
                openlca_message = str(exc)
        else:
            capability = self.openlca.capability()
            if capability.get("status") == "unavailable":
                openlca_message = capability.get("message")

        value, llama_message, used_llama = self.estimate_gwp(
            material_query, resolution.process_name
        )
        if value is not None and used_llama:
            source = "llm_knowledge"
        elif value is not None:
            source = "deterministic_fallback"
        else:
            source = "unavailable"
        return {
            "input": material_query,
            "source": source,
            "kg_co2e_per_kg": value,
            "unit": "kg CO2e/kg",
            "process_name": resolution.process_name,
            "product_system": None,
            "impact_method": None,
            "reference_exchanges": [],
            "impacts": [],
            "top_flows": [],
            "message": _join_messages(
                resolution.message, openlca_message, llama_message
            ),
        }
