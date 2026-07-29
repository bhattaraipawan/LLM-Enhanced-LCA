"""Application configuration.

All settings are read once when :func:`get_settings` is first called.  The
defaults match the two original scripts while keeping destructive openLCA
operations opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_positive_int(name: str, default: int) -> int:
    value = _env_int(name, default)
    return value if value > 0 else default


@dataclass(frozen=True, slots=True)
class Settings:
    app_host: str
    app_port: int
    openlca_host: str
    openlca_port: int
    openlca_calculation_timeout_seconds: int
    recreate_product_systems: bool
    show_top_flows: bool
    bom_max_upload_bytes: int
    llama_model_id: str
    llama_allow_mps: bool
    llama_local_files_only: bool
    llama_max_new_tokens: int

    @property
    def openlca_url(self) -> str:
        return f"http://{self.openlca_host}:{self.openlca_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=_env_positive_int("APP_PORT", 8000),
        openlca_host=os.getenv("OPENLCA_HOST", "localhost"),
        openlca_port=_env_positive_int("OPENLCA_PORT", 8080),
        openlca_calculation_timeout_seconds=_env_positive_int(
            "OPENLCA_CALCULATION_TIMEOUT_SECONDS", 600
        ),
        recreate_product_systems=_env_bool("OPENLCA_RECREATE_PRODUCT_SYSTEMS", False),
        show_top_flows=_env_bool("OPENLCA_SHOW_TOP_FLOWS", True),
        bom_max_upload_bytes=_env_positive_int(
            "BOM_MAX_UPLOAD_BYTES", 25 * 1024 * 1024
        ),
        llama_model_id=os.getenv(
            "LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"
        ),
        llama_allow_mps=_env_bool("LLAMA_ALLOW_MPS", True),
        llama_local_files_only=_env_bool("LLAMA_LOCAL_FILES_ONLY", False),
        llama_max_new_tokens=_env_positive_int("LLAMA_MAX_NEW_TOKENS", 512),
    )


DEFAULT_MATERIAL_QUERY = "sand"

IMPACT_METHOD_CANDIDATES = (
    "IPCC 2013 GWP 100a",
    "IPCC 2013, climate change, GWP 100a",
    "IPCC 2013, climate change, GWP 100a, incl. climate-carbon feedbacks",
    "IPCC 2013, climate change, GWP 100a, excl. climate-carbon feedbacks",
    "IPCC 2013, climate change, GWP 100a, no LT",
    "IPCC 2013, climate change, GWP 100a, LT",
)

# BAFU:2025/openLCA lookup names used before an LLM fallback.
MATERIALS = {
    "cement": "cement, Portland",
    "ordinary portland cement": "cement, Portland",
    "portland cement": "cement, Portland",
    "natural gravel": "gravel",
    "gravel": "gravel",
    "sand": "sand",
    "river sand": "sand",
    "stone": "crushed stone",
    "soil": "excavated soil",
    "local wood": "sawnwood",
    "wood": "sawnwood",
    "19mm plywood": "plywood",
    "plywood": "plywood",
    "3mm commercial plywood": "plywood",
    "3mm commercial ply": "plywood",
    "commercial plywood": "plywood",
    "commercial ply": "plywood",
    "nail": "steel nail",
    "10mm rebar": "reinforcing steel",
    "7mm rebar": "reinforcing steel",
    "rebar": "reinforcing steel",
    "reinforcement": "reinforcing steel",
    "bindingwire": "steel wire",
    "binding wire": "steel wire",
    "bamboo": "bamboo",
    "cgi sheet": "galvanised steel sheet",
    "cgi sheets": "galvanised steel sheet",
    "24 gauge cgi sheets": "galvanised steel sheet",
    "24 gauge cgi sheet": "galvanised steel sheet",
    "c gi sheet": "galvanised steel sheet",
    "galvanized sheet": "galvanised steel sheet",
    "galvanised sheet": "galvanised steel sheet",
    "brick": "brick",
}
