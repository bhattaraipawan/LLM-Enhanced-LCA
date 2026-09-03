from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "production"
sys.path.insert(0, str(PROD))

from external_ef_resolver import ExternalEFResolver


def main() -> None:
    src = (PROD / "external_ef_resolver.py").read_text(encoding="utf-8")

    assert "DYNAMIC_FROZEN_ELCD_SAME_FAMILY_PROXY" in src
    assert "DYNAMIC_FROZEN_ELCD_SEMANTIC_ANALOG_PROXY" in src
    assert "repeated_value_consensus_used\": False" in src
    assert "EXACT_MATERIAL_UNIT_TRIANGULATED_LLM_CONSENSUS" not in src
    assert "DYNAMIC_ANALOG_UNIT_TRIANGULATED_CONSENSUS" not in src

    ok, reason = ExternalEFResolver._model_product_identity_compatible(
        "CGI Sheet",
        "corrugated galvanized steel roofing sheet",
        rationale="zinc-coated steel sheet used as corrugated roofing",
        basis="galvanized flat steel product",
    )
    assert ok, reason

    bad, bad_reason = ExternalEFResolver._model_product_identity_compatible(
        "CGI Sheet",
        "generic construction sheet",
        rationale="cement production and plastic manufacturing dominate the product",
        basis="cement and plastic composite",
    )
    assert not bad, "semantic drift should be vetoed"

    print("final dynamic Class-4 tests: PASS")


if __name__ == "__main__":
    main()
