"""Offline regression checks for the final indicator and identity guardrails.

Run from the repository root:
    python scripts/test_guardrails.py

No network, model download, or environmental-factor table is required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

import external_ef_resolver as efr
from external_ef_resolver import ExternalEFResolver


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def test_indicator_component_rejection():
    material = "24 Gauge CGI Sheets"
    _assert(not efr._gwp_indicator_ok("Climate change (GWP) - biogenic", material),
            "biogenic GWP component must not pass ordinary GWP-total")
    _assert(not efr._gwp_indicator_ok("Climate change (GWP) - fossil", material),
            "fossil GWP component must not pass ordinary GWP-total")
    _assert(not efr._gwp_indicator_ok("Climate change (GWP) - LULUC", material),
            "LULUC GWP component must not pass ordinary GWP-total")
    _assert(efr._gwp_indicator_ok("Climate change (GWP) - total", material),
            "explicit total GWP should pass ordinary material indicator gate")


def test_terminal_interpretation_first_identity_gate():
    material = "24 Gauge CGI Sheets"
    bad, bad_reason = ExternalEFResolver._model_product_identity_compatible(
        material,
        "Composite Glass Insulation Sheets",
        "Composite insulation sheet used in building envelope applications.",
        "",
    )
    _assert(not bad, "wrong CGI acronym expansion must be rejected")

    good, good_reason = ExternalEFResolver._model_product_identity_compatible(
        material,
        "Corrugated galvanized iron roofing sheet made from zinc-coated steel sheet",
        "Ferrous zinc-coated corrugated roofing product.",
        "",
    )
    _assert(good, f"explicit galvanized-steel CGI interpretation should pass: {good_reason}")


def test_cache_policy_versioning():
    import evidence_cache
    path = ROOT / "runtime" / "_test_evidence_cache.json"
    try:
        old_payload = {
            "cache_version": "obsolete-policy",
            "records": {"example": {"resolved": {"value": 1}}},
        }
        path.write_text(json.dumps(old_payload), encoding="utf-8")
        cache = evidence_cache.RuntimeEvidenceCache(path)
        _assert(cache.count() == 0, "old-policy cache must not be reused")
    finally:
        path.unlink(missing_ok=True)


def main():
    tests = [
        test_indicator_component_rejection,
        test_terminal_interpretation_first_identity_gate,
        test_cache_policy_versioning,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All final guardrail regression checks passed.")


if __name__ == "__main__":
    main()
