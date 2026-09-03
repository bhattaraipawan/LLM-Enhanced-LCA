"""Offline regression checks for bounded production retrieval and material-agnostic fallbacks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from evidence_consensus import robust_model_ensemble_consensus
from evidence_cache import RuntimeEvidenceCache
from external_ef_resolver import ExternalEFResolver
from property_resolver import WebPropertyResolver
from guardrails import emission_factor_cap, property_range


def test_material_agnostic_consensus():
    r = robust_model_ensemble_consensus([0.3, 0.4, 0.5, 100.0, 300.0])
    assert r.accepted
    assert r.retained_count >= 3
    assert 100.0 not in [ [0.3,0.4,0.5,100.0,300.0][i] for i in r.retained_indices ]
    assert emission_factor_cap("any material", "kg") is None
    assert property_range("any material", "density_kg_m3") is None


def test_runtime_cache_is_empty_then_retrieved():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cache.json"
        cache = RuntimeEvidenceCache(path)
        assert cache.count() == 0
        rec = {
            "verification": "EXTERNAL_VERIFIED",
            "verification_tier": "RELAXED",
            "source_class": "EXTERNAL_VERIFIED_RELAXED",
            "ef_value": 1.234,
            "reference_unit": "kg",
            "source_url": "https://example.org/retrieved-evidence",
            "boundary": "A1-A3",
        }
        cache.put(category="emission_factor", material="test product", target_geography="Nepal", resolved=rec)
        got = cache.get(category="emission_factor", material="test product", target_geography="Nepal")
        assert got and got["ef_value"] == 1.234
        assert cache.get(category="emission_factor", material="test product", target_geography="India") is None


def test_bounded_candidate_collection():
    resolver = ExternalEFResolver(None, class3_source_budget=3, class4_total_source_budget=5, max_search_queries_per_material=6)
    resolver._search = lambda q: [
        {"title": f"Source {i}", "url": f"https://example{i}.org/doc", "snippet": "A1-A3 GWP declared unit"}
        for i in range(20)
    ]
    resolver._extract = lambda url: "A1-A3 GWP declared unit"
    candidates, rows, used = resolver._collect_evidence(
        ["q1", "q2", "q3"], "GLOBAL", "X", "test material",
        max_candidates=5, max_queries=2, seen_urls=set(),
    )
    assert len(candidates) <= 5
    assert len(rows) <= 5
    assert used <= 2

    prop = WebPropertyResolver(None, class3_source_budget=3, class4_total_source_budget=5, max_search_queries_per_property=5)
    prop._search = lambda q: [
        {"title": f"Source {i}", "url": f"https://property{i}.org/doc", "snippet": "density kg/m3"}
        for i in range(20)
    ]
    prop._extract = lambda url: "density kg/m3"
    candidates, rows, used = prop._collect_evidence(
        ["q1", "q2", "q3"], "GLOBAL", "X", "test material", "density_kg_m3",
        max_candidates=5, max_queries=2, seen_urls=set(),
    )
    assert len(candidates) <= 5
    assert len(rows) <= 5
    assert used <= 2


if __name__ == "__main__":
    test_material_agnostic_consensus()
    test_runtime_cache_is_empty_then_retrieved()
    test_bounded_candidate_collection()
    print("Bounded-production regression tests: PASS")
