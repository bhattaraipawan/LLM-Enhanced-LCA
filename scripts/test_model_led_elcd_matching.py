"""Offline checks for the model-led ELCD matching stage.

No model download, web access, or openLCA server is required.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

import qwen_matcher as qm


def main():
    # The production system prompt is intentionally the benchmark prompt.
    benchmark_text = (ROOT / "scripts" / "benchmark_four_llms.py").read_text(encoding="utf-8")
    start = benchmark_text.index('SYSTEM_PROMPT = """') + len('SYSTEM_PROMPT = """')
    end = benchmark_text.index('"""', start)
    assert qm.SYSTEM_PROMPT == benchmark_text[start:end]

    # Raw Qwen labels are preserved, while a separate production-use gate may
    # veto an obviously incompatible product family without selecting a replacement.
    status, reason = qm.production_safety_gate(
        "Plain Cement Concrete", "Plain Cement Concrete", "proxy", "Portland cement production"
    )
    assert status == "VETO"
    assert "plain_concrete_requires_concrete_process" in str(reason)

    status, reason = qm.production_safety_gate(
        "CGI Sheet", "CGI Sheet", "direct", "corrugated board sheets; mixed technology"
    )
    assert status == "VETO"
    assert "cgi_requires_galvanized_flat_steel" in str(reason)

    # Candidate-pool/schema integrity remains strict.
    supplied = {"uuid-a", "uuid-b"}
    good = json.dumps({
        "normalized_material": "Example material",
        "ranked_process_uuids": ["uuid-a"],
        "match_type": "proxy",
    })
    parsed, stat = qm.parse_output(good, supplied)
    assert stat == "ok" and parsed["ranked_process_uuids"] == ["uuid-a"]

    out_of_pool = json.dumps({
        "normalized_material": "Example material",
        "ranked_process_uuids": ["uuid-x"],
        "match_type": "proxy",
    })
    parsed, stat = qm.parse_output(out_of_pool, supplied)
    assert parsed is None and stat == "invalid_ranking"

    contradictory = json.dumps({
        "normalized_material": "Example material",
        "ranked_process_uuids": ["uuid-a"],
        "match_type": "review_required",
    })
    parsed, stat = qm.parse_output(contradictory, supplied)
    assert parsed is None and stat == "invalid_ranking"

    print("Model-led ELCD matching checks: PASS")


if __name__ == "__main__":
    main()
