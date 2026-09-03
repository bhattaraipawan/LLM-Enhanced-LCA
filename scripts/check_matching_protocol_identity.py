#!/usr/bin/env python3
"""Verify that production Qwen matching settings equal the selected four-model run.

The ELCD catalog is intentionally excluded from the identity requirement: it may
be freshly regenerated and is checked separately through ELCD_Catalog_Lock.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

import qwen_matcher as qm  # noqa: E402
from catalog_lock import verify_lock  # noqa: E402

BENCHMARK_RESULTS = ROOT / "Four_Models" / "Output" / "qwen" / "benchmark_results.xlsx"
CATALOG = ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
LOCK = ROOT / "ELCD_Check" / "ELCD_Catalog_Lock.json"


def _metadata(path: Path) -> dict[str, object]:
    df = pd.read_excel(path, sheet_name="Metadata")
    return {str(r["field"]): r["value"] for _, r in df.iterrows()}


def expected_protocol() -> dict[str, object]:
    return {
        "benchmark_script_version": qm.BENCHMARK_SCRIPT_VERSION,
        "model_id": qm.MODEL_ID,
        "model_revision": qm.MODEL_REVISION,
        "system_prompt_sha256": hashlib.sha256(qm.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "seed": qm.BENCHMARK_SEED,
        "candidate_pool_size": qm.BENCHMARK_CANDIDATE_POOL_SIZE,
        "reported_top_k": qm.BENCHMARK_TOP_K,
        "max_new_tokens": qm.BENCHMARK_MAX_NEW_TOKENS,
        "temperature": 0,
        "decoding": "greedy",
        "do_sample": False,
        "quantization": "4-bit NF4",
        "retrieval_method": "character n-gram TF-IDF",
        "retrieval_analyzer": "char_wb",
        "retrieval_ngram_range": "3-5",
        "retrieval_query_source": "original BOM description only",
        "candidate_presentation": "deterministic SHA-256 shuffle by sample_id + process_uuid",
        "retrieval_rank_visible_to_llm": False,
        "retrieval_score_visible_to_llm": False,
    }


def main() -> int:
    meta = _metadata(BENCHMARK_RESULTS)
    prompt_hash = hashlib.sha256(qm.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    checks = {
        "model_id": (qm.MODEL_ID, str(meta.get("model_id") or "")),
        "model_revision": (qm.MODEL_REVISION, str(meta.get("model_revision") or "")),
        "seed": (qm.BENCHMARK_SEED, int(meta.get("base_seed"))),
        "candidate_pool_size": (qm.BENCHMARK_CANDIDATE_POOL_SIZE, int(meta.get("candidate_pool_size"))),
        "reported_top_k": (qm.BENCHMARK_TOP_K, int(meta.get("reported_top_k"))),
        "max_new_tokens": (qm.BENCHMARK_MAX_NEW_TOKENS, int(meta.get("max_new_tokens"))),
        "temperature": (0.0, float(meta.get("temperature"))),
        "decoding": ("greedy", str(meta.get("decoding") or "")),
        "quantization": ("4-bit NF4", str(meta.get("quantization") or "")),
        "system_prompt_sha256": (prompt_hash, str(meta.get("system_prompt_sha256") or "")),
    }
    failures = [f"{k}: production={a!r}, benchmark={b!r}" for k,(a,b) in checks.items() if a != b]
    lock = verify_lock(CATALOG, LOCK, expected_matching_protocol=expected_protocol())

    print("Four-model Qwen protocol identity check")
    for key, (a, b) in checks.items():
        print(f"  {key:28s} {'PASS' if a == b else 'FAIL'}")
    print(f"  frozen ELCD catalog lock      PASS ({lock['catalog_content_sha256']})")
    print(
        "  historical catalog identity   "
        + ("YES" if lock.get("same_as_historical_benchmark_catalog") else "NO — allowed by final design")
    )
    if failures:
        print("FAIL")
        for x in failures:
            print(" -", x)
        return 1
    print("PASS — every non-catalog Qwen matching setting matches the selected four-model run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
