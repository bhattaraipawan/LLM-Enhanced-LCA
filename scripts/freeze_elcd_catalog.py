#!/usr/bin/env python3
"""Freeze an already-exported ELCD catalog with the four-model matching protocol."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from catalog_lock import build_lock  # noqa: E402
import qwen_matcher as qm  # noqa: E402

DEFAULT_CATALOG = ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
DEFAULT_LOCK = ROOT / "ELCD_Check" / "ELCD_Catalog_Lock.json"


def matching_protocol() -> dict:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--database-label", default="ELCD 3.2")
    args = ap.parse_args()
    lock = build_lock(
        args.catalog,
        args.lock,
        database_label=args.database_label,
        matching_protocol=matching_protocol(),
        historical_benchmark_catalog_hash=qm.HISTORICAL_BENCHMARK_CATALOG_CONTENT_SHA256,
    )
    print(f"Frozen catalog: {args.catalog}")
    print(f"Process count:  {lock['catalog_process_count']}")
    print(f"Semantic SHA:   {lock['catalog_content_sha256']}")
    print(f"Lock file:      {args.lock}")
    print(
        "Historical benchmark catalog: "
        + ("IDENTICAL" if lock["same_as_historical_benchmark_catalog"] else "DIFFERENT (allowed; protocol remains locked)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
