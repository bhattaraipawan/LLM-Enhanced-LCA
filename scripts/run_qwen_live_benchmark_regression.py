#!/usr/bin/env python3
"""Run the production Qwen matcher on all 35 frozen benchmark inputs.

Use this in the same Colab T4 environment used for production before freezing a
paper run. It compares only the RAW Qwen benchmark fields; downstream safety,
unit conversion, External Verified, and fallback logic are intentionally ignored.

If the freshly frozen ELCD catalog is semantically identical to the historical
benchmark catalog, the target is 35/35 raw labels and UUID/review outcomes. If
the fresh catalog differs, historical output equality is only a diagnostic; the
script still confirms that the same 35 inputs can be processed under the locked
Qwen protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from qwen_matcher import QwenMatcher  # noqa: E402

CATALOG = ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
RESULTS = ROOT / "Four_Models" / "Output" / "qwen" / "benchmark_results.xlsx"


def _text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--benchmark-results", type=Path, default=RESULTS)
    args = parser.parse_args()

    expected = pd.read_excel(args.benchmark_results, sheet_name="Predictions")
    matcher = QwenMatcher(args.catalog).load_model()
    same_catalog = bool(matcher.catalog_lock.get("same_as_historical_benchmark_catalog"))

    rows = []
    for i, exp in expected.iterrows():
        item = {
            "ID": _text(exp["sample_id"]),
            "Material": _text(exp["material_description"]),
            "Quantity": exp.get("quantity", ""),
            "Unit": exp.get("unit", ""),
        }
        got = matcher.match_item(item)
        exp_uuid = _text(exp.get("selected_process_uuid"))
        got_uuid = _text(got.selected_process_uuid)
        rec = {
            "sample_id": item["ID"],
            "label_expected": _text(exp.get("match_type")),
            "label_got": _text(got.match_type),
            "uuid_expected": exp_uuid,
            "uuid_got": got_uuid,
            "parse_expected": _text(exp.get("parse_status")),
            "parse_got": _text(got.parse_status),
            "label_same": _text(exp.get("match_type")) == _text(got.match_type),
            "uuid_same": exp_uuid == got_uuid,
            "parse_same": _text(exp.get("parse_status")) == _text(got.parse_status),
            "raw_model_output": got.raw_model_output,
        }
        rows.append(rec)
        print(
            f"[{i+1:02d}/35] {item['ID']} | label={rec['label_same']} "
            f"uuid={rec['uuid_same']} parse={rec['parse_same']}"
        )

    df = pd.DataFrame(rows)
    label_n = int(df["label_same"].sum())
    uuid_n = int(df["uuid_same"].sum())
    parse_n = int(df["parse_same"].sum())
    print("\nLive Qwen benchmark regression")
    print(f"  Raw Direct/Proxy/RR labels: {label_n}/35")
    print(f"  Selected UUID/RR outcome:   {uuid_n}/35")
    print(f"  Parse status:               {parse_n}/35")
    if same_catalog:
        if label_n == uuid_n == parse_n == 35:
            print("PASS — identical catalog + locked runtime reproduced the historical Qwen run 35/35.")
            return 0
        print("REVIEW — identical catalog but live Qwen differed from the historical run.")
        return 1
    print("INFO — the final ELCD catalog differs from the historical benchmark catalog, so 35/35 historical output equality is not required.")
    print("PASS — live Qwen completed the 35-row protocol check under the freshly frozen catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
