"""Regression test for formal case-study ID / candidate-presentation alignment.

The frozen benchmark uses S01..., B01..., and A01... as the deterministic
candidate-presentation seed. Legacy BOM workbooks may contain numeric IDs. This
test guarantees production normalizes those numeric IDs before Qwen matching and
therefore recreates the frozen benchmark presentation order when the Top-5 pool
is the same.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "production") not in sys.path:
    sys.path.insert(0, str(ROOT / "production"))

from qwen_matcher import align_formal_case_study_ids, deterministic_present

BENCHMARK = ROOT / "Four_Models" / "Output" / "qwen" / "benchmark_results.xlsx"


def _presentation(candidates, item_id):
    return [r["process_uuid"] for r in deterministic_present(candidates, item_id)]


def main():
    # Basic legacy-ID normalization checks.
    df = pd.DataFrame({"ID": [1, 2, 14], "Material": ["x", "y", "z"], "Quantity": [1, 1, 1], "Unit": ["kg"] * 3})
    aligned, info = align_formal_case_study_ids(df, "20260830_031056_BOM_Bamboo")
    assert aligned["ID"].tolist() == ["B01", "B02", "B14"]
    assert info["changed"] == 3

    df = pd.DataFrame({"ID": ["S1", "S02", 12], "Material": ["x", "y", "z"], "Quantity": [1, 1, 1], "Unit": ["kg"] * 3})
    aligned, _ = align_formal_case_study_ids(df, "BOM_Stonecrete.xlsx")
    assert aligned["ID"].tolist() == ["S01", "S02", "S12"]

    df = pd.DataFrame({"ID": [1, 9], "Material": ["x", "y"], "Quantity": [1, 1], "Unit": ["kg", "kg"]})
    aligned, _ = align_formal_case_study_ids(df, "BOM_Attic.xlsx")
    assert aligned["ID"].tolist() == ["A01", "A09"]

    # Arbitrary/non-formal BOMs are untouched.
    df = pd.DataFrame({"ID": [1, 2], "Material": ["x", "y"], "Quantity": [1, 1], "Unit": ["kg", "kg"]})
    unchanged, info = align_formal_case_study_ids(df, "client_project.xlsx")
    assert unchanged["ID"].tolist() == [1, 2]
    assert info["prefix"] is None

    # Strong end-to-end reproducibility check against the frozen Qwen workbook.
    if not BENCHMARK.exists():
        raise AssertionError(f"Frozen Qwen benchmark workbook not found: {BENCHMARK}")
    pred = pd.read_excel(BENCHMARK, sheet_name="Predictions")
    checked = 0
    for _, row in pred.iterrows():
        sid = str(row["sample_id"]).strip()
        if not sid or sid[0] not in {"S", "B", "A"}:
            continue
        n = int(sid[1:])
        source = {"S": "BOM_Stonecrete.xlsx", "B": "BOM_Bamboo.xlsx", "A": "BOM_Attic.xlsx"}[sid[0]]
        legacy = pd.DataFrame({"ID": [n], "Material": [row["material_description"]], "Quantity": [row["quantity"]], "Unit": [row["unit"]]})
        aligned, _ = align_formal_case_study_ids(legacy, source)
        assert aligned.iloc[0]["ID"] == sid, (sid, aligned.iloc[0]["ID"])

        pool = json.loads(row["candidate_pool_uuids"])
        candidates = [{"process_uuid": u} for u in pool]
        actual = _presentation(candidates, aligned.iloc[0]["ID"])
        expected = json.loads(row["presented_candidate_uuids"])
        assert actual == expected, f"presentation mismatch for {sid}: {actual} != {expected}"
        checked += 1

    assert checked == 35, f"Expected 35 frozen benchmark rows, checked {checked}"
    print(f"PASS: formal IDs and benchmark candidate presentation align for all {checked} rows.")


if __name__ == "__main__":
    main()
