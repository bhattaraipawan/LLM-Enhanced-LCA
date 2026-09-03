#!/usr/bin/env python3
"""Prove that production and benchmark matching algorithms behave identically on the active frozen catalog.

Unlike historical-output comparison, this check remains valid when the final ELCD
catalog is newly regenerated. It compares both implementations on the SAME active
catalog and same 35 benchmark inputs: retrieval, candidate presentation, exact
user prompt, JSON extraction, and schema/candidate validation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

import qwen_matcher as qm  # noqa: E402

spec = importlib.util.spec_from_file_location("benchmark_four_llms_impl", ROOT / "scripts" / "benchmark_four_llms.py")
bm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bm
assert spec.loader is not None
spec.loader.exec_module(bm)

CATALOG = ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
INPUT = ROOT / "Four_Models" / "Input" / "LLM_Model_Evaluation_Reference_Set.xlsx"
HISTORICAL_QWEN = ROOT / "Four_Models" / "Output" / "qwen" / "benchmark_results.xlsx"


def _text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _core_prediction(d: dict) -> dict:
    keys = [
        "parse_status", "structured_output_valid", "usable_response",
        "normalized_material", "match_type", "selected_process_uuid",
        "selected_process_name", "ranked_process_uuids", "ranked_process_names",
        "ranking_field_valid", "selection_field_valid", "match_type_field_valid",
        "normalization_field_valid", "field_recovery_used",
    ]
    return {k: d.get(k) for k in keys}


def main() -> int:
    # Production implementation
    prod = qm.QwenMatcher(CATALOG)

    # Benchmark implementation, pointed at the SAME active final catalog.
    catalog_df = bm.load_catalog(CATALOG)
    retriever = bm.build_catalog_retriever(catalog_df)
    inputs = pd.read_excel(INPUT, sheet_name="Reference_Set")
    historical = pd.read_excel(HISTORICAL_QWEN, sheet_name="Predictions")
    hist_by_id = {str(r["sample_id"]): r for _, r in historical.iterrows()}

    retrieval_ok = presentation_ok = prompt_ok = parser_ok = 0
    failures = []

    for _, row in inputs.iterrows():
        sid = str(row["sample_id"])
        material = str(row["material_description"])

        b_candidates = bm.retrieve_candidate_pool(row, catalog_df, retriever, qm.BENCHMARK_CANDIDATE_POOL_SIZE)
        p_candidates = prod.retriever.retrieve(material)
        b_ids = [str(c["process_uuid"]).strip().lower() for c in b_candidates]
        p_ids = [str(c["process_uuid"]).strip().lower() for c in p_candidates]
        same_retrieval = b_ids == p_ids
        retrieval_ok += int(same_retrieval)

        b_presented = bm.present_candidate_pool(row, b_candidates)
        p_presented = qm.deterministic_present(p_candidates, sid)
        bp_ids = [str(c["process_uuid"]).strip().lower() for c in b_presented]
        pp_ids = [str(c["process_uuid"]).strip().lower() for c in p_presented]
        same_presentation = bp_ids == pp_ids
        presentation_ok += int(same_presentation)

        b_prompt = bm.build_user_prompt(row, b_presented, qm.BENCHMARK_TOP_K)
        p_prompt = qm.build_benchmark_user_prompt(
            sid,
            material,
            row.get("quantity", ""),
            row.get("unit", ""),
            p_presented,
        )
        same_prompt = b_prompt == p_prompt
        prompt_ok += int(same_prompt)

        # Parser equivalence: use the historical raw text merely as a test string,
        # but validate it against the CURRENT final candidate pool in both implementations.
        raw = _text(hist_by_id[sid].get("raw_model_output"))
        b_obj, b_status = bm.extract_json(raw)
        p_obj, p_status = qm.extract_json(raw)
        b_pred = bm.validate_prediction(b_obj, b_presented, b_status, qm.BENCHMARK_TOP_K)
        p_pred = qm.validate_prediction(p_obj, p_presented, p_status, qm.BENCHMARK_TOP_K)
        same_parser = _core_prediction(b_pred) == _core_prediction(p_pred)
        parser_ok += int(same_parser)

        if not all([same_retrieval, same_presentation, same_prompt, same_parser]):
            failures.append(
                f"{sid}: retrieval={same_retrieval}, presentation={same_presentation}, "
                f"prompt={same_prompt}, parser={same_parser}"
            )

    n = len(inputs)
    print("Benchmark-vs-production algorithm identity on CURRENT frozen catalog")
    print(f"  Retrieval:             {retrieval_ok}/{n}")
    print(f"  Candidate presentation:{presentation_ok}/{n}")
    print(f"  User prompt:           {prompt_ok}/{n}")
    print(f"  Parser/validator:      {parser_ok}/{n}")
    if failures:
        print("FAIL")
        for x in failures:
            print(" -", x)
        return 1
    print("PASS — production and benchmark matching implementations are identical on the active catalog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
