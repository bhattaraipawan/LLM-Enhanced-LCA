"""Run the final four-model benchmark efficiently in Google Colab.

Final study design
------------------
Main benchmark:
    35 materials x 4 models x 1 run = 140 responses.

Repeatability check:
    12 stratified materials x 4 models x 1 additional run = 48 responses.

Total formal model responses = 188.

Each model is loaded once. Its 35-item main benchmark is run first, followed by
its 12-item second-pass repeatability check before the model process exits.
This avoids loading each checkpoint twice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_benchmark_reference.py"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_four_llms.py"
REFERENCE_PATH = REPO_ROOT / "Four_Models" / "Input" / "LLM_Model_Evaluation_Reference_Set.xlsx"
REPEATABILITY_PATH = REPO_ROOT / "Four_Models" / "Input" / "Repeatability_Subset.xlsx"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Four_Models" / "Output"
MODEL_ORDER = ["llama", "qwen", "deepseek", "mistral"]
EXPECTED_BENCHMARK_SCRIPT_VERSION = "2.3.0"

# Fixed, balanced subset: 4 Direct + 4 Proxy + 4 Review Required.
REPEATABILITY_IDS = [
    "A01", "A03", "B10", "S01",  # Direct
    "A09", "B06", "B11", "B12",  # Proxy
    "A04", "A08", "B05", "S11",  # Review Required
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final 188-response four-LLM benchmark."
    )
    parser.add_argument("--candidate-pool-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use the already-frozen benchmark reference workbook.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a model only when both its main and repeatability workbooks already exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only 2 materials x 1 run per model. Repeatability pass is skipped.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def create_repeatability_subset() -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Frozen reference workbook not found: {REFERENCE_PATH}. "
            "Run prepare_benchmark_reference.py first."
        )

    reference = pd.read_excel(REFERENCE_PATH, sheet_name="Reference_Set")
    reference["sample_id"] = reference["sample_id"].astype(str).str.strip()
    subset = reference[reference["sample_id"].isin(REPEATABILITY_IDS)].copy()

    missing = sorted(set(REPEATABILITY_IDS) - set(subset["sample_id"]))
    if missing:
        raise ValueError(f"Repeatability subset IDs missing from reference set: {missing}")

    subset["_order"] = subset["sample_id"].map(
        {sid: i for i, sid in enumerate(REPEATABILITY_IDS)}
    )
    subset = subset.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    counts = subset["ground_truth_match_type"].value_counts().to_dict()
    expected = {"Direct": 4, "Proxy": 4, "Review Required": 4}
    if any(counts.get(label, 0) != n for label, n in expected.items()):
        raise ValueError(
            "Repeatability subset is no longer balanced as intended. "
            f"Observed counts: {counts}; expected: {expected}."
        )

    metadata = pd.DataFrame(
        [
            {"field": "purpose", "value": "One additional test-retest pass"},
            {"field": "subset_size", "value": len(subset)},
            {"field": "direct_rows", "value": 4},
            {"field": "proxy_rows", "value": 4},
            {"field": "review_required_rows", "value": 4},
            {"field": "selection_rule", "value": "Fixed balanced subset selected before final model scoring"},
        ]
    )

    REPEATABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(REPEATABILITY_PATH, engine="openpyxl") as writer:
        subset.to_excel(writer, sheet_name="Reference_Set", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
    print(f"Repeatability subset saved: {REPEATABILITY_PATH}")
    return subset


def norm_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def selection_label(row: pd.Series, suffix: str) -> str:
    match_type = str(row.get(f"match_type_{suffix}", "")).strip()
    if match_type == "review_required":
        return "__REVIEW_REQUIRED__"
    return str(row.get(f"selected_process_uuid_{suffix}", "")).strip().lower()


def safe_mean(series: pd.Series) -> float | None:
    return float(series.mean()) if len(series) else None


def agreement_given(mask: pd.Series, agreement: pd.Series) -> float | None:
    valid = agreement[mask.astype(bool)]
    return float(valid.mean()) if len(valid) else None


def workbook_matches_protocol(path: Path, args: argparse.Namespace, expected_seed: int) -> bool:
    """Only resume outputs produced by the current benchmark protocol."""
    try:
        meta = pd.read_excel(path, sheet_name="Metadata")
        values = dict(zip(meta["field"].astype(str), meta["value"]))
        checks = {
            "script_version": EXPECTED_BENCHMARK_SCRIPT_VERSION,
            "runs_per_sample": 1,
            "candidate_pool_size": args.candidate_pool_size,
            "reported_top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "base_seed": expected_seed,
            "candidate_presentation": "deterministic SHA-256 shuffle by sample_id + process_uuid",
            "retrieval_rank_visible_to_llm": False,
            "retrieval_score_visible_to_llm": False,
        }
        for key, expected in checks.items():
            actual = values.get(key)
            if str(actual).strip() != str(expected).strip():
                return False
        return True
    except Exception:
        return False


def write_repeatability_summary(output_root: Path, subset: pd.DataFrame) -> Path:
    """Write test-retest results without rewarding repeated invalid outputs."""
    summaries: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []

    for model in MODEL_ORDER:
        main_path = output_root / model / "benchmark_results.xlsx"
        repeat_path = output_root / "repeatability" / model / "benchmark_results.xlsx"
        if not main_path.exists() or not repeat_path.exists():
            raise FileNotFoundError(
                f"Missing main/repeatability workbook for {model}: {main_path} / {repeat_path}"
            )

        main = pd.read_excel(main_path, sheet_name="Predictions")
        repeat = pd.read_excel(repeat_path, sheet_name="Predictions")
        main = main[main["sample_id"].astype(str).isin(REPEATABILITY_IDS)].copy()

        keep = [
            "sample_id", "normalized_material", "selected_process_uuid", "match_type",
            "structured_output_valid", "usable_response",
            "normalization_field_valid", "selection_field_valid", "match_type_field_valid",
            "parse_status", "generation_seconds",
        ]
        m = main[keep].copy()
        r = repeat[keep].copy()
        paired = m.merge(r, on="sample_id", how="inner", suffixes=("_run1", "_run2"))
        if len(paired) != len(REPEATABILITY_IDS):
            raise ValueError(f"Expected 12 paired rows for {model}; found {len(paired)}")

        paired["normalization_pair_valid"] = (
            paired["normalization_field_valid_run1"].astype(bool)
            & paired["normalization_field_valid_run2"].astype(bool)
        )
        paired["process_pair_valid"] = (
            paired["usable_response_run1"].astype(bool)
            & paired["usable_response_run2"].astype(bool)
        )
        paired["match_type_pair_valid"] = (
            paired["match_type_field_valid_run1"].astype(bool)
            & paired["match_type_field_valid_run2"].astype(bool)
        )
        paired["all_three_pair_valid"] = paired["process_pair_valid"]

        paired["normalization_agreement"] = paired.apply(
            lambda x: norm_text(x["normalized_material_run1"]) == norm_text(x["normalized_material_run2"]), axis=1
        )
        paired["selection_agreement"] = paired.apply(
            lambda x: selection_label(x, "run1") == selection_label(x, "run2"), axis=1
        )
        paired["match_type_agreement"] = (
            paired["match_type_run1"].astype(str) == paired["match_type_run2"].astype(str)
        )
        paired["all_three_agreement"] = (
            paired["normalization_agreement"]
            & paired["selection_agreement"]
            & paired["match_type_agreement"]
        )

        paired["normalization_strict_agreement"] = paired["normalization_pair_valid"] & paired["normalization_agreement"]
        paired["selection_strict_agreement"] = paired["process_pair_valid"] & paired["selection_agreement"]
        paired["match_type_strict_agreement"] = paired["match_type_pair_valid"] & paired["match_type_agreement"]
        paired["all_three_strict_agreement"] = paired["all_three_pair_valid"] & paired["all_three_agreement"]

        paired.insert(0, "model_key", model)
        paired_frames.append(paired)

        summaries.append(
            {
                "model_key": model,
                "n_repeatability_items": len(paired),
                "normalization_valid_pair_rate": float(paired["normalization_pair_valid"].mean()),
                "normalization_agreement_given_valid": agreement_given(paired["normalization_pair_valid"], paired["normalization_agreement"]),
                "normalization_strict_agreement": float(paired["normalization_strict_agreement"].mean()),
                "process_valid_pair_rate": float(paired["process_pair_valid"].mean()),
                "process_or_review_agreement_given_valid": agreement_given(paired["process_pair_valid"], paired["selection_agreement"]),
                "process_or_review_strict_agreement": float(paired["selection_strict_agreement"].mean()),
                "match_type_valid_pair_rate": float(paired["match_type_pair_valid"].mean()),
                "match_type_agreement_given_valid": agreement_given(paired["match_type_pair_valid"], paired["match_type_agreement"]),
                "match_type_strict_agreement": float(paired["match_type_strict_agreement"].mean()),
                "all_three_valid_pair_rate": float(paired["all_three_pair_valid"].mean()),
                "all_three_agreement_given_valid": agreement_given(paired["all_three_pair_valid"], paired["all_three_agreement"]),
                "all_three_strict_agreement": float(paired["all_three_strict_agreement"].mean()),
                "run1_structured_output_valid_rate": float(paired["structured_output_valid_run1"].astype(bool).mean()),
                "run2_structured_output_valid_rate": float(paired["structured_output_valid_run2"].astype(bool).mean()),
                "run1_usable_response_rate": float(paired["usable_response_run1"].astype(bool).mean()),
                "run2_usable_response_rate": float(paired["usable_response_run2"].astype(bool).mean()),
            }
        )

    output_path = output_root / "repeatability" / "repeatability_check.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="Summary", index=False)
        pd.concat(paired_frames, ignore_index=True).to_excel(
            writer, sheet_name="PairedPredictions", index=False
        )
        subset.to_excel(writer, sheet_name="Subset", index=False)
    print(f"Repeatability comparison saved: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    requested_output_root = args.output_root.expanduser().resolve()
    # Smoke tests are written to a separate subtree so they can never be mistaken
    # for or overwrite the formal 35+12 benchmark outputs.
    output_root = requested_output_root / "smoke" if args.smoke else requested_output_root

    if not args.skip_prepare:
        run([sys.executable, str(PREPARE_SCRIPT)])

    check_cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--check-inputs",
        "--candidate-pool-size", str(args.candidate_pool_size),
        "--top-k", str(args.top_k),
    ]
    run(check_cmd)

    subset = None if args.smoke else create_repeatability_subset()

    for model_key in MODEL_ORDER:
        main_path = output_root / model_key / "benchmark_results.xlsx"
        repeat_path = output_root / "repeatability" / model_key / "benchmark_results.xlsx"
        if args.resume and main_path.exists() and (args.smoke or repeat_path.exists()):
            main_ok = workbook_matches_protocol(main_path, args, args.base_seed)
            repeat_ok = True if args.smoke else workbook_matches_protocol(
                repeat_path, args, args.base_seed + 1
            )
            if main_ok and repeat_ok:
                print(f"Skipping {model_key}: current-protocol result workbook(s) already exist.")
                continue
            print(f"Re-running {model_key}: existing outputs do not match protocol v{EXPECTED_BENCHMARK_SCRIPT_VERSION}.")

        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--model", model_key,
            "--runs", "1",
            "--candidate-pool-size", str(args.candidate_pool_size),
            "--top-k", str(args.top_k),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--base-seed", str(args.base_seed),
            "--output-root", str(output_root),
        ]
        if args.smoke:
            command.extend(["--limit", "2"])
        else:
            command.extend(["--repeatability-benchmark", str(REPEATABILITY_PATH)])
        if args.no_4bit:
            command.append("--no-4bit")
        run(command)

    run([
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--combine-results",
        "--output-root", str(output_root),
    ])

    if not args.smoke:
        assert subset is not None
        write_repeatability_summary(output_root, subset)

    print("\nFour-model benchmark completed successfully.")
    if args.smoke:
        print(f"Smoke comparison: {output_root / 'combined' / 'four_model_comparison.xlsx'}")
        print("Smoke outputs are isolated under Four_Models/Output/smoke and will not affect the final run.")
    else:
        print(f"Main comparison: {output_root / 'combined' / 'four_model_comparison.xlsx'}")
        print(f"Repeatability comparison: {output_root / 'repeatability' / 'repeatability_check.xlsx'}")
        print("Formal response count: 35 x 4 + 12 x 4 = 188")


if __name__ == "__main__":
    main()
