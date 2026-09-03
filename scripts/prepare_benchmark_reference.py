"""Validate and freeze the reconciled expert reference set for LLM benchmarking.

The human expert workbook is the scientific ground truth. This helper does not
invent, average, or change expert labels. It only:

1. reads the final values from the Reconciliation sheet;
2. enforces the study's simplified resolution rule;
3. validates Direct/Proxy process UUIDs against the exported ELCD catalog; and
4. writes a compact frozen workbook used by the four-model benchmark.

Study rule
----------
Direct
    A selected ELCD process is considered a sufficiently direct representation.
Proxy
    An ELCD process is available and intentionally used as a substitute.
Review Required
    No usable ELCD process is available. These rows intentionally have no
    process UUID and are the rows eligible for the later LLM-fallback stage.

Default input:
    ELCD_Check/expert_reference/LLM_LCA_Expert_Reference_Set_With_ELCD.xlsx

Default output:
    Four_Models/Input/LLM_Model_Evaluation_Reference_Set.xlsx

Examples
--------
Validate only::

    python scripts/prepare_benchmark_reference.py --validate-only

Validate and freeze::

    python scripts/prepare_benchmark_reference.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_VERSION = "2.0.0"
EXPECTED_ROWS = 35
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERT_WORKBOOK = (
    REPO_ROOT
    / "ELCD_Check"
    / "expert_reference"
    / "LLM_LCA_Expert_Reference_Set_With_ELCD.xlsx"
)
DEFAULT_CATALOG = REPO_ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
DEFAULT_OUTPUT = (
    REPO_ROOT / "Four_Models" / "Input" / "LLM_Model_Evaluation_Reference_Set.xlsx"
)

UNAVAILABLE_LABELS = {
    "",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "none",
    "no match",
    "no defensible match",
    "no defensible elcd match",
}


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def key(value: Any) -> str:
    return " ".join(
        clean(value).lower().replace("_", " ").replace("-", " ").split()
    )


def is_unavailable(value: Any) -> bool:
    return key(value) in UNAVAILABLE_LABELS


def canonical_match_type(value: Any) -> str:
    text = key(value)
    if text in {"direct", "direct match", "exact", "exact match"}:
        return "Direct"
    if text in {"proxy", "proxy match", "documented proxy"}:
        return "Proxy"
    if text in {
        "review required",
        "review",
        "unresolved",
        "no match",
        "no defensible match",
        "no defensible elcd match",
        "not available",
    }:
        return "Review Required"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate expert reconciliation and freeze the four-model benchmark input."
    )
    parser.add_argument("--expert-workbook", type=Path, default=DEFAULT_EXPERT_WORKBOOK)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all 35 rows but do not write the frozen benchmark workbook.",
    )
    return parser.parse_args()


def find_header(columns: list[Any], candidates: list[str]) -> str | None:
    normalized = {key(col): str(col) for col in columns}
    for candidate in candidates:
        found = normalized.get(key(candidate))
        if found is not None:
            return found
    return None


def load_base_rows(expert_path: Path) -> pd.DataFrame:
    """Read immutable BOM descriptors from Expert_A.

    Only ID/case/BOM/quantity/unit are used. Expert_A's judgment fields are not
    used to manufacture the final labels.
    """
    base = pd.read_excel(expert_path, sheet_name="Expert_A", header=3)
    rename = {
        "ID": "sample_id",
        "Case Study": "case_study",
        "Original BOM Description": "material_description",
        "Qty.": "quantity",
        "Qty": "quantity",
        "Quantity": "quantity",
        "Unit": "unit",
    }
    base = base.rename(columns={c: rename[c] for c in base.columns if c in rename})
    required = ["sample_id", "case_study", "material_description", "quantity", "unit"]
    missing = [c for c in required if c not in base.columns]
    if missing:
        raise ValueError(f"Expert_A sheet is missing columns: {missing}")
    base = base[required].copy()
    base = base[base["sample_id"].notna()].copy()
    base["sample_id"] = base["sample_id"].map(clean)
    return base.reset_index(drop=True)


def load_reconciliation(expert_path: Path) -> pd.DataFrame:
    rec = pd.read_excel(expert_path, sheet_name="Reconciliation", header=5)

    # Header aliases support both the older and the cleaned workbook versions.
    column_map = {
        "sample_id": find_header(list(rec.columns), ["ID"]),
        "reconciliation_description": find_header(
            list(rec.columns), ["Original BOM Description"]
        ),
        "ground_truth_normalized_material": find_header(
            list(rec.columns), ["Final Normalized Material"]
        ),
        "ground_truth_process_name": find_header(
            list(rec.columns), ["Final Reference Process"]
        ),
        "ground_truth_process_uuid": find_header(
            list(rec.columns), ["Final Process UUID", "Final Process UUID (auto)"]
        ),
        "ground_truth_match_type": find_header(
            list(rec.columns), ["Final Decision", "Final Match Type"]
        ),
        "reviewer_notes": find_header(list(rec.columns), ["Notes"]),
    }

    missing = [name for name, source in column_map.items() if source is None and name != "reviewer_notes"]
    if missing:
        raise ValueError(f"Reconciliation sheet is missing required fields: {missing}")

    data: dict[str, Any] = {}
    for target, source in column_map.items():
        data[target] = rec[source] if source is not None else ""
    out = pd.DataFrame(data)
    out = out[out["sample_id"].notna()].copy()
    out["sample_id"] = out["sample_id"].map(clean)
    return out.reset_index(drop=True)


def load_catalog(catalog_path: Path) -> pd.DataFrame:
    catalog = pd.read_excel(catalog_path, sheet_name="Processes")
    required = ["process_uuid", "process_name"]
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"Catalog is missing columns: {missing}")

    for optional in ["category", "location", "library", "process_type"]:
        if optional not in catalog.columns:
            catalog[optional] = ""

    cols = required + ["category", "location", "library", "process_type"]
    catalog = catalog[cols].copy()
    for col in cols:
        catalog[col] = catalog[col].map(clean)

    catalog = catalog[
        catalog["process_uuid"].ne("") & catalog["process_name"].ne("")
    ].copy()
    if catalog["process_uuid"].str.lower().duplicated().any():
        raise ValueError("Catalog contains duplicate process UUIDs.")
    return catalog.reset_index(drop=True)


def validate_and_build(
    base: pd.DataFrame,
    rec: pd.DataFrame,
    catalog: pd.DataFrame,
    source_path: Path,
) -> pd.DataFrame:
    if len(base) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} BOM rows in Expert_A; found {len(base)}."
        )
    if len(rec) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} reconciliation rows; found {len(rec)}."
        )
    if base["sample_id"].duplicated().any() or rec["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found in the expert workbook.")

    merged = base.merge(rec, on="sample_id", how="left", validate="one_to_one")
    if merged["reconciliation_description"].isna().any():
        missing_ids = merged.loc[
            merged["reconciliation_description"].isna(), "sample_id"
        ].tolist()
        raise ValueError(f"Reconciliation is missing sample IDs: {missing_ids}")

    uuid_to_row = {
        clean(row.process_uuid).lower(): row
        for row in catalog.itertuples(index=False)
    }

    errors: list[str] = []
    output_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        sid = clean(row["sample_id"])
        base_description = clean(row["material_description"])
        rec_description = clean(row["reconciliation_description"])
        normalized = clean(row["ground_truth_normalized_material"])
        process_name = clean(row["ground_truth_process_name"])
        process_uuid = clean(row["ground_truth_process_uuid"]).lower()
        match_type = canonical_match_type(row["ground_truth_match_type"])
        notes = clean(row["reviewer_notes"])

        if key(base_description) != key(rec_description):
            errors.append(
                f"{sid}: BOM description differs between Expert_A and Reconciliation."
            )
        if not normalized:
            errors.append(f"{sid}: Final Normalized Material is blank.")
        if not match_type:
            errors.append(
                f"{sid}: Final Decision must be Direct, Proxy, or Review Required."
            )

        unresolved = match_type == "Review Required"

        if unresolved:
            # The expert workbook may display "Not available" for readability.
            # The frozen benchmark stores unresolved process fields as blank.
            if process_uuid:
                errors.append(
                    f"{sid}: Review Required row must not contain a process UUID."
                )
            if process_name and not is_unavailable(process_name):
                errors.append(
                    f"{sid}: Review Required is allowed only when the final process is unavailable; "
                    f"found {process_name!r}."
                )
            canonical_name = ""
            canonical_uuid = ""
            category = ""
            location = ""
            library = ""
            process_type = ""
        else:
            if is_unavailable(process_name):
                errors.append(
                    f"{sid}: {match_type} requires a selected ELCD process, not 'Not available'."
                )
            if not process_uuid:
                errors.append(
                    f"{sid}: {match_type} row is missing Final Process UUID."
                )
                canonical_name = ""
                canonical_uuid = ""
                category = location = library = process_type = ""
            elif process_uuid not in uuid_to_row:
                errors.append(
                    f"{sid}: Final Process UUID {process_uuid!r} is not present in the exported catalog."
                )
                canonical_name = ""
                canonical_uuid = process_uuid
                category = location = library = process_type = ""
            else:
                catalog_row = uuid_to_row[process_uuid]
                canonical_uuid = clean(catalog_row.process_uuid).lower()
                canonical_name = clean(catalog_row.process_name)
                category = clean(catalog_row.category)
                location = clean(catalog_row.location)
                library = clean(catalog_row.library)
                process_type = clean(catalog_row.process_type)

        output_rows.append(
            {
                "sample_id": sid,
                "case_study": clean(row["case_study"]),
                "material_description": base_description,
                "quantity": row["quantity"],
                "unit": clean(row["unit"]),
                "ground_truth_normalized_material": normalized,
                "ground_truth_process_name": canonical_name,
                "ground_truth_process_uuid": canonical_uuid,
                "ground_truth_match_type": match_type,
                "ground_truth_unresolved": unresolved,
                "ground_truth_process_category": category,
                "ground_truth_process_location": location,
                "ground_truth_process_library": library,
                "ground_truth_process_type": process_type,
                "reference_status": "FINAL",
                "reviewer_notes": notes,
                "source_location": (
                    str(source_path.relative_to(REPO_ROOT))
                    if source_path.is_relative_to(REPO_ROOT)
                    else str(source_path)
                ),
            }
        )

    if errors:
        shown = "\n  - ".join(errors[:40])
        extra = f"\n  ... plus {len(errors) - 40} more" if len(errors) > 40 else ""
        raise ValueError(
            "Expert reconciliation is not ready to freeze:\n  - " + shown + extra
        )

    result = pd.DataFrame(output_rows)

    # Final semantic guardrails implementing the user's intended hierarchy.
    bad_review = result[
        (result["ground_truth_match_type"] == "Review Required")
        & result["ground_truth_process_uuid"].ne("")
    ]
    bad_matched = result[
        result["ground_truth_match_type"].isin(["Direct", "Proxy"])
        & result["ground_truth_process_uuid"].eq("")
    ]
    if not bad_review.empty or not bad_matched.empty:
        raise AssertionError("Internal classification validation failed.")

    return result


def write_output(
    df: pd.DataFrame,
    output_path: Path,
    expert_path: Path,
    catalog_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def repo_display(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    counts = df["ground_truth_match_type"].value_counts().to_dict()
    metadata = pd.DataFrame(
        [
            {"field": "script_version", "value": SCRIPT_VERSION},
            {"field": "reference_set_rows", "value": len(df)},
            {"field": "reference_status", "value": "FINAL"},
            {
                "field": "frozen_at_utc",
                "value": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
            {"field": "direct_rows", "value": counts.get("Direct", 0)},
            {"field": "proxy_rows", "value": counts.get("Proxy", 0)},
            {
                "field": "review_required_rows",
                "value": counts.get("Review Required", 0),
            },
            {"field": "source_expert_workbook", "value": repo_display(expert_path)},
            {"field": "catalog_path", "value": repo_display(catalog_path)},
            {
                "field": "prepared_by",
                "value": "scripts/prepare_benchmark_reference.py",
            },
        ]
    )

    instructions = pd.DataFrame(
        [
            {
                "item": "Status",
                "details": "FINAL frozen human reference set. Do not edit after model scoring begins.",
            },
            {
                "item": "Direct",
                "details": "A selected ELCD process is considered a sufficiently direct representation.",
            },
            {
                "item": "Proxy",
                "details": "A selected ELCD process is intentionally used as a defensible substitute.",
            },
            {
                "item": "Review Required",
                "details": "No usable ELCD process is available; process name/UUID are intentionally blank in this frozen file.",
            },
            {
                "item": "No leakage",
                "details": "The benchmark retrieves candidates from the original BOM description only; human normalized labels are used only for scoring.",
            },
        ]
    )

    # openpyxl is used here because this is a repository script intended to be
    # executed by the researcher, not model-generated scientific content.
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        df.to_excel(writer, sheet_name="Reference_Set", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cells in ws.columns:
                letter = cells[0].column_letter
                max_len = max(len(str(c.value or "")) for c in cells[:200])
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 60)


def main() -> None:
    args = parse_args()
    expert_path = args.expert_workbook.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not expert_path.exists():
        raise FileNotFoundError(f"Expert workbook not found: {expert_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    base = load_base_rows(expert_path)
    rec = load_reconciliation(expert_path)
    catalog = load_catalog(catalog_path)
    frozen = validate_and_build(base, rec, catalog, expert_path)

    counts = frozen["ground_truth_match_type"].value_counts().to_dict()
    print("Expert reference validation passed.")
    print(f"Rows: {len(frozen)}")
    print(
        "Final labels: "
        f"Direct={counts.get('Direct', 0)}, "
        f"Proxy={counts.get('Proxy', 0)}, "
        f"Review Required={counts.get('Review Required', 0)}"
    )

    if args.validate_only:
        print("Validation only: no frozen workbook written.")
        return

    write_output(frozen, output_path, expert_path, catalog_path)
    print(f"Frozen benchmark reference set saved: {output_path}")


if __name__ == "__main__":
    main()
