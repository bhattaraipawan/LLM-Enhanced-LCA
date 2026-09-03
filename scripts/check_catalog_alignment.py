"""Verify that the frozen retrieval catalog and quantitative export use the same UUID universe.

The XLSX catalog is generated with the benchmark exporter and then hash-frozen.
The JSON catalog is an auxiliary quantitative-export representation used for the
factor/reference-unit snapshot. Descriptor text is not required to be byte-for-
byte identical across those two representations; exact process UUID identity is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL = ROOT / "ELCD_Check" / "ELCD_Process_Catalog.xlsx"
DEFAULT_PRODUCTION = ROOT / "runtime" / "openlca_catalog.json"


def clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval-catalog", type=Path, default=DEFAULT_RETRIEVAL)
    ap.add_argument("--production-catalog", type=Path, default=DEFAULT_PRODUCTION)
    args = ap.parse_args()

    b = pd.read_excel(args.retrieval_catalog, sheet_name="Processes")
    rows = json.loads(args.production_catalog.read_text(encoding="utf-8"))
    p = pd.DataFrame(rows)
    b_ids = {clean(v) for v in b["process_uuid"].tolist() if clean(v)}
    p_ids = {clean(v) for v in p["process_uuid"].tolist() if clean(v)}
    missing = sorted(b_ids - p_ids)
    extra = sorted(p_ids - b_ids)

    print(f"Frozen retrieval processes: {len(b_ids)}")
    print(f"Quantitative export processes: {len(p_ids)}")
    print(f"Missing from quantitative export: {len(missing)}")
    print(f"Extra in quantitative export: {len(extra)}")
    if missing[:10]: print("Missing UUID examples:", missing[:10])
    if extra[:10]: print("Extra UUID examples:", extra[:10])
    if missing or extra:
        print("FAIL — retrieval and factor snapshots were not generated from the same process universe.")
        return 1
    print("PASS — retrieval catalog and quantitative snapshot use the same process UUID universe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
