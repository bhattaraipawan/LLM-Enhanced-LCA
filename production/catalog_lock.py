"""Freeze and verify the active ELCD retrieval catalog.

The matching protocol is benchmark-locked, while the ELCD catalog may be freshly
exported once for the final production run.  Immediately after export, its
semantic content hash is written to ``ELCD_Catalog_Lock.json``.  Production then
refuses to run if that catalog changes.

The semantic hash uses the exact six descriptor fields and sort rule used by the
four-model benchmark metadata, so XLSX timestamps/formatting do not affect it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CATALOG_LOCK_SCHEMA_VERSION = "1.0"
CATALOG_COLUMNS = [
    "process_uuid", "process_name", "category", "location", "library", "process_type"
]


def _safe_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value


def catalog_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path, sheet_name="Processes")
    required = {"process_uuid", "process_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    for col in CATALOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["process_uuid"].ne("") & df["process_name"].ne("")].copy()
    if df["process_uuid"].duplicated().any():
        dupes = df.loc[df["process_uuid"].duplicated(), "process_uuid"].head(5).tolist()
        raise ValueError(f"Catalog contains duplicate process UUIDs, e.g. {dupes}")
    return df[CATALOG_COLUMNS].copy()


def semantic_catalog_hash(path: str | Path) -> str:
    df = catalog_dataframe(path)
    use = df.copy().sort_values(["process_uuid"], kind="mergesort").reset_index(drop=True)
    records = []
    for _, row in use.iterrows():
        records.append({col: _safe_value(row[col]) for col in CATALOG_COLUMNS})
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_lock(
    catalog_path: str | Path,
    lock_path: str | Path,
    *,
    database_label: str,
    matching_protocol: dict[str, Any],
    historical_benchmark_catalog_hash: str | None = None,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    lock_path = Path(lock_path)
    df = catalog_dataframe(catalog_path)
    semantic_hash = semantic_catalog_hash(catalog_path)
    lock = {
        "schema_version": CATALOG_LOCK_SCHEMA_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "database_label_user_supplied": str(database_label),
        "catalog_file": catalog_path.name,
        "catalog_process_count": int(len(df)),
        "catalog_content_sha256": semantic_hash,
        "catalog_file_sha256": file_sha256(catalog_path),
        "historical_benchmark_catalog_content_sha256": historical_benchmark_catalog_hash or "",
        "same_as_historical_benchmark_catalog": bool(
            historical_benchmark_catalog_hash and semantic_hash == historical_benchmark_catalog_hash
        ),
        "matching_protocol": matching_protocol,
        "freeze_rule": (
            "This ELCD catalog is the final production retrieval snapshot. Regenerating or editing "
            "the catalog requires creating a new lock and constitutes a new production data snapshot."
        ),
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    return lock


def verify_lock(
    catalog_path: str | Path,
    lock_path: str | Path,
    *,
    expected_matching_protocol: dict[str, Any],
) -> dict[str, Any]:
    catalog_path = Path(catalog_path)
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise RuntimeError(
            f"Frozen ELCD catalog lock is missing: {lock_path}. Run `python main.py export ...` "
            "or `python scripts/freeze_elcd_catalog.py ...` after generating the final catalog."
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if str(lock.get("schema_version")) != CATALOG_LOCK_SCHEMA_VERSION:
        raise RuntimeError("Unsupported ELCD catalog lock schema version.")
    current = semantic_catalog_hash(catalog_path)
    frozen = str(lock.get("catalog_content_sha256") or "")
    if current != frozen:
        raise RuntimeError(
            "Frozen ELCD catalog drift detected. "
            f"Lock={frozen}; current={current}. Regenerate/freeze intentionally before continuing."
        )
    if int(lock.get("catalog_process_count") or -1) != len(catalog_dataframe(catalog_path)):
        raise RuntimeError("Frozen ELCD catalog process count does not match the lock.")
    recorded = lock.get("matching_protocol") or {}
    mismatches = {
        key: (recorded.get(key), expected)
        for key, expected in expected_matching_protocol.items()
        if recorded.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Matching-protocol drift detected in ELCD catalog lock: "
            + "; ".join(f"{k}: lock={a!r}, expected={b!r}" for k, (a, b) in mismatches.items())
        )
    return lock
