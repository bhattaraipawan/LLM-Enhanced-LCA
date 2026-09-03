"""One-time export of process-level GWP factors from the active openLCA database.

The resulting frozen snapshot lets the Colab production workflow perform later
BOM calculations without reconnecting to openLCA. The snapshot is tied to the
exact process-catalog hash and LCIA method/category identifiers.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import olca_schema as o

from .local_openlca import current_catalog_hash
from .local_calculate import calculate_ef, resolve_method_and_category

FACTOR_SNAPSHOT_VERSION = "1.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_matches(path: Path, *, catalog_hash: str, method_id: str, category_id: str) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(obj.get("schema_version")) == FACTOR_SNAPSHOT_VERSION
            and str(obj.get("catalog_content_sha256")) == str(catalog_hash)
            and str(obj.get("impact_method_id")) == str(method_id)
            and str(obj.get("impact_category_id")) == str(category_id)
            and isinstance(obj.get("factors"), dict)
        )
    except Exception:
        return False


def export_factor_snapshot(
    client,
    output_json: Path,
    metadata_json: Path,
    method_query: str,
    category_query: str,
    *,
    reuse_if_valid: bool = True,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Calculate one-unit LCIA factors for every process in the active catalog."""
    catalog_hash, process_count = current_catalog_hash(client)
    method_ref, category_ref = resolve_method_and_category(client, method_query, category_query)

    if reuse_if_valid and _snapshot_matches(
        output_json,
        catalog_hash=catalog_hash,
        method_id=method_ref.id,
        category_id=category_ref.id,
    ):
        obj = json.loads(output_json.read_text(encoding="utf-8"))
        meta = {
            "schema_version": FACTOR_SNAPSHOT_VERSION,
            "reused_existing_snapshot": True,
            "catalog_content_sha256": catalog_hash,
            "process_count": process_count,
            "impact_method_id": method_ref.id,
            "impact_method_name": method_ref.name,
            "impact_category_id": category_ref.id,
            "impact_category_name": category_ref.name,
            "impact_category_ref_unit": getattr(category_ref, "ref_unit", None),
            "resolved_factors": sum(1 for r in obj["factors"].values() if r.get("status") == "OK"),
            "failed_factors": sum(1 for r in obj["factors"].values() if r.get("status") != "OK"),
            "snapshot_sha256": _sha256_file(output_json),
        }
        metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    refs = list(client.get_descriptors(o.Process))
    refs.sort(key=lambda r: (str(getattr(r, "id", "")), str(getattr(r, "name", ""))))
    factors: dict[str, Any] = {}
    ok_count = 0

    print(f"Calculating frozen GWP factors for {len(refs)} processes...")
    for idx, ref in enumerate(refs, start=1):
        uid = str(getattr(ref, "id", "") or "").strip()
        if not uid:
            continue
        rec = {
            "process_uuid": uid,
            "process_name": getattr(ref, "name", None),
            "status": "ERROR",
            "emission_factor": None,
            "reference_unit": None,
            "process_type": None,
            "impact_basis": None,
            "impact_unit": getattr(category_ref, "ref_unit", None),
            "error": None,
        }
        try:
            ef, ref_unit, live_name, process_type, impact_basis = calculate_ef(
                client, uid, method_ref, category_ref
            )
            rec.update({
                "process_name": live_name,
                "status": "OK",
                "emission_factor": float(ef),
                "reference_unit": ref_unit,
                "process_type": process_type,
                "impact_basis": impact_basis,
            })
            ok_count += 1
        except Exception as exc:
            rec["error"] = str(exc)
        factors[uid] = rec
        if progress_every and (idx % progress_every == 0 or idx == len(refs)):
            print(f"  {idx}/{len(refs)} processes; factors resolved: {ok_count}")

    snapshot = {
        "schema_version": FACTOR_SNAPSHOT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_content_sha256": catalog_hash,
        "process_count": process_count,
        "impact_method_id": method_ref.id,
        "impact_method_name": method_ref.name,
        "impact_category_id": category_ref.id,
        "impact_category_name": category_ref.name,
        "impact_category_ref_unit": getattr(category_ref, "ref_unit", None),
        "calculation_rule": {
            "LCI_RESULT": "direct characterized impact of the selected target process",
            "UNIT_PROCESS": "total linked-system characterized impact",
        },
        "factors": factors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    sha = _sha256_file(output_json)
    metadata = {
        "schema_version": FACTOR_SNAPSHOT_VERSION,
        "reused_existing_snapshot": False,
        "generated_at_utc": snapshot["generated_at_utc"],
        "catalog_content_sha256": catalog_hash,
        "process_count": process_count,
        "impact_method_id": method_ref.id,
        "impact_method_name": method_ref.name,
        "impact_category_id": category_ref.id,
        "impact_category_name": category_ref.name,
        "impact_category_ref_unit": getattr(category_ref, "ref_unit", None),
        "resolved_factors": ok_count,
        "failed_factors": len(factors) - ok_count,
        "snapshot_sha256": sha,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
