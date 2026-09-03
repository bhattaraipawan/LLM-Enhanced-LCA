"""Local openLCA utilities for the split production workflow.

This module is intentionally local-only. It connects to openLCA Desktop on
localhost and never loads an LLM.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import olca_ipc as ipc
import olca_schema as o


@dataclass(frozen=True)
class CatalogRecord:
    process_uuid: str
    process_name: str
    category: str | None
    location: str | None
    process_type: str | None
    ref_unit: str | None
    library: str | None


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _descriptor_text(value: Any) -> str | None:
    """Match the benchmark catalog export's descriptor-to-text conversion."""
    if value is None:
        return None
    text = getattr(value, "value", str(value))
    text = str(text).strip()
    return text or None


def _category_path(value: Any) -> str | None:
    if not value:
        return None
    text = " > ".join(str(part) for part in value).strip()
    return text or None


def connect(port: int = 8080) -> ipc.Client:
    return ipc.Client(port)


def process_catalog(client: ipc.Client) -> list[CatalogRecord]:
    refs = client.get_descriptors(o.Process)
    records: list[CatalogRecord] = []
    for ref in refs:
        uid = _clean(getattr(ref, "id", None))
        name = _clean(getattr(ref, "name", None))
        if not uid or not name:
            continue
        records.append(
            CatalogRecord(
                process_uuid=uid,
                process_name=name,
                category=_category_path(getattr(ref, "category_path", None)),
                location=_descriptor_text(getattr(ref, "location", None)),
                process_type=_descriptor_text(getattr(ref, "process_type", None)),
                ref_unit=_descriptor_text(getattr(ref, "ref_unit", None)),
                library=_descriptor_text(getattr(ref, "library", None)),
            )
        )
    records.sort(key=lambda r: (r.process_uuid, r.process_name))
    return records


def catalog_content_hash(records: Iterable[CatalogRecord]) -> str:
    rows = [asdict(r) for r in records]
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_catalog(
    client: ipc.Client,
    output_json: Path,
    metadata_json: Path,
    database_label: str = "ELCD 3.2",
    port: int = 8080,
) -> dict[str, Any]:
    records = process_catalog(client)
    if not records:
        raise RuntimeError(
            "No process descriptors were returned. Confirm openLCA is open, the intended database is active, "
            "and the IPC server is running."
        )
    content_hash = catalog_content_hash(records)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "schema_version": "1.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_label_user_supplied": database_label,
        "ipc_endpoint": f"http://localhost:{port}",
        "process_count": len(records),
        "catalog_content_sha256": content_hash,
        "note": (
            "The database label is user-supplied metadata. The script exports the database that is active "
            "in openLCA Desktop when the IPC server is running."
        ),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def list_impact_methods(client: ipc.Client) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    refs = client.get_descriptors(o.ImpactMethod)
    for ref in refs:
        method = client.get(o.ImpactMethod, uid=ref.id)
        if method is None:
            continue
        categories = getattr(method, "impact_categories", None) or []
        if not categories:
            rows.append({
                "impact_method_id": ref.id,
                "impact_method_name": ref.name,
                "impact_category_id": None,
                "impact_category_name": None,
                "impact_category_ref_unit": None,
            })
            continue
        for cat in categories:
            rows.append({
                "impact_method_id": ref.id,
                "impact_method_name": ref.name,
                "impact_category_id": getattr(cat, "id", None),
                "impact_category_name": getattr(cat, "name", None),
                "impact_category_ref_unit": getattr(cat, "ref_unit", None),
            })
    return rows


def find_quantitative_reference(process: o.Process):
    exchanges = getattr(process, "exchanges", None) or []
    for ex in exchanges:
        if bool(getattr(ex, "is_quantitative_reference", False)):
            return ex
    return None


def current_catalog_hash(client: ipc.Client) -> tuple[str, int]:
    records = process_catalog(client)
    return catalog_content_hash(records), len(records)



def export_process_reference_map(client: ipc.Client, output_json: Path) -> dict[str, Any]:
    """Export live quantitative-reference metadata separately from the catalog hash.

    This auxiliary map lets the Colab property resolver know the selected process
    reference unit before local LCIA. It does not alter the process-catalog hash
    used by the reproducibility safeguard.
    """
    refs = client.get_descriptors(o.Process)
    mapping: dict[str, Any] = {}
    resolved = 0
    for ref in refs:
        uid = _clean(getattr(ref, "id", None))
        if not uid:
            continue
        try:
            process = client.get(o.Process, uid=uid)
            if process is None:
                continue
            qref = find_quantitative_reference(process)
            if qref is None:
                continue
            unit = getattr(qref, "unit", None)
            flow = getattr(qref, "flow", None)
            mapping[uid] = {
                "ref_unit": _clean(getattr(unit, "name", None)),
                "qref_amount": getattr(qref, "amount", None),
                "qref_flow_name": _clean(getattr(flow, "name", None)),
                "qref_flow_uuid": _clean(getattr(flow, "id", None)),
                "process_type": _enum_value(getattr(process, "process_type", None)),
            }
            if mapping[uid]["ref_unit"]:
                resolved += 1
        except Exception as exc:
            mapping[uid] = {"error": str(exc)}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "processes_seen": len(refs),
        "reference_units_resolved": resolved,
        "output": str(output_json),
    }
