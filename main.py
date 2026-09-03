"""Local command-line entry point for the split Qwen/openLCA workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/"runtime"
DEFAULT_METHOD_ID="a4fa1dc6-317b-30ad-b2eb-6744ff77dcf0"
DEFAULT_CATEGORY_ID="209bd9be-d5c1-317b-a79b-2a76b79e495b"
FROZEN_ELCD_CATALOG=ROOT/"ELCD_Check"/"ELCD_Process_Catalog.xlsx"
FROZEN_ELCD_LOCK=ROOT/"ELCD_Check"/"ELCD_Catalog_Lock.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_python_constant(path: Path, name: str) -> str:
    import re
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(name)}\s*=\s*[\"\']([^\"\']+)[\"\']", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not read {name} from {path}")
    return m.group(1)


def _build_payload_manifest() -> dict:
    qwen = ROOT / "production/qwen_matcher.py"
    ext = ROOT / "production/external_ef_resolver.py"
    prop = ROOT / "production/property_resolver.py"
    tax = ROOT / "production/material_taxonomy.py"
    uncertainty = ROOT / "production/uncertainty.py"
    guardrails = ROOT / "production/guardrails.py"
    semantic_analog = ROOT / "production/semantic_analog.py"
    technical_equivalence = ROOT / "production/technical_equivalence.py"
    evidence_cache = ROOT / "production/evidence_cache.py"
    files = {
        "qwen_matcher.py": qwen,
        "catalog_lock.py": ROOT / "production/catalog_lock.py",
        "external_ef_resolver.py": ext,
        "property_resolver.py": prop,
        "unit_conversion.py": ROOT / "production/unit_conversion.py",
        "material_taxonomy.py": tax,
        "uncertainty.py": uncertainty,
        "guardrails.py": guardrails,
        "evidence_consensus.py": ROOT / "production/evidence_consensus.py",
        "semantic_analog.py": semantic_analog,
        "technical_equivalence.py": technical_equivalence,
        "evidence_cache.py": evidence_cache,
        "colab_calculate.py": ROOT / "production/colab_calculate.py",
        "colab_gui_runtime.py": ROOT / "colab/colab_gui_runtime.py",
    }
    return {
        "workflow_id": "llm-assisted-a1-a3-embodied-carbon-screening",
        "model_id": _read_python_constant(qwen, "MODEL_ID"),
        "model_revision": _read_python_constant(qwen, "MODEL_REVISION"),
        "production_prompt_version": _read_python_constant(qwen, "PRODUCTION_PROMPT_VERSION"),
        "safety_gate_version": _read_python_constant(qwen, "SAFETY_GATE_VERSION"),
        "external_ef_resolver_version": _read_python_constant(ext, "EXTERNAL_EF_RESOLVER_VERSION"),
        "property_resolver_version": _read_python_constant(prop, "PROPERTY_RESOLVER_VERSION"),
        "material_taxonomy_version": _read_python_constant(tax, "MATERIAL_TAXONOMY_VERSION"),
        "uncertainty_method_version": _read_python_constant(uncertainty, "UNCERTAINTY_METHOD_VERSION"),
        "guardrail_version": _read_python_constant(guardrails, "GUARDRAIL_VERSION"),
        "semantic_analog_version": _read_python_constant(semantic_analog, "SEMANTIC_ANALOG_VERSION"),
        "technical_equivalence_version": _read_python_constant(technical_equivalence, "TECHNICAL_EQUIVALENCE_VERSION"),
        "files_sha256": {name: _sha256_file(path) for name, path in files.items()},
    }


def make_colab_payload(catalog_json: Path, metadata_json: Path, reference_map_json: Path, factor_snapshot_json: Path, factor_snapshot_metadata_json: Path):
    RUNTIME.mkdir(exist_ok=True)
    manifest = _build_payload_manifest()
    if not FROZEN_ELCD_CATALOG.exists():
        raise FileNotFoundError(f"Frozen ELCD catalog missing: {FROZEN_ELCD_CATALOG}")
    if not FROZEN_ELCD_LOCK.exists():
        raise FileNotFoundError(f"Frozen ELCD catalog lock missing: {FROZEN_ELCD_LOCK}")
    catalog_lock = json.loads(FROZEN_ELCD_LOCK.read_text(encoding="utf-8"))
    manifest["retrieval_catalog_content_sha256"] = catalog_lock.get("catalog_content_sha256")
    manifest["historical_benchmark_catalog_content_sha256"] = catalog_lock.get("historical_benchmark_catalog_content_sha256")
    manifest["same_as_historical_benchmark_catalog"] = catalog_lock.get("same_as_historical_benchmark_catalog")
    manifest["matching_protocol"] = catalog_lock.get("matching_protocol")
    manifest["payload_data_sha256"] = {
        "ELCD_Process_Catalog.xlsx": _sha256_file(FROZEN_ELCD_CATALOG),
        "ELCD_Catalog_Lock.json": _sha256_file(FROZEN_ELCD_LOCK),
        "openlca_catalog.json": _sha256_file(catalog_json),
        "openlca_catalog_metadata.json": _sha256_file(metadata_json),
        "process_reference_units.json": _sha256_file(reference_map_json),
        "process_gwp_snapshot.json": _sha256_file(factor_snapshot_json),
        "process_gwp_snapshot_metadata.json": _sha256_file(factor_snapshot_metadata_json),
    }
    manifest_path = RUNTIME / "payload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    payload = RUNTIME / "Qwen_Colab_Payload.zip"
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Freshly exported final ELCD retrieval catalog, frozen immediately after export.
        # Matching protocol remains identical to benchmark script v2.3.0.
        z.write(FROZEN_ELCD_CATALOG, "ELCD_Process_Catalog.xlsx")
        z.write(FROZEN_ELCD_LOCK, "ELCD_Catalog_Lock.json")
        # Live local catalog remains audit metadata for the factor snapshot only;
        # Qwen never retrieves candidates from this JSON.
        z.write(catalog_json, "openlca_catalog.json")
        z.write(metadata_json, "openlca_catalog_metadata.json")
        z.write(reference_map_json, "process_reference_units.json")
        z.write(factor_snapshot_json, "process_gwp_snapshot.json")
        z.write(factor_snapshot_metadata_json, "process_gwp_snapshot_metadata.json")
        z.write(ROOT / "production/qwen_matcher.py", "qwen_matcher.py")
        z.write(ROOT / "production/catalog_lock.py", "catalog_lock.py")
        z.write(ROOT / "production/property_resolver.py", "property_resolver.py")
        z.write(ROOT / "production/external_ef_resolver.py", "external_ef_resolver.py")
        z.write(ROOT / "production/unit_conversion.py", "unit_conversion.py")
        z.write(ROOT / "production/material_taxonomy.py", "material_taxonomy.py")
        z.write(ROOT / "production/uncertainty.py", "uncertainty.py")
        z.write(ROOT / "production/guardrails.py", "guardrails.py")
        z.write(ROOT / "production/evidence_consensus.py", "evidence_consensus.py")
        z.write(ROOT / "production/semantic_analog.py", "semantic_analog.py")
        z.write(ROOT / "production/technical_equivalence.py", "technical_equivalence.py")
        z.write(ROOT / "production/evidence_cache.py", "evidence_cache.py")
        z.write(ROOT / "production/colab_calculate.py", "colab_calculate.py")
        z.write(ROOT / "colab/colab_gui_runtime.py", "colab_gui_runtime.py")
        z.write(manifest_path, "payload_manifest.json")
        z.write(ROOT / "requirements-colab.txt", "requirements-colab.txt")
    return payload, manifest_path, manifest


def cmd_export(args):
    """Regenerate ELCD data once, freeze it, and build the Colab payload.

    The process catalog is exported with the SAME exporter used by the four-model
    selection workflow. Prompt/model/retrieval/decoding/parser settings remain
    benchmark-identical; only the environmental-data snapshot is newly generated.
    """
    try:
        import subprocess
        import pandas as pd
        from production.local_openlca import connect, export_catalog, export_process_reference_map
        from production.factor_snapshot import export_factor_snapshot
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing local openLCA dependencies. Install them with: "
            "python -m pip install -r requirements-local.txt"
        ) from exc

    RUNTIME.mkdir(exist_ok=True)
    FROZEN_ELCD_CATALOG.parent.mkdir(parents=True, exist_ok=True)

    # 1) Exact same catalog exporter used for the four-model benchmark workflow.
    exporter = ROOT / "scripts" / "export_openlca_process_catalog.py"
    subprocess.run([
        sys.executable, str(exporter),
        "--port", str(args.port),
        "--database-label", str(args.database_label),
        "--output-dir", str(FROZEN_ELCD_CATALOG.parent),
        "--output-file", FROZEN_ELCD_CATALOG.name,
    ], check=True)

    # 2) Freeze the newly generated retrieval catalog together with the exact
    # benchmark matching protocol. No old catalog hash is required.
    freezer = ROOT / "scripts" / "freeze_elcd_catalog.py"
    subprocess.run([
        sys.executable, str(freezer),
        "--catalog", str(FROZEN_ELCD_CATALOG),
        "--lock", str(FROZEN_ELCD_LOCK),
        "--database-label", str(args.database_label),
    ], check=True)

    # 3) Export the quantitative data from the same currently active database.
    client=connect(args.port)
    catalog=RUNTIME/"openlca_catalog.json"
    metadata=RUNTIME/"openlca_catalog_metadata.json"
    meta=export_catalog(client,catalog,metadata,args.database_label,args.port)

    # Fail closed if the benchmark-style XLSX export and the live quantitative
    # export do not refer to the same process UUID universe.
    frozen_df = pd.read_excel(FROZEN_ELCD_CATALOG, sheet_name="Processes")
    frozen_uuids = set(frozen_df["process_uuid"].dropna().astype(str).str.strip().str.lower())
    live_rows = json.loads(catalog.read_text(encoding="utf-8"))
    live_uuids = {str(r.get("process_uuid") or "").strip().lower() for r in live_rows}
    live_uuids.discard("")
    if frozen_uuids != live_uuids:
        missing = sorted(frozen_uuids - live_uuids)[:10]
        extra = sorted(live_uuids - frozen_uuids)[:10]
        raise RuntimeError(
            "Fresh ELCD retrieval catalog and quantitative export do not use the same process UUID universe. "
            f"Missing from quantitative export={missing}; extra={extra}."
        )

    reference_map=RUNTIME/"process_reference_units.json"
    ref_meta=export_process_reference_map(client,reference_map)

    factor_snapshot=RUNTIME/"process_gwp_snapshot.json"
    factor_snapshot_meta=RUNTIME/"process_gwp_snapshot_metadata.json"
    factor_meta=export_factor_snapshot(
        client, factor_snapshot, factor_snapshot_meta, args.method, args.category,
        reuse_if_valid=False,
    )

    payload, manifest_path, manifest = make_colab_payload(
        catalog, metadata, reference_map, factor_snapshot, factor_snapshot_meta
    )
    lock = json.loads(FROZEN_ELCD_LOCK.read_text(encoding="utf-8"))
    print(f"Processes exported: {meta['process_count']}")
    print(f"Reference units resolved: {ref_meta['reference_units_resolved']}/{ref_meta['processes_seen']}")
    print(f"Frozen GWP factors resolved: {factor_meta['resolved_factors']}/{factor_meta['process_count']}")
    print(f"Impact method: {factor_meta['impact_method_name']} [{factor_meta['impact_method_id']}]")
    print(f"Impact category: {factor_meta['impact_category_name']} [{factor_meta['impact_category_id']}]")
    print(f"Fresh frozen retrieval catalog: {FROZEN_ELCD_CATALOG}")
    print(f"Fresh retrieval semantic hash: {lock['catalog_content_sha256']}")
    print(
        "Same semantic catalog as historical four-model benchmark: "
        + str(lock.get('same_as_historical_benchmark_catalog'))
    )
    print(f"Catalog lock: {FROZEN_ELCD_LOCK}")
    print(f"Factor snapshot: {factor_snapshot}")
    print(f"Colab payload: {payload}")
    print(f"Payload manifest: {manifest_path}")
    print(
        "Matching protocol remains benchmark-locked: "
        f"model_revision={manifest['model_revision']}, "
        f"prompt={manifest['production_prompt_version']}, "
        "pool=5, top_k=3, max_new_tokens=256, seed=42, greedy, 4-bit NF4."
    )


def cmd_methods(args):
    """List LCIA methods and impact categories in the active openLCA database."""
    try:
        from production.local_openlca import connect, list_impact_methods
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing local openLCA dependencies. Install them with: "
            "python -m pip install -r requirements-local.txt"
        ) from exc

    RUNTIME.mkdir(exist_ok=True)
    rows = list_impact_methods(connect(args.port))
    if args.filter:
        q = str(args.filter).lower()
        rows = [
            r for r in rows
            if q in str(r.get("impact_method_name", "")).lower()
            or q in str(r.get("impact_category_name", "")).lower()
        ]

    out = RUNTIME / "openlca_impact_methods.csv"
    fields = [
        "impact_method_id",
        "impact_method_name",
        "impact_category_id",
        "impact_category_name",
        "impact_category_ref_unit",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        print(
            "No impact methods/categories matched. "
            "The active database may not contain an LCIA method."
        )
    else:
        for row in rows[:100]:
            print(
                f"METHOD: {row['impact_method_name']} | "
                f"CATEGORY: {row['impact_category_name']} | "
                f"UNIT: {row['impact_category_ref_unit']}"
            )
        if len(rows) > 100:
            print(f"... {len(rows) - 100} additional rows written to CSV")
    print(f"Saved: {out}")


def cmd_calculate(args):
    try:
        from production.local_calculate import calculate_selection_file
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing local calculation dependencies. Install them with: "
            "python -m pip install -r requirements-local.txt"
        ) from exc
    out=Path(args.output) if args.output else RUNTIME/"Qwen_OpenLCA_Final_Results.xlsx"
    path,_,summary=calculate_selection_file(
        args.input,out,args.method,args.category,port=args.port,allow_catalog_mismatch=args.allow_catalog_mismatch
    )
    print(summary.to_string(index=False))
    print(f"Final workbook: {path}")


def main():
    p=argparse.ArgumentParser(description="Qwen/openLCA split production workflow")
    sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("export",help="Regenerate ELCD data, freeze the final catalog, and build the Colab payload")
    e.add_argument("--port",type=int,default=8080)
    e.add_argument("--database-label",default="ELCD 3.2")
    e.add_argument("--method",default=DEFAULT_METHOD_ID,help="Impact method UUID/name query")
    e.add_argument("--category",default=DEFAULT_CATEGORY_ID,help="Impact category UUID/name query")
    e.set_defaults(func=cmd_export)
    m=sub.add_parser("methods",help="List LCIA methods/categories in active openLCA")
    m.add_argument("--port",type=int,default=8080); m.add_argument("--filter",default=""); m.set_defaults(func=cmd_methods)
    c=sub.add_parser("calculate",help="Calculate GWP locally from Qwen selection workbook")
    c.add_argument("--input",required=True); c.add_argument("--method",required=True); c.add_argument("--category",required=True)
    c.add_argument("--output"); c.add_argument("--port",type=int,default=8080); c.add_argument("--allow-catalog-mismatch",action="store_true")
    c.set_defaults(func=cmd_calculate)
    args=p.parse_args(); args.func(args)

if __name__=="__main__": main()
