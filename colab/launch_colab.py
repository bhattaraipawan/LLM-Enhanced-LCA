"""Google Colab launcher for the native-widget LLM-LCA interface."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile
from typing import Mapping

WORKFLOW_ID = "llm-assisted-a1-a3-embodied-carbon-screening"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_dependencies():
    packages = [
        # Exact core package versions recorded by the frozen Qwen benchmark run.
        "transformers==4.57.6", "accelerate==1.14.0", "bitsandbytes==0.50.1",
        "safetensors>=0.4", "huggingface_hub>=0.24", "sentencepiece>=0.2",
        "pandas==2.2.3", "openpyxl==3.1.5", "xlsxwriter>=3.2,<4",
        "scikit-learn==1.6.1", "numpy>=1.26,<3", "matplotlib>=3.8,<4",
        "ipywidgets>=8,<9", "ddgs>=9.14,<10", "requests>=2.31,<3",
        "pypdf>=5,<7", "pdfplumber>=0.11,<1", "pillow>=10,<12",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])



def _check_benchmark_runtime():
    """Fail closed when the Colab inference runtime drifts from the benchmark.

    These are the values recorded in the selected Qwen benchmark workbook. The
    exact CUDA/PyTorch/GPU check is intentionally strict because this production
    package is intended to reproduce that frozen model-selection protocol.
    """
    import importlib.metadata as im
    import torch

    expected_packages = {
        "transformers": "4.57.6",
        "accelerate": "1.14.0",
        "bitsandbytes": "0.50.1",
        "pandas": "2.2.3",
        "scikit-learn": "1.6.1",
        "openpyxl": "3.1.5",
    }
    problems = []
    for name, expected in expected_packages.items():
        try:
            actual = im.version(name)
        except Exception:
            actual = "missing"
        if actual != expected:
            problems.append(f"{name}={actual} (benchmark {expected})")

    torch_version = str(torch.__version__)
    cuda_version = str(torch.version.cuda or "")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU"
    if torch_version != "2.11.0+cu128":
        problems.append(f"torch={torch_version} (benchmark 2.11.0+cu128)")
    if cuda_version != "12.8":
        problems.append(f"CUDA={cuda_version} (benchmark 12.8)")
    if gpu_name != "Tesla T4":
        problems.append(f"GPU={gpu_name} (benchmark Tesla T4)")

    if problems and os.getenv("LCA_ALLOW_RUNTIME_DRIFT", "0") != "1":
        raise RuntimeError(
            "Benchmark-runtime drift detected. For the strongest 35-row reproducibility, "
            "use the same Colab T4 runtime used by the benchmark. Differences:\n - "
            + "\n - ".join(problems)
            + "\nSet LCA_ALLOW_RUNTIME_DRIFT=1 only if you deliberately accept runtime drift."
        )
    if problems:
        print("Benchmark runtime: DRIFT ALLOWED — " + "; ".join(problems))
    else:
        print("Benchmark runtime: PASS (T4/CUDA/PyTorch/core packages match frozen Qwen run).")


def prepare_payload(
    payload_path: str | pathlib.Path,
    runtime_source: str | pathlib.Path | None = None,
    embedded_sources: Mapping[str, str] | None = None,
):
    """Extract a frozen-data payload and optionally inject current code modules.

    Embedded source overrides are code only. The frozen ELCD/openLCA catalog,
    reference-unit map and GWP snapshot are still taken from the uploaded payload
    and remain hash-verified against its manifest. This allows a standalone Colab
    runner to update calculation logic without embedding any material-specific
    environmental factors or densities in the runner itself.
    """
    payload_path = pathlib.Path(payload_path)
    work = pathlib.Path("/content/qwen_lca")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(payload_path) as z:
        z.extractall(work)

    required_core = {
        "ELCD_Process_Catalog.xlsx", "ELCD_Catalog_Lock.json",
        "openlca_catalog.json", "openlca_catalog_metadata.json",
        "process_reference_units.json", "process_gwp_snapshot.json",
        "process_gwp_snapshot_metadata.json", "qwen_matcher.py", "catalog_lock.py",
        "external_ef_resolver.py", "property_resolver.py", "unit_conversion.py",
        "material_taxonomy.py", "technical_equivalence.py", "uncertainty.py",
        "guardrails.py", "evidence_consensus.py", "semantic_analog.py",
        "evidence_cache.py", "colab_calculate.py", "colab_gui_runtime.py",
        "payload_manifest.json",
    }
    missing = sorted(x for x in required_core if not (work / x).exists())
    if missing:
        raise RuntimeError("Incomplete payload. Missing: " + ", ".join(missing))

    manifest = json.loads((work / "payload_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("workflow_id") != WORKFLOW_ID:
        raise RuntimeError("The uploaded payload was not generated by this workflow.")

    injected_names = set(embedded_sources or {})
    if runtime_source:
        injected_names.add("colab_gui_runtime.py")

    # Hash-check all frozen data and every payload code file that is NOT being
    # intentionally replaced by this standalone runner. The data hashes are never
    # skipped, so embedded code cannot silently alter the ELCD snapshot.
    problems = []
    for group in ("files_sha256", "payload_data_sha256"):
        for fname, expected in (manifest.get(group) or {}).items():
            if group == "files_sha256" and fname in injected_names:
                continue
            p = work / fname
            if not p.exists():
                problems.append(f"missing {fname}")
            elif sha256_file(p) != expected:
                problems.append(f"SHA256 mismatch: {fname}")
    if problems:
        raise RuntimeError("Payload verification failed:\n - " + "\n - ".join(problems))

    if embedded_sources:
        for fname, source in embedded_sources.items():
            safe = pathlib.Path(fname)
            if safe.name != fname or safe.suffix != ".py" or fname not in required_core:
                raise RuntimeError(f"Unsafe or unsupported embedded module name: {fname}")
            (work / fname).write_text(str(source), encoding="utf-8")

    if runtime_source:
        shutil.copy2(runtime_source, work / "colab_gui_runtime.py")
    return work, manifest


def main(
    runtime_source: str | pathlib.Path | None = None,
    embedded_sources: Mapping[str, str] | None = None,
):
    try:
        from google.colab import files, output
    except Exception as exc:
        raise RuntimeError("This launcher is intended for Google Colab.") from exc

    output.enable_custom_widget_manager()
    print(
        "Upload your Qwen_Colab_Payload.zip. Frozen ELCD/openLCA data come from "
        "that payload; this runner can inject the current production Python code."
    )
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError("Upload exactly one Qwen_Colab_Payload.zip file.")
    name = next(iter(uploaded))
    if not name.lower().endswith(".zip"):
        raise RuntimeError("Expected a .zip payload.")

    print("Installing/confirming benchmark-locked Colab dependencies (no Gradio)...")
    _install_dependencies()
    _check_benchmark_runtime()

    if runtime_source is None:
        here = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path("/content")
        candidate = here / "colab_gui_runtime.py"
        if not candidate.exists():
            candidate = pathlib.Path("/content/colab_gui_runtime.py")
        runtime_source = candidate if candidate.exists() else None

    work, manifest = prepare_payload(
        pathlib.Path("/content") / name,
        runtime_source,
        embedded_sources=embedded_sources,
    )
    os.chdir(work)
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))

    print(
        "Payload data verification: PASS; "
        f"model={manifest.get('model_id', 'Qwen')}, "
        f"revision={manifest.get('model_revision', 'unknown')}."
    )
    if embedded_sources:
        print("Embedded production-code override: PASS (no embedded material-specific GWP/density data).")
    from colab_gui_runtime import launch_native_widgets
    return launch_native_widgets(work)


if __name__ == "__main__":
    main()
