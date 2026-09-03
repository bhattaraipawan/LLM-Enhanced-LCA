"""Native ipywidgets interface for the full-Colab LLM-LCA workflow.

No Gradio server/share link is used. The UI renders directly inside Google Colab.
The Qwen model is loaded once and reused for repeated BOM and single-material jobs.
"""
from __future__ import annotations

import io
import json
import pathlib
import time
import traceback
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd

WIDGET_UI_VERSION = "16.0-fresh-elcd-protocol-locked"


def _safe_stem(name: str) -> str:
    return pathlib.Path(str(name)).stem.replace(" ", "_")


def _uploaded_items(value: Any) -> list[dict[str, Any]]:
    """Normalize ipywidgets 7/8 FileUpload.value into name/content dicts."""
    if not value:
        return []
    if isinstance(value, dict):
        rows = []
        for name, meta in value.items():
            rec = dict(meta) if isinstance(meta, dict) else {"content": meta}
            rec.setdefault("name", name)
            rows.append(rec)
        return rows
    return [dict(x) for x in list(value)]


def _bytes_from_upload(rec: dict[str, Any]) -> bytes:
    content = rec.get("content", b"")
    if isinstance(content, memoryview):
        return content.tobytes()
    if isinstance(content, bytearray):
        return bytes(content)
    return content


def _gpu_info() -> tuple[str, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return "CPU only", "CUDA unavailable"
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        return name, f"CUDA available · {allocated:.1f}/{total:.1f} GB allocated"
    except Exception as exc:
        return "Unknown", f"GPU check failed: {exc}"


class NativeColabLCAApp:
    def __init__(self, work_dir: str | pathlib.Path = "/content/qwen_lca"):
        import ipywidgets as widgets
        from IPython.display import display, HTML
        import torch
        from qwen_matcher import QwenMatcher, align_formal_case_study_ids
        from external_ef_resolver import ExternalEFResolver
        from property_resolver import WebPropertyResolver
        from colab_calculate import load_factor_snapshot, process_selection_dataframe
        from material_taxonomy import classify_material, BIOGENIC_STORAGE_EXCLUDED_FAMILIES

        self.widgets = widgets
        self.display = display
        self.HTML = HTML
        self.ExternalEFResolver = ExternalEFResolver
        self.align_formal_case_study_ids = align_formal_case_study_ids
        self.WebPropertyResolver = WebPropertyResolver
        self.process_selection_dataframe = process_selection_dataframe
        self.classify_material = classify_material
        self.biogenic_families = set(BIOGENIC_STORAGE_EXCLUDED_FAMILIES)

        self.work = pathlib.Path(work_dir)
        self.snapshot = load_factor_snapshot(self.work / "process_gwp_snapshot.json")
        self.session_root = pathlib.Path("/content/LLM_LCA_Results")
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.upload_root = pathlib.Path("/content/LLM_LCA_Uploads")
        self.upload_root.mkdir(parents=True, exist_ok=True)
        # Generated only from successfully retrieved evidence; the codebase ships
        # with no material-specific GWP/density cache. Kept outside /content/qwen_lca
        # so rerunning the launcher in the same Colab runtime can reuse it.
        self.evidence_cache_path = pathlib.Path("/content/LLM_LCA_Runtime_Evidence_Cache.json")
        self.processed_log: list[dict[str, Any]] = []
        self.latest_batch_zip: pathlib.Path | None = None
        self.latest_files: list[pathlib.Path] = []
        # BOM files staged through a browser-native HTML <input type="file">.
        # This avoids both ipywidgets.FileUpload rendering problems and the
        # google.colab.files.upload() limitation inside widget callbacks.
        self.pending_uploads: list[pathlib.Path] = []

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. In Colab choose Runtime > Change runtime type > T4 GPU, then restart."
            )

        print("Loading Qwen2.5-7B-Instruct once for this Colab session...")
        self.matcher = QwenMatcher(
            self.work / "ELCD_Process_Catalog.xlsx",
            reference_map_path=self.work / "process_reference_units.json",
        ).load_model()

        # Retrieval and calculation snapshots intentionally use different semantic
        # hash schemes, but both are freshly generated from the same active ELCD
        # database and frozen into one payload. Their UUID universes must be
        # identical, and the local catalog metadata must match the factor snapshot.
        local_meta = json.loads(
            (self.work / "openlca_catalog_metadata.json").read_text(encoding="utf-8")
        )
        if str(local_meta.get("catalog_content_sha256")) != str(self.snapshot.get("catalog_content_sha256")):
            raise RuntimeError("Live catalog/factor snapshot mismatch in payload.")
        retrieval_uuids = set(self.matcher.by_uuid)
        factor_uuids = set((self.snapshot.get("factors") or {}).keys())
        if retrieval_uuids != factor_uuids:
            missing = sorted(retrieval_uuids - factor_uuids)[:10]
            extra = sorted(factor_uuids - retrieval_uuids)[:10]
            raise RuntimeError(
                "Frozen final ELCD retrieval catalog and factor snapshot do not use "
                f"the same UUID universe. Missing factors={missing}; extra factors={extra}."
            )
        print("Qwen loaded. Building native Colab interface...")

        self._build_ui()
        self._refresh_system_status()

    def _build_ui(self):
        w = self.widgets
        title = w.HTML(
            "<div style='text-align:center'>"
            "<h2 style='margin-bottom:2px'>LLM-Assisted A1–A3 Embodied Carbon Screening</h2>"
            "<div style='color:#666'>Qwen2.5 + frozen ELCD/openLCA snapshot · Native Colab interface</div>"
            "</div>"
        )
        self.system_status = w.HTML()
        self.stage = w.HTML("<b>Status:</b> Ready")
        self.file_progress = w.IntProgress(value=0, min=0, max=1, description="BOMs:", bar_style="")
        self.item_progress = w.IntProgress(value=0, min=0, max=1, description="Items:", bar_style="")
        self.timer = w.HTML("<b>Elapsed:</b> 0.0 s")

        self.geography = w.Text(value="Nepal", description="Geography:", style={"description_width":"110px"})
        settings_box = w.VBox([
            self.geography,
            w.HTML(
                "<small><b>Result design:</b> verified subtotal and complete exploratory estimate are reported separately. "
                "Final hierarchy: Class 1 ELCD Direct → Class 2 ELCD Proxy → Class 3 External Verified → Class 4 Unverified Fallback Estimate. "
                "Class 3 uses two sequential evidence phases: Phase A first searches strict target-geography/direct-product evidence (up to 60 s), then Phase B starts a fresh clock (up to 120 s) and broadens geography and technically relevant source-supported evidence. Both accepted phases are reported as External Verified while their phase/tier remains auditable. "
                "GWP is the only environmental impact indicator extracted; structured EPD GWP tables are parsed deterministically by Python first, including direct A1+A2+A3 summation. Explicit GWP-biogenic, GWP-LULUC, and fossil-component rows cannot be substituted for ordinary GWP-total, and Qwen is used only when the deterministic route cannot validate the source. "
                "Class 4 is used only after ELCD and both External Verified phases fail. It first re-searches the frozen ELCD/openLCA catalog using dynamically generated same-family technical terminology, then a dynamically inferred nearby semantic analog if needed. Only if both database-anchored routes fail may one terminal model-only value be retained. Class 4 is excluded from the verified subtotal. No material-specific GWP/property value, density, correction factor, or plausible range is encoded.<br>"
                "<b>Uncertainty:</b> GSD is assigned automatically from source uncertainty, accepted-source dispersion, or a documented pedigree method. "
                "Monte Carlo runs are selected automatically by convergence using a fixed reproducible seed.<br>"
                "<b>Biogenic policy:</b> timber, plywood, and bamboo exclude carbon-storage/sequestration credits.</small>"
            ),
        ])
        settings = w.VBox([
            w.HTML("<h3 style='margin:12px 0 4px 0'>Assessment settings</h3>"),
            settings_box,
        ], layout=w.Layout(border="1px solid #666", padding="10px", margin="6px 0"))

        # BOM upload: browser-native HTML file input + Colab kernel callback.
        # IMPORTANT: google.colab.files.upload() cannot reliably open its chooser
        # when called from an ipywidgets Button callback. A real HTML file input
        # is therefore used and file bytes are sent to Python through
        # google.colab.kernel.invokeFunction().
        self.upload_html = w.Output(layout=w.Layout(width="100%"))
        self.process_bom_btn = w.Button(
            description="PROCESS SELECTED BOM(S)",
            button_style="primary",
            icon="play",
            layout=w.Layout(width="300px", height="44px"),
        )
        self.clear_upload_btn = w.Button(description="Clear selected files", icon="trash")
        self.selected_files = w.HTML(
            "<div style='padding:8px;border:1px dashed #bbb;border-radius:6px'>"
            "<b>Selected BOMs:</b> None</div>"
        )
        self.process_bom_btn.on_click(self._on_process_boms)
        self.clear_upload_btn.on_click(self._on_clear_uploads)

        # Register browser -> Python file-transfer callback before rendering HTML.
        try:
            from google.colab import output as colab_output
            self._upload_callback_name = "llm_lca.upload_boms"
            colab_output.register_callback(self._upload_callback_name, self._receive_browser_upload)
        except Exception as exc:
            raise RuntimeError(f"Could not register Colab upload callback: {exc}") from exc

        upload_markup = r"""
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:2px 0 6px 0;">
  <input id="llm-lca-bom-file-input" type="file"
         accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
         multiple style="display:none;" />
  <button id="llm-lca-bom-upload-button" type="button"
          style="background:#36b85c;color:white;border:0;border-radius:4px;padding:12px 20px;
                 font-size:14px;font-weight:600;cursor:pointer;min-width:300px;height:44px;">
    ⬆ CHOOSE / UPLOAD EXCEL BOM(S)
  </button>
  <span id="llm-lca-bom-browser-status" style="font-size:13px;">No browser files selected.</span>
</div>
<script>
(function(){
  const input = document.getElementById('llm-lca-bom-file-input');
  const button = document.getElementById('llm-lca-bom-upload-button');
  const status = document.getElementById('llm-lca-bom-browser-status');
  if (!input || !button || !status) return;

  button.onclick = function(){
    input.click();
  };

  function toBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = String(reader.result || '');
        const comma = result.indexOf(',');
        resolve(comma >= 0 ? result.slice(comma + 1) : result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  input.onchange = async function(){
    const selected = Array.from(input.files || []);
    if (!selected.length) {
      status.textContent = 'No files selected.';
      return;
    }
    const bad = selected.filter(f => !f.name.toLowerCase().endsWith('.xlsx'));
    if (bad.length) {
      status.textContent = 'Only .xlsx files are accepted.';
      input.value = '';
      return;
    }
    button.disabled = true;
    button.style.opacity = '0.65';
    status.textContent = 'Uploading ' + selected.length + ' file(s) to the Colab runtime...';
    try {
      const payload = [];
      for (const f of selected) {
        payload.push({name: f.name, data: await toBase64(f), size: f.size});
      }
      const result = await google.colab.kernel.invokeFunction(
        'llm_lca.upload_boms', [payload], {}
      );
      let message = selected.length + ' file(s) uploaded to Colab.';
      try {
        const j = result.data['application/json'];
        if (j && j.message) message = j.message;
      } catch (e) {}
      status.textContent = message + ' Now click PROCESS SELECTED BOM(S).';
    } catch (err) {
      console.error(err);
      status.textContent = 'Upload failed: ' + (err && err.message ? err.message : String(err));
    } finally {
      button.disabled = false;
      button.style.opacity = '1';
      input.value = '';
    }
  };
})();
</script>
"""
        with self.upload_html:
            self.display(self.HTML(upload_markup))

        bom_tab = w.VBox([
            w.HTML(
                "<b>Step 1:</b> Click the green browser upload control and select one or more Excel BOMs.<br>"
                "<b>Step 2:</b> Confirm the filenames below, then click PROCESS SELECTED BOM(S).<br>"
                "<b>Required columns:</b> Material, Quantity, Unit. <b>Recommended:</b> ID."
            ),
            self.upload_html,
            self.process_bom_btn,
            self.selected_files,
            self.clear_upload_btn,
        ])

        # Single material tab
        self.item_id = w.Text(value="SINGLE-001", description="ID:")
        self.material = w.Text(placeholder="e.g., Portland cement 43 grade", description="Material:", layout=w.Layout(width="600px"))
        self.quantity = w.FloatText(value=0.0, description="Quantity:")
        self.unit = w.Text(placeholder="kg, m3, m2, Nos.", description="Unit:")
        self.density = w.FloatText(value=0.0, description="Density kg/m³:")
        self.thickness = w.FloatText(value=0.0, description="Thickness mm:")
        self.item_mass = w.FloatText(value=0.0, description="Mass/item kg:")
        self.conv_factor = w.FloatText(value=0.0, description="Conv. factor:")
        self.notes = w.Textarea(description="Notes:", layout=w.Layout(width="800px", height="70px"))
        self.process_single_btn = w.Button(description="Calculate material", button_style="primary", icon="calculator")
        self.process_single_btn.on_click(self._on_process_single)
        single_tab = w.VBox([
            w.HBox([self.item_id, self.material]),
            w.HBox([self.quantity, self.unit]),
            w.HBox([self.density, self.thickness]),
            w.HBox([self.item_mass, self.conv_factor]),
            self.notes,
            self.process_single_btn,
        ])

        bom_section = w.VBox([
            w.HTML("<h3 style=\'margin:12px 0 4px 0\'>1. Upload BOM Excel</h3>"),
            bom_tab,
        ], layout=w.Layout(border="2px solid #3a9d5d", padding="12px", margin="8px 0"))

        single_section = w.VBox([
            w.HTML("<h3 style=\'margin:12px 0 4px 0\'>2. Single Material</h3>"),
            single_tab,
        ], layout=w.Layout(border="1px solid #666", padding="12px", margin="8px 0"))

        self.download_batch_btn = w.Button(description="Download latest batch ZIP", button_style="success", icon="download", disabled=True)
        self.download_batch_btn.on_click(self._on_download_batch)
        self.result_summary = w.HTML("<b>Results:</b> Nothing processed yet.")
        self.history = w.HTML("<b>Session history:</b> None")
        self.log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="260px", overflow_y="auto"))
        self.chart_out = w.Output()

        status_box = w.VBox([
            self.system_status,
            self.stage,
            w.HBox([self.file_progress, self.item_progress, self.timer]),
        ])
        results_box = w.VBox([
            self.result_summary,
            self.download_batch_btn,
            self.history,
            w.HTML("<b>Latest contribution chart</b>"),
            self.chart_out,
            w.HTML("<b>Processing log</b>"),
            self.log,
        ])

        self.root = w.VBox([
            title,
            status_box,
            settings,
            bom_section,
            single_section,
            results_box,
        ], layout=w.Layout(width="100%"))

    def show(self):
        self.display(self.root)
        return self

    def _refresh_system_status(self):
        gpu, cuda = _gpu_info()
        count = len(self.snapshot.get("factors", {})) if isinstance(self.snapshot.get("factors"), dict) else self.snapshot.get("process_count", "?")
        self.system_status.value = (
            "<div style='padding:8px;border:1px solid #ddd;border-radius:6px'>"
            f"<b>Runtime:</b> Google Colab &nbsp; | &nbsp; <b>GPU:</b> {gpu} &nbsp; | &nbsp; "
            f"<b>CUDA:</b> {cuda} &nbsp; | &nbsp; <b>Qwen:</b> Loaded ✓ &nbsp; | &nbsp; "
            f"<b>Protocol lock:</b> Four-model Qwen settings ✓ &nbsp; | &nbsp; <b>ELCD lock:</b> Final catalog frozen ✓ &nbsp; | &nbsp; "
            f"<b>Factor snapshot:</b> Loaded ✓ &nbsp; | &nbsp; <b>Processes:</b> {count}"
            "</div>"
        )

    def _set_stage(self, text: str):
        self.stage.value = f"<b>Status:</b> {text}"
        self._refresh_system_status()

    def _log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.log:
            print(f"[{stamp}] {text}")

    def _set_busy(self, busy: bool):
        self.process_bom_btn.disabled = busy
        self.process_single_btn.disabled = busy
        self.clear_upload_btn.disabled = busy

    @staticmethod
    def _display_upload_name(path: pathlib.Path) -> str:
        # Stored names are YYYYMMDD_HHMMSS_microseconds_original.xlsx.
        parts = path.name.split("_", 3)
        return parts[3] if len(parts) == 4 else path.name

    def _refresh_selected_files(self):
        if not self.pending_uploads:
            self.selected_files.value = (
                "<div style='padding:8px;border:1px dashed #bbb;border-radius:6px'>"
                "<b>Selected BOMs:</b> None</div>"
            )
            return
        names = [self._display_upload_name(p) for p in self.pending_uploads]
        lines = "<br>".join(f"✅ {name}" for name in names)
        self.selected_files.value = (
            "<div style='padding:8px;border:1px solid #9ccc9c;border-radius:6px'>"
            f"<b>Selected BOMs ({len(names)}):</b><br>{lines}</div>"
        )

    def _receive_browser_upload(self, payload):
        """Receive .xlsx bytes sent by the browser-native Colab file input."""
        import base64

        if not isinstance(payload, list):
            raise ValueError("Upload payload must be a list of files.")
        accepted = 0
        skipped = []
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            safe_name = pathlib.Path(str(rec.get("name", ""))).name
            if not safe_name.lower().endswith(".xlsx"):
                skipped.append(safe_name or "unnamed file")
                continue
            encoded = rec.get("data") or ""
            try:
                content = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError(f"Could not decode {safe_name}: {exc}") from exc
            if not content:
                raise ValueError(f"Uploaded file is empty: {safe_name}")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = self.upload_root / f"{stamp}_{safe_name}"
            target.write_bytes(content)
            self.pending_uploads.append(target)
            accepted += 1
            self._log(f"Staged BOM in Colab runtime: {safe_name}")

        self._refresh_selected_files()
        if accepted:
            msg = f"{accepted} BOM file(s) uploaded and ready"
            self._set_stage(msg + " — click PROCESS SELECTED BOM(S)")
        else:
            msg = "No valid .xlsx BOM files were uploaded"
            self._set_stage(msg)
        if skipped:
            self._log("Skipped non-XLSX files: " + ", ".join(skipped))
        return {"accepted": accepted, "skipped": skipped, "message": msg}

    def _on_clear_uploads(self, _=None):
        self.pending_uploads.clear()
        self._refresh_selected_files()
        self._set_stage("Ready — selected BOM list cleared")

    def _save_uploads(self) -> list[pathlib.Path]:
        # Files are already staged by the browser-native HTML uploader.
        return [p for p in self.pending_uploads if p.exists() and p.suffix.lower() == ".xlsx"]

    def _match_with_progress(self, df: pd.DataFrame) -> pd.DataFrame:
        from dataclasses import asdict
        required = {"Material", "Quantity", "Unit"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"BOM is missing required columns: {missing}")
        rows = []
        records = df.to_dict(orient="records")
        self.item_progress.max = max(1, len(records))
        self.item_progress.value = 0
        for i, rec in enumerate(records, start=1):
            if not str(rec.get("ID", "")).strip() or str(rec.get("ID")).lower() == "nan":
                rec["ID"] = f"ITEM-{i:03d}"
            self._set_stage(f"Qwen process selection · item {i}/{len(records)} · {rec.get('Material','')}")
            rows.append(asdict(self.matcher.match_item(rec)))
            self.item_progress.value = i
        return pd.DataFrame(rows)

    def _apply_biogenic_reporting_policy(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Route bio-based rows away from ELCD GWP-total when storage may be included.

        The frozen openLCA snapshot uses the selected LCIA category and may include
        biogenic carbon uptake/storage. For the manuscript's harmonized screening
        comparison we exclude that storage credit for timber, plywood, and bamboo.
        These rows therefore proceed to the same live external-evidence resolver as
        rows that leave ELCD matching without an approved database factor. No material-specific factor or source is encoded;
        an accepted source must explicitly support positive A1-A3 GWP-GHG (preferred)
        or GWP-fossil. Raw Qwen/openLCA selections remain in the audit columns.
        """
        out = batch.copy()
        families = []
        overridden = []
        reasons = []
        for idx, rec in out.iterrows():
            material = rec.get("normalized_material") or rec.get("original_material") or ""
            family = self.classify_material(material)
            families.append(family)
            is_bio = family in self.biogenic_families
            overridden.append(bool(is_bio))
            if not is_bio:
                reasons.append(None)
                continue
            reasons.append(
                "Biogenic carbon storage/sequestration excluded from screening; "
                "use positive A1-A3 GWP-GHG (or GWP-fossil if explicitly documented) "
                "instead of an ELCD/openLCA GWP-total value."
            )
            if bool(rec.get("structured_output_valid", False)):
                # Preserve Qwen's Direct/Proxy/Review Required classification and
                # selected UUID. This is a reporting/factor-use policy only: the
                # ELCD GWP-total factor is bypassed so storage/sequestration credit
                # is not included in the harmonized bio-based result.
                out.at[idx, "production_approved"] = False
        out["material_family_final"] = families
        out["biogenic_reporting_override"] = overridden
        out["biogenic_reporting_policy"] = reasons
        return out

    def _resolve_and_calculate(self, df: pd.DataFrame, base_name: str) -> dict[str, Any]:
        df, id_alignment = self.align_formal_case_study_ids(df, base_name)
        if id_alignment.get("prefix"):
            self._log(
                "Benchmark ID alignment · "
                f"prefix {id_alignment['prefix']} · "
                f"aligned {id_alignment['aligned']}/{len(df)} row(s) · "
                f"changed {id_alignment['changed']} ID(s)"
            )
        self._set_stage("Qwen material normalization and process selection")
        batch = self._match_with_progress(df)
        batch = self._apply_biogenic_reporting_policy(batch)
        bio_count = int(batch.get("biogenic_reporting_override", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if bio_count:
            self._log(
                f"Biogenic reporting policy applied to {bio_count} row(s): "
                "storage/sequestration credit excluded; positive GWP-GHG/GWP-fossil evidence is sought first, with any terminal model-only estimate isolated if source-supported pathways fail."
            )

        self._set_stage("Resolving Review Required emission factors")
        ef_resolver = self.ExternalEFResolver(
            self.matcher,
            factor_snapshot=self.snapshot,
            target_geography=self.geography.value.strip() or "Nepal",
            allow_source_supported_provisional=True,
            allow_llm_unverified_estimate=True,
            allow_conservative_analog_estimate=True,
            class3_source_budget=3,
            class4_total_source_budget=10,
            max_search_queries_per_material=6,
            adaptive_max_search_queries_per_material=12,
            adaptive_total_source_budget=10,
            timeout=8,
            max_external_seconds_per_material=60.0,
            adaptive_external_seconds_per_material=120.0,
            progress_callback=self._log,
            evidence_cache_path=str(self.evidence_cache_path),
        )
        batch, ef_evidence = ef_resolver.resolve_batch(batch)

        self._set_stage("Checking and resolving only required conversion properties")
        prop_resolver = self.WebPropertyResolver(
            self.matcher,
            target_geography=self.geography.value.strip() or "Nepal",
            allow_source_supported_provisional=True,
            allow_llm_unverified_estimate=True,
            allow_conservative_analog_estimate=True,
            class3_source_budget=3,
            class4_total_source_budget=10,
            max_search_queries_per_property=5,
            adaptive_max_search_queries_per_property=10,
            adaptive_total_source_budget=9,
            timeout=8,
            max_external_seconds_per_property=60.0,
            adaptive_external_seconds_per_property=120.0,
            progress_callback=self._log,
            evidence_cache_path=str(self.evidence_cache_path),
        )
        batch, prop_evidence = prop_resolver.resolve_batch(batch)
        if "property_lookup_needed" in batch.columns:
            needed = int(batch["property_lookup_needed"].fillna(False).astype(bool).sum())
            self._log(f"Conversion-property lookup needed for {needed}/{len(batch)} row(s); direct/unit-only conversions skipped property searching for {len(batch)-needed} row(s).")

        self._set_stage("Calculating A1–A3 GWP and uncertainty")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = self.session_root / f"{base_name}_{stamp}"
        result = self.process_selection_dataframe(
            batch,
            self.snapshot,
            outdir,
            base_name,
            property_evidence=prop_evidence,
            external_evidence=ef_evidence,
        )
        return result

    def _summary_dict(self, result: dict[str, Any]) -> dict[str, Any]:
        return result["summary"].set_index("metric")["value"].to_dict()

    def _primary_gwp(self, summary: dict[str, Any]) -> tuple[float, float, bool]:
        verified = float(summary.get("Verified A1-A3 GWP subtotal (classes 1-3)", 0) or 0)
        complete_key = "Complete exploratory GWP estimate (classes 1-4)"
        partial_key = "Calculated exploratory subtotal (input/model failures remain)"
        if complete_key in summary:
            return verified, float(summary.get(complete_key, 0) or 0), True
        if partial_key in summary:
            return verified, float(summary.get(partial_key, 0) or 0), False
        return verified, 0.0, False

    def _make_batch_zip(self, result_files: list[pathlib.Path]) -> pathlib.Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.session_root / f"LLM_LCA_Batch_Results_{stamp}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in result_files:
                if p and pathlib.Path(p).exists():
                    z.write(p, arcname=pathlib.Path(p).name)
        return target

    def _show_chart(self, png: pathlib.Path | str | None):
        from IPython.display import Image, display, clear_output
        with self.chart_out:
            clear_output(wait=True)
            if png and pathlib.Path(png).exists():
                display(Image(filename=str(png)))
            else:
                print("No chart generated.")

    def _update_history(self):
        if not self.processed_log:
            self.history.value = "<b>Session history:</b> None"
            return
        df = pd.DataFrame(self.processed_log)
        self.history.value = "<b>Session history</b>" + df.to_html(index=False, border=0)

    def _on_process_boms(self, _=None):
        paths = self._save_uploads()
        if not paths:
            self._set_stage("No Excel BOM selected — click UPLOAD EXCEL BOM(S) first")
            self._log("Click the green UPLOAD EXCEL BOM(S) button, choose at least one .xlsx file, then process.")
            return
        self._set_busy(True)
        self.file_progress.max = max(1, len(paths))
        self.file_progress.value = 0
        started = time.time()
        generated: list[pathlib.Path] = []
        messages = []
        last_png = None
        try:
            for idx, path in enumerate(paths, start=1):
                bom_start = time.time()
                self._log(f"Starting {path.name} ({idx}/{len(paths)})")
                try:
                    bom = pd.read_excel(path)
                    result = self._resolve_and_calculate(bom, _safe_stem(path.name))
                    s = self._summary_dict(result)
                    files = [pathlib.Path(result[k]) for k in ("xlsx", "verified_png", "complete_png", "raw_mc", "zip") if result.get(k)]
                    generated.extend(files)
                    last_png = result.get("complete_png")
                    verified_gwp, exploratory_gwp, screening_complete = self._primary_gwp(s)
                    review_rows = int(float(s.get("Input/model failure rows", 0)))
                    self.processed_log.append({
                        "Input": path.name,
                        "Verified GWP": round(verified_gwp, 3),
                        "Exploratory GWP": round(exploratory_gwp, 3),
                        "Exploratory status": "complete" if screening_complete else "subtotal — input/model failures remain",
                        "Input/model failure rows": review_rows,
                        "Workbook": pathlib.Path(result["xlsx"]).name,
                    })
                    status_text = "complete" if screening_complete else "SUBTOTAL — input/model failures remain"
                    messages.append(
                        f"✅ {path.name}: verified {verified_gwp:,.3f}; complete exploratory {exploratory_gwp:,.3f} "
                        f"{self.snapshot.get('impact_category_ref_unit') or 'kg CO₂e'} · {status_text} · input/model failures {review_rows}"
                    )
                    self._log(f"Finished {path.name} in {time.time()-bom_start:.1f}s")
                except Exception as exc:
                    messages.append(f"❌ {path.name}: {exc}")
                    self._log(f"ERROR in {path.name}: {exc}")
                    with self.log:
                        traceback.print_exc(limit=8)
                self.file_progress.value = idx
                self.timer.value = f"<b>Elapsed:</b> {time.time()-started:.1f} s"

            if generated:
                self.latest_files = generated
                self.latest_batch_zip = self._make_batch_zip(generated)
                self.download_batch_btn.disabled = False
            self.result_summary.value = "<b>Latest batch</b><br>" + "<br>".join(messages)
            self._update_history()
            self._show_chart(last_png)
            self._set_stage("Finished. Select another BOM and process again — no model reload required.")
            self._on_clear_uploads()
        finally:
            self.timer.value = f"<b>Elapsed:</b> {time.time()-started:.1f} s"
            self._set_busy(False)
            self._refresh_system_status()

    def _optional_number(self, widget):
        try:
            v = float(widget.value)
            return None if v == 0 else v
        except Exception:
            return None

    def _on_process_single(self, _=None):
        material = self.material.value.strip()
        unit = self.unit.value.strip()
        if not material or not unit or float(self.quantity.value) == 0:
            self._set_stage("Single material requires material, non-zero quantity, and unit")
            return
        df = pd.DataFrame([{
            "ID": self.item_id.value.strip() or "SINGLE-001",
            "Material": material,
            "Quantity": float(self.quantity.value),
            "Unit": unit,
            "Density_kg_m3": self._optional_number(self.density),
            "Thickness_mm": self._optional_number(self.thickness),
            "Mass_per_item_kg": self._optional_number(self.item_mass),
            "Conversion_factor_to_ref_unit": self._optional_number(self.conv_factor),
            "Notes": self.notes.value,
        }])
        self._set_busy(True)
        started = time.time()
        try:
            result = self._resolve_and_calculate(df, "Single_" + _safe_stem(material[:50]))
            s = self._summary_dict(result)
            verified_gwp, exploratory_gwp, screening_complete = self._primary_gwp(s)
            review_rows = int(float(s.get("Input/model failure rows", 0)))
            files = [pathlib.Path(result[k]) for k in ("xlsx", "verified_png", "complete_png", "raw_mc", "zip") if result.get(k)]
            self.latest_files = files
            self.latest_batch_zip = self._make_batch_zip(files)
            self.download_batch_btn.disabled = False
            self.processed_log.append({
                "Input": f"Single: {material}",
                "Verified GWP": round(verified_gwp, 3),
                "Exploratory GWP": round(exploratory_gwp, 3),
                "Exploratory status": "complete" if screening_complete else "subtotal — input/model failures remain",
                "Input/model failure rows": review_rows,
                "Workbook": pathlib.Path(result["xlsx"]).name,
            })
            status_text = "complete" if screening_complete else "SUBTOTAL — input/model failures remain"
            self.result_summary.value = (
                f"<b>Latest result:</b> {material} · verified {verified_gwp:,.3f}; complete exploratory {exploratory_gwp:,.3f} "
                f"{self.snapshot.get('impact_category_ref_unit') or 'kg CO₂e'} · {status_text} · Input/model failures: {review_rows}"
            )
            self._update_history()
            self._show_chart(result.get("complete_png"))
            self._log(f"Finished single material: {material}")
            self._set_stage("Finished. Enter another material or switch to BOM upload.")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            with self.log:
                traceback.print_exc(limit=8)
            self._set_stage(f"Error: {exc}")
        finally:
            self.timer.value = f"<b>Elapsed:</b> {time.time()-started:.1f} s"
            self._set_busy(False)
            self._refresh_system_status()

    def _on_download_batch(self, _=None):
        if not self.latest_batch_zip or not self.latest_batch_zip.exists():
            self._log("No result ZIP is available yet.")
            return
        try:
            from google.colab import files
            self._log(f"Starting download: {self.latest_batch_zip.name}")
            files.download(str(self.latest_batch_zip))
        except Exception as exc:
            self._log(f"Download failed: {exc}")


def launch_native_widgets(work_dir: str | pathlib.Path = "/content/qwen_lca"):
    """Load Qwen once and render the persistent notebook-native UI."""
    app = NativeColabLCAApp(work_dir)
    return app.show()
