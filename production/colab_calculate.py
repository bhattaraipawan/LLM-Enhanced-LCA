"""Frozen-snapshot A1-A3 GWP calculation and publication-ready reporting.

Four publication-facing evidence classes are used:
1. ELCD Direct
2. ELCD Proxy
3. External Verified
4. Unverified Fallback Estimate

External Verified contains two sequential retrieval phases: a strict Nepal/direct-
product phase followed, when needed, by a fresh-clock relaxed-geography/source-
supported phase. Both phases remain auditable but are reported under Class 3.
The verified subtotal contains Classes 1-3; the complete exploratory estimate
contains Classes 1-4. Rows that cannot produce a structurally usable value remain
input/model failures and are never silently treated as zero.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unit_conversion import convert_quantity
from material_taxonomy import classify_material, BIOGENIC_STORAGE_EXCLUDED_FAMILIES, choose_resolution_material
from uncertainty import (
    UNCERTAINTY_METHOD_VERSION,
    MONTE_CARLO_SEED,
    MONTE_CARLO_SCHEDULE,
    MONTE_CARLO_MIN_FINAL_RUNS,
    MONTE_CARLO_CONVERGENCE_TOLERANCE,
    enrich_row_uncertainty,
    monte_carlo_with_convergence,
)

COLAB_CALCULATOR_VERSION = "7.0-four-class-external-verified-merged"
CHART_OTHER_THRESHOLD_PCT = 0.1


def _norm(v: Any) -> str:
    return "" if v is None else str(v).strip().lower()


def _number(v: Any):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_bool(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return _norm(v) in {"1", "true", "yes", "y"}


def _safe_name(v: str, default: str = "BOM") -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(v or "").strip()).strip("._")
    return s or default


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_factor_snapshot(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj.get("factors"), dict):
        raise ValueError("Invalid process_gwp_snapshot.json: missing factor mapping.")
    obj["_file_sha256"] = _sha256_file(p)
    return obj


def _conversion_inputs(rec: dict[str, Any]):
    density = _number(rec.get("density_kg_m3"))
    if density is None:
        density = _number(rec.get("resolved_density_kg_m3"))
    thickness = _number(rec.get("thickness_mm"))
    if thickness is None:
        thickness = _number(rec.get("resolved_thickness_mm"))
    mass_per_item = _number(rec.get("mass_per_item_kg"))
    if mass_per_item is None:
        mass_per_item = _number(rec.get("resolved_mass_per_item_kg"))
    return density, thickness, mass_per_item


def _property_traceable(rec: dict[str, Any]) -> bool:
    status = _norm(rec.get("property_verification"))
    return status not in {
        "unverified_fallback_estimate", "llm_unverified_estimate",
        "conservative_analog_estimate", "unresolved", "input_or_model_failure",
    }


def _external_relaxed_factor_safe(material: Any, ef: float, ref_unit: Any) -> tuple[bool, str]:
    """Structural guard for relaxed externally sourced factors.

    This contains no material-specific expected GWP ranges. It checks only sign
    and strong reference-unit family constraints.
    """
    name = _norm(material)
    ref = _norm(ref_unit).replace("³", "3").replace("²", "2")
    if ef <= 0:
        return False, "Non-positive relaxed-phase external A1-A3 factor rejected; credits require verified evidence."
    is_metal = any(x in name for x in (
        "steel", "iron", "nail", "binding wire", "rebar", "galvanized", "galvanised", "cgi", "corrugated"
    ))
    if is_metal and ref not in {"kg", "t", "ton", "tonne"}:
        return False, "Metal-product relaxed external factor must use a mass reference unit."
    return True, "ok"


def _final_evidence_class(row: dict[str, Any], *, ef_traceable: bool, property_traceable: bool) -> str:
    ef_ver = _norm(row.get("emission_factor_verification"))
    prop_ver = _norm(row.get("property_verification"))
    source = _norm(row.get("emission_factor_source_class"))
    external_source = _norm(row.get("external_ef_source_class"))

    fallback = {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}
    if ef_ver in fallback or prop_ver in fallback:
        return "UNVERIFIED_FALLBACK_ESTIMATE"

    # The former provisional/source-supported pathway is now the relaxed phase of
    # Class 3. A relaxed externally sourced conversion property also makes Class 3
    # the weakest publication-facing evidence level for the row.
    if (
        source == "external_verified"
        or ef_ver in {"external_verified", "provisional_source_supported"}
        or prop_ver in {"traceable_web_relaxed", "provisional_source_supported"}
        or external_source == "external_verified_relaxed"
    ):
        return "EXTERNAL_VERIFIED"

    if not ef_traceable or not property_traceable:
        return "INPUT_OR_MODEL_FAILURE"
    if source == "database_direct":
        return "ELCD_DIRECT"
    if source == "database_proxy":
        return "ELCD_PROXY"
    return "INPUT_OR_MODEL_FAILURE"


def calculate_rows(selection_df: pd.DataFrame, snapshot: dict[str, Any]) -> pd.DataFrame:
    # Qwen retrieval uses the final hash-frozen ELCD descriptor snapshot, while
    # numerical factors are exported separately from the same active openLCA
    # database. Their semantic hashes use different representations; exact process
    # UUID availability is therefore the cross-check.
    impact_unit = snapshot.get("impact_category_ref_unit") or "kg CO2 eq"
    out: list[dict[str, Any]] = []
    factors = snapshot["factors"]
    selected_uuids = {
        str(v).strip()
        for v in selection_df.get(
            "production_selected_process_uuid", pd.Series(dtype=str)
        ).dropna().astype(str)
        if str(v).strip()
    }
    missing_factors = sorted(selected_uuids - set(factors))
    if missing_factors:
        raise RuntimeError(
            "Selected frozen-catalog UUID(s) are missing from the frozen factor "
            f"snapshot: {missing_factors[:10]}"
        )

    for rec in selection_df.to_dict(orient="records"):
        row = dict(rec)
        material_for_family = choose_resolution_material(
            rec.get("original_material") or rec.get("Material"), rec.get("normalized_material")
        )[0]
        row.update({
            "calculation_status": None,
            "calculation_engine": None,
            "calculation_message": None,
            "process_reference_unit_final": None,
            "quantity_in_reference_unit": None,
            "conversion_method": None,
            "emission_factor_per_reference_unit": None,
            "emission_factor_source_class": None,
            "emission_factor_verification": None,
            "emission_factor_source_url_final": None,
            "emission_factor_source_geography_final": None,
            "emission_factor_source_year_final": None,
            "emission_factor_boundary_final": None,
            "impact_basis": None,
            "impact_unit": impact_unit,
            "gwp_total": None,
            "calculation_traceability": None,
            "emission_factor_fallback_method": None,
            "property_fallback_method_final": None,
            "unverified_fallback_method": None,
            "fully_traceable_row": False,
            "result_evidence_class": "INPUT_OR_MODEL_FAILURE",
            "included_in_verified_calculation": False,
            "included_in_evidence_supported_screening": False,
            "included_in_complete_exploratory_screening": False,
            "included_in_complete_screening": False,
            "material_family_final": classify_material(material_for_family),
            "biogenic_storage_excluded": False,
            "biogenic_reporting_indicator": None,
        })

        structured_valid = _as_bool(rec.get("structured_output_valid"))
        approved = _as_bool(rec.get("production_approved"))
        match_type = _norm(rec.get("production_match_type"))
        uid = str(rec.get("production_selected_process_uuid") or "").strip()
        ext_status = _norm(rec.get("external_ef_resolution_status"))
        ext_ready = ext_status in {"resolved_external_verified", "resolved_provisional_source_supported", "resolved_unverified_fallback_estimate", "resolved_llm_unverified_estimate", "resolved_conservative_analog_estimate"}

        if not structured_valid and not ext_ready:
            row["calculation_status"] = "REVIEW_REQUIRED_MODEL_OUTPUT"
            out.append(row)
            continue

        density, thickness, mass_per_item = _conversion_inputs(rec)
        family = row["material_family_final"]
        is_biogenic_policy = family in BIOGENIC_STORAGE_EXCLUDED_FAMILIES

        ef_traceable = False
        if approved and match_type in {"direct", "proxy"} and uid and not is_biogenic_policy:
            snap = factors.get(uid)
            if not snap or snap.get("status") != "OK":
                row["calculation_status"] = "REVIEW_REQUIRED_SNAPSHOT_FACTOR"
                row["calculation_message"] = (snap or {}).get("error") or "No usable frozen openLCA factor for selected process."
                out.append(row)
                continue
            ef = _number(snap.get("emission_factor"))
            ref_unit = snap.get("reference_unit")
            if ef is None or not ref_unit:
                row["calculation_status"] = "REVIEW_REQUIRED_SNAPSHOT_FACTOR"
                row["calculation_message"] = "Frozen factor or reference unit is missing."
                out.append(row)
                continue
            row["calculation_engine"] = "FROZEN_OPENLCA_SNAPSHOT"
            row["process_reference_unit_final"] = ref_unit
            row["emission_factor_source_class"] = "DATABASE_DIRECT" if match_type == "direct" else "DATABASE_PROXY"
            row["emission_factor_verification"] = "DATABASE_TRACEABLE"
            row["emission_factor_source_geography_final"] = rec.get("production_selected_process_location")
            row["emission_factor_boundary_final"] = "SELECTED_LCIA_OF_ELCD_PROCESS"
            row["impact_basis"] = snap.get("impact_basis")
            ef_traceable = True
        elif ext_ready:
            ef = _number(rec.get("external_ef_value"))
            ref_unit = rec.get("external_ef_reference_unit")
            if ef is None or not ref_unit:
                row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_INVALID"
                out.append(row)
                continue
            verification = _norm(rec.get("external_ef_verification"))
            if is_biogenic_policy:
                indicator = str(rec.get("external_ef_indicator") or "").upper().replace("_", "-")
                if ef <= 0:
                    row["calculation_status"] = "REVIEW_REQUIRED_BIOGENIC_FACTOR_NONPOSITIVE"
                    row["calculation_message"] = "Biogenic storage is excluded; bio-based EF must be strictly positive."
                    row["biogenic_storage_excluded"] = True
                    out.append(row)
                    continue
                if verification in {"external_verified", "provisional_source_supported", "unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                    if "GWP-GHG" not in indicator and "GWP-FOSSIL" not in indicator:
                        row["calculation_status"] = "REVIEW_REQUIRED_BIOGENIC_INDICATOR"
                        row["calculation_message"] = "Bio-based factors must explicitly report GWP-GHG or GWP-fossil when storage credit is excluded."
                        row["biogenic_storage_excluded"] = True
                        out.append(row)
                        continue
                    row["biogenic_reporting_indicator"] = rec.get("external_ef_indicator")
                else:
                    row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_INVALID"
                    row["calculation_message"] = "Unsupported external-factor verification class."
                    out.append(row)
                    continue
                row["biogenic_storage_excluded"] = True

            if verification == "external_verified":
                row["emission_factor_source_class"] = "EXTERNAL_VERIFIED"
                ef_traceable = True
            elif verification == "provisional_source_supported":  # legacy payload compatibility
                safe, msg = _external_relaxed_factor_safe(
                    rec.get("original_material") or rec.get("normalized_material"), ef, ref_unit
                )
                if not safe:
                    row["calculation_status"] = "INPUT_OR_MODEL_FAILURE_INVALID_RELAXED_EXTERNAL_EF"
                    row["calculation_message"] = msg
                    out.append(row)
                    continue
                row["emission_factor_source_class"] = "EXTERNAL_VERIFIED"
                row["emission_factor_verification"] = "EXTERNAL_VERIFIED"
                ef_traceable = True
            elif verification in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                if ef <= 0:
                    row["calculation_status"] = "INPUT_OR_MODEL_FAILURE_INVALID_LLM_ESTIMATE"
                    row["calculation_message"] = "LLM-only emission factor must be strictly positive."
                    out.append(row)
                    continue
                row["emission_factor_source_class"] = "UNVERIFIED_FALLBACK_ESTIMATE"
                ef_traceable = False
            else:
                row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_INVALID"
                row["calculation_message"] = "Unsupported external-factor verification class."
                out.append(row)
                continue

            row["calculation_engine"] = "EXTERNAL_FACTOR"
            row["process_reference_unit_final"] = ref_unit
            row["emission_factor_verification"] = ("EXTERNAL_VERIFIED" if verification in {"external_verified", "provisional_source_supported"} else rec.get("external_ef_verification"))
            row["emission_factor_fallback_method"] = rec.get("external_ef_fallback_method")
            row["emission_factor_source_url_final"] = rec.get("external_ef_source_url")
            row["emission_factor_source_geography_final"] = rec.get("external_ef_source_geography")
            row["emission_factor_source_year_final"] = rec.get("external_ef_source_year")
            row["emission_factor_boundary_final"] = rec.get("external_ef_boundary")
            row["impact_basis"] = (
                "EXTERNAL_A1_A3_GWP_GHG_NO_BIOGENIC_STORAGE" if is_biogenic_policy else "EXTERNAL_A1_A3_GWP_FACTOR"
            )
        else:
            row["calculation_status"] = "REVIEW_REQUIRED_NO_FACTOR"
            row["calculation_message"] = rec.get("safety_reason") or rec.get("external_ef_reason")
            out.append(row)
            continue

        conv = convert_quantity(
            _number(rec.get("quantity")), rec.get("unit"), ref_unit,
            density_kg_m3=density,
            thickness_mm=thickness,
            mass_per_item_kg=mass_per_item,
            conversion_factor_to_ref_unit=_number(rec.get("conversion_factor_to_ref_unit")),
        )
        row["conversion_method"] = conv.method
        if not conv.ok:
            row["calculation_status"] = "REVIEW_REQUIRED_UNIT_CONVERSION"
            row["calculation_message"] = conv.message
            out.append(row)
            continue

        row["quantity_in_reference_unit"] = float(conv.quantity_in_ref_unit)
        row["emission_factor_per_reference_unit"] = float(ef)
        row["gwp_total"] = float(conv.quantity_in_ref_unit) * float(ef)
        prop_ver = _norm(rec.get("property_verification"))
        prop_traceable = _property_traceable(rec)
        if prop_ver in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"} or _norm(row.get("emission_factor_verification")) in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
            row["calculation_traceability"] = "USES_UNVERIFIED_FALLBACK_ESTIMATE"
        elif prop_ver in {"traceable_web_relaxed", "provisional_source_supported"}:
            row["calculation_traceability"] = "USES_EXTERNAL_VERIFIED_RELAXED_PROPERTY"
        elif _norm(rec.get("external_ef_source_class")) == "external_verified_relaxed":
            row["calculation_traceability"] = "USES_EXTERNAL_VERIFIED_RELAXED_EF"
        elif not ef_traceable:
            row["calculation_traceability"] = "EXTERNAL_EVIDENCE_NOT_TRACEABLE"
        elif prop_ver in {"traceable_web", "project_or_bom", "project_input", "bom_extracted"}:
            row["calculation_traceability"] = "TRACEABLE_OR_PROJECT_PROPERTY"
        else:
            row["calculation_traceability"] = "NO_EXTERNAL_PROPERTY_NEEDED"

        row["property_fallback_method_final"] = rec.get("property_fallback_method")
        fallback_methods = [str(x) for x in (row.get("emission_factor_fallback_method"), row.get("property_fallback_method_final")) if x not in (None, "", "nan")]
        row["unverified_fallback_method"] = ";".join(dict.fromkeys(fallback_methods)) or None
        row["fully_traceable_row"] = bool(ef_traceable and prop_traceable)
        row["result_evidence_class"] = _final_evidence_class(row, ef_traceable=ef_traceable, property_traceable=prop_traceable)
        row["included_in_verified_calculation"] = row["result_evidence_class"] in {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"}
        row["included_in_evidence_supported_screening"] = row["included_in_verified_calculation"]  # legacy alias
        row["included_in_complete_exploratory_screening"] = row["result_evidence_class"] in {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED", "UNVERIFIED_FALLBACK_ESTIMATE"}
        row["included_in_complete_screening"] = row["included_in_complete_exploratory_screening"]
        row["calculation_status"] = "CALCULATED"
        out.append(row)

    return pd.DataFrame(out)


def _material_label(rec: pd.Series) -> str:
    # Paper figures prefer the concise BOM wording; normalized terminology
    # remains available in the workbook audit sheets.
    original = str(rec.get("original_material") or rec.get("Material") or "").strip()
    normalized = str(rec.get("normalized_material") or "").strip()
    return original or normalized or "Material"


def _format_pct(value: float) -> str:
    v = float(value)
    if abs(v) >= 10:
        return f"{v:.1f}%"
    if abs(v) >= 1:
        return f"{v:.2f}%"
    if abs(v) >= 0.1:
        return f"{v:.3f}%"
    return f"{v:.4f}%"


def build_contribution_table(results: pd.DataFrame, *, scope: str) -> tuple[pd.DataFrame, str]:
    calc = results[results["calculation_status"] == "CALCULATED"].copy()
    if scope == "verified":
        allowed = {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"}
        calc = calc[calc["result_evidence_class"].isin(allowed)].copy()
        scope_note = "Includes evidence classes 1-3: ELCD Direct, ELCD Proxy, and External Verified."
    elif scope == "complete_exploratory":
        allowed = {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED", "UNVERIFIED_FALLBACK_ESTIMATE"}
        calc = calc[calc["result_evidence_class"].isin(allowed)].copy()
        scope_note = "Includes evidence classes 1-4. Class 4 is the Unverified Fallback Estimate."
    else:
        raise ValueError(f"Unknown contribution scope: {scope}")

    cols = ["material", "evidence_class", "gwp_total", "contribution_pct", "cumulative_pct"]
    if calc.empty:
        return pd.DataFrame(columns=cols), scope_note

    calc["material"] = calc.apply(_material_label, axis=1)
    # Aggregate repeated rows of the same displayed material before calculating
    # chart percentages. If those rows span more than one evidence class, the
    # chart table records MIXED; the publication figure itself does not encode
    # evidence class.
    agg = (
        calc.groupby("material", as_index=False)
        .agg(
            gwp_total=("gwp_total", "sum"),
            result_evidence_class=("result_evidence_class", lambda x: next(iter(set(x))) if len(set(x)) == 1 else "MIXED"),
        )
    )
    agg = agg.sort_values("gwp_total", ascending=False).reset_index(drop=True)
    denom = float(agg["gwp_total"].sum())
    agg["contribution_pct"] = np.where(denom, agg["gwp_total"] / denom * 100.0, 0.0)

    # Publication grouping rule: every material contributing <0.1% individually
    # is aggregated into a single neutral "Other" bar. The aggregation is based
    # only on contribution magnitude, never on evidence class.
    small = agg["contribution_pct"] < CHART_OTHER_THRESHOLD_PCT
    main = agg.loc[~small].copy()
    other = agg.loc[small].copy()
    if not other.empty:
        main = pd.concat([main, pd.DataFrame([{
            "material": "Other",
            "result_evidence_class": "MIXED",
            "gwp_total": float(other["gwp_total"].sum()),
            "contribution_pct": float(other["contribution_pct"].sum()),
        }])], ignore_index=True)
    main = main.sort_values("gwp_total", ascending=False).reset_index(drop=True)
    main["cumulative_pct"] = main["contribution_pct"].cumsum()
    main = main.rename(columns={"result_evidence_class": "evidence_class"})
    grouping_note = (
        f"Materials contributing less than {CHART_OTHER_THRESHOLD_PCT:g}% individually are combined as Other; "
        f"{int(small.sum())} material group(s) were aggregated."
    )
    return main[cols], scope_note + " " + grouping_note


def make_contribution_png(contrib: pd.DataFrame, path: str | Path, impact_unit: str, *, title: str | None = None, unresolved_count: int = 0):
    """Create a compact paper-ready material-contribution chart.

    Publication figures intentionally contain no title, hatch/symbol coding, or
    evidence-class legend. Evidence classes remain available in the workbook.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if contrib.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        ax.text(0.5, 0.5, "No numerical material contributions", ha="center", va="center", fontsize=7.5)
        ax.axis("off")
    else:
        plot_df = contrib.iloc[::-1].copy()
        labels = [textwrap.fill(str(x), width=29) for x in plot_df["material"]]
        fig_h = min(max(2.8, 0.34 * len(plot_df) + 1.15), 6.7)
        fig, ax = plt.subplots(figsize=(6.2, fig_h))
        bars = ax.barh(labels, plot_df["gwp_total"])

        ax.set_xlabel(f"GWP ({impact_unit})", fontsize=7.1, labelpad=3)
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=6.3)
        ax.tick_params(axis="y", labelsize=6.4, pad=2)
        ax.grid(axis="x", alpha=0.16, linewidth=0.5)
        ax.set_axisbelow(True)

        max_abs = max(float(plot_df["gwp_total"].abs().max()), 1.0)
        for bar, pct in zip(bars, plot_df["contribution_pct"]):
            x = float(bar.get_width())
            ax.text(
                x + max_abs * 0.012, bar.get_y() + bar.get_height()/2,
                _format_pct(float(pct)), va="center", ha="left", fontsize=5.9,
            )
        ax.set_xlim(left=0, right=max_abs * 1.19)
        # No evidence-class legend/symbols. If failures exist they remain in the
        # workbook audit/summary rather than being represented as a chart symbol.
        fig.subplots_adjust(left=0.34, right=0.97, top=0.97, bottom=0.16)

    fig.savefig(p, dpi=320, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    try:
        with Image.open(p) as im:
            im.save(p, format="PNG", optimize=True, compress_level=9, dpi=(320, 320))
    except Exception:
        pass
    return p


def build_summary(results: pd.DataFrame, uncertainty: pd.DataFrame, impact_unit: str) -> pd.DataFrame:
    calc = results[results["calculation_status"] == "CALCULATED"].copy()
    verified_classes = {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"}
    verified = calc[calc["result_evidence_class"].isin(verified_classes)].copy()
    fallback = calc[calc["result_evidence_class"] == "UNVERIFIED_FALLBACK_ESTIMATE"].copy()
    failures = results[~results["calculation_status"].eq("CALCULATED")].copy()

    verified_total = float(verified["gwp_total"].sum()) if not verified.empty else 0.0
    fallback_total = float(fallback["gwp_total"].sum()) if not fallback.empty else 0.0
    exploratory_total = float(calc["gwp_total"].sum()) if not calc.empty else 0.0
    exploratory_complete = len(failures) == 0
    verified_share = (verified_total / exploratory_total * 100.0) if exploratory_total > 0 else 0.0
    numeric_coverage = (len(calc) / len(results) * 100.0) if len(results) else 0.0

    rows = [
        ["Verified A1-A3 GWP subtotal (classes 1-3)", verified_total, impact_unit],
        ["Unverified fallback GWP contribution (class 4)", fallback_total, impact_unit],
        ["Complete exploratory GWP estimate (classes 1-4)" if exploratory_complete else "Calculated exploratory subtotal (input/model failures remain)", exploratory_total, impact_unit],
        ["Verified share of complete exploratory GWP", verified_share, "%"],
        ["Numerical row coverage", numeric_coverage, "% of BOM rows"],
        ["ELCD Direct rows", int((results["result_evidence_class"] == "ELCD_DIRECT").sum()), "rows"],
        ["ELCD Proxy rows", int((results["result_evidence_class"] == "ELCD_PROXY").sum()), "rows"],
        ["External Verified rows", int((results["result_evidence_class"] == "EXTERNAL_VERIFIED").sum()), "rows"],
        ["Unverified Fallback Estimate rows", int((results["result_evidence_class"] == "UNVERIFIED_FALLBACK_ESTIMATE").sum()), "rows"],
        ["Input/model failure rows", int(len(failures)), "rows"],
        ["Exploratory screening complete", bool(exploratory_complete), "boolean"],
    ]
    for _, r in uncertainty.iterrows():
        prefix = str(r["scope"])
        rows.extend([
            [f"{prefix} Monte Carlo median", float(r["median"]), impact_unit],
            [f"{prefix} Monte Carlo 2.5th percentile", float(r["p2_5"]), impact_unit],
            [f"{prefix} Monte Carlo 97.5th percentile", float(r["p97_5"]), impact_unit],
        ])
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def _evidence_audit(results: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "item_id", "original_material", "normalized_material", "quantity", "unit",
        "production_match_type", "production_selected_process_uuid", "production_selected_process_name",
        "safety_status", "safety_reason", "external_ef_resolution_status",
        "emission_factor_source_class", "emission_factor_verification", "emission_factor_per_reference_unit",
        "process_reference_unit_final", "emission_factor_source_url_final", "emission_factor_source_geography_final",
        "emission_factor_source_year_final", "emission_factor_boundary_final", "external_ef_indicator",
        "external_ef_evidence_quote", "external_ef_reason", "external_ef_match_type", "external_ef_proxy_basis",
        "external_ef_fallback_method", "emission_factor_fallback_method",
        "property_lookup_needed", "property_lookup_reason", "property_required", "property_fallback_method", "property_fallback_method_final", "unverified_fallback_method",
        "property_verification", "property_source_url", "property_source_geography", "property_source_year",
        "conversion_method", "quantity_in_reference_unit", "result_evidence_class",
        "included_in_verified_calculation", "included_in_evidence_supported_screening",
        "included_in_complete_exploratory_screening", "included_in_complete_screening", "gwp_total",
        "factor_uncertainty_gsd", "factor_uncertainty_method", "property_uncertainty_gsd",
        "property_uncertainty_method", "row_uncertainty_gsd", "calculation_status", "calculation_message",
    ]
    cols = [c for c in preferred if c in results.columns]
    extras = [c for c in results.columns if c not in cols]
    return results[cols + extras].copy()


def _write_raw_mc_gz(raw_mc: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        raw_mc.to_csv(f, index=False)
    return path


def write_final_workbook(
    results: pd.DataFrame,
    selections: pd.DataFrame,
    property_evidence: pd.DataFrame,
    external_evidence: pd.DataFrame,
    uncertainty: pd.DataFrame,
    convergence: pd.DataFrame,
    verified_contribution: pd.DataFrame,
    complete_exploratory_contribution: pd.DataFrame,
    verified_note: str,
    complete_exploratory_note: str,
    snapshot: dict[str, Any],
    output_xlsx: str | Path,
    verified_png: str | Path,
    complete_exploratory_png: str | Path,
):
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    impact_unit = snapshot.get("impact_category_ref_unit") or "kg CO2 eq"
    summary = build_summary(results, uncertainty, impact_unit)
    calc = results[results["calculation_status"] == "CALCULATED"].copy()
    verified = calc[calc["result_evidence_class"].isin({"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"})].copy()
    complete_exploratory = results.copy()
    failures = results[~results["calculation_status"].eq("CALCULATED")].copy()
    audit = _evidence_audit(results)
    uncertainty_by_row_cols = [c for c in results.columns if c.startswith("gwp_") or "uncertainty" in c or c.startswith("pedigree_")]
    id_cols = [c for c in ("item_id", "original_material", "normalized_material", "result_evidence_class", "gwp_total") if c in results.columns]
    uncertainty_by_row = results[id_cols + [c for c in uncertainty_by_row_cols if c not in id_cols]].copy()

    metadata = pd.DataFrame([
        ["calculated_at_utc", datetime.now(timezone.utc).isoformat()],
        ["colab_calculator_version", COLAB_CALCULATOR_VERSION],
        ["catalog_content_sha256", snapshot.get("catalog_content_sha256")],
        ["factor_snapshot_sha256", snapshot.get("_file_sha256")],
        ["impact_method_name", snapshot.get("impact_method_name")],
        ["impact_method_id", snapshot.get("impact_method_id")],
        ["impact_category_name", snapshot.get("impact_category_name")],
        ["impact_category_id", snapshot.get("impact_category_id")],
        ["impact_category_ref_unit", impact_unit],
        ["result_design", "Verified subtotal (classes 1-3) and complete exploratory estimate (classes 1-4) are reported separately."],
        ["external_verified_retrieval", "Class 3 uses two sequential source-search phases: strict target-geography/direct-product first, then a fresh-clock relaxed geography/technical-equivalence phase. Both are reported as EXTERNAL_VERIFIED; phase/tier remains auditable."],
        ["external_verified_timing", "Strict phase up to 60 s; if unresolved, relaxed phase starts a fresh clock up to 120 s with its broader bounded query/source budget."],
        ["uncertainty_method_version", UNCERTAINTY_METHOD_VERSION],
        ["uncertainty_hierarchy", "source-reported quantitative uncertainty -> accepted-source dispersion -> adapted pedigree matrix; relaxed Class-3 evidence retains conservative representativeness uncertainty"],
        ["uncertainty_distribution", "Independent row-level lognormal multiplicative uncertainty; factor and required property log-variances combined."],
        ["monte_carlo_seed", MONTE_CARLO_SEED],
        ["monte_carlo_schedule", json.dumps(MONTE_CARLO_SCHEDULE)],
        ["monte_carlo_min_final_runs", MONTE_CARLO_MIN_FINAL_RUNS],
        ["monte_carlo_convergence_tolerance", MONTE_CARLO_CONVERGENCE_TOLERANCE],
        ["chart_other_threshold_pct", CHART_OTHER_THRESHOLD_PCT],
        ["verified_chart_note", verified_note],
        ["complete_exploratory_chart_note", complete_exploratory_note],
        ["final_evidence_hierarchy", "1 ELCD_DIRECT; 2 ELCD_PROXY; 3 EXTERNAL_VERIFIED; 4 UNVERIFIED_FALLBACK_ESTIMATE"],
        ["unverified_fallback_pathway", "Class 4 is attempted only after ELCD and both External Verified retrieval phases fail. It first performs a dynamic same-family search of the frozen ELCD/openLCA catalog, then a dynamically inferred semantic-analog catalog search. Qwen may generate terminology and rank supplied process identities, but Python reads numerical factors from the hash-verified frozen snapshot. Only if both database-anchored routes fail may one terminal model-only value be retained after deterministic validation. Class 4 is excluded from the verified subtotal."],
        ["biogenic_carbon_reporting_policy", "Timber/plywood/bamboo exclude biogenic storage credits; verified external GWP-GHG preferred, explicit GWP-fossil allowed."],
    ], columns=["field", "value"])

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Summary", summary),
        ("Verified_Calculation", verified),
        ("Complete_Exploratory_Screening", complete_exploratory),
        ("Evidence_Audit", audit),
        ("Uncertainty_Summary", uncertainty),
        ("Uncertainty_By_Row", uncertainty_by_row),
        ("MC_Convergence", convergence),
        ("Verified_GWP_Contribution", verified_contribution),
        ("Complete_Exploratory_GWP", complete_exploratory_contribution),
        ("Input_Model_Failures", failures),
        ("Selections_Audit", selections),
        ("Run_Metadata", metadata),
    ]
    if property_evidence is not None and not property_evidence.empty:
        sheets.append(("Property_Evidence", property_evidence))
    if external_evidence is not None and not external_evidence.empty:
        sheets.append(("External_EF_Evidence", external_evidence))

    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        for name, df in sheets:
            df.to_excel(writer, sheet_name=name, index=False)

        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#1F4E78", "border": 1, "align": "center", "valign": "vcenter"})
        num_fmt = wb.add_format({"num_format": "0.000"})
        wrap_fmt = wb.add_format({"text_wrap": True, "valign": "top"})

        for name, df in sheets:
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, max(len(df), 1), len(df.columns)-1)
            for c, col in enumerate(df.columns):
                ws.write(0, c, col, header_fmt)
                sample = [str(col)] + [str(x) for x in df[col].head(80).fillna("").tolist()]
                width = min(max(max(len(x) for x in sample) + 2, 10), 38)
                ws.set_column(c, c, width, wrap_fmt if width >= 30 else None)

        sws = writer.sheets["Summary"]
        sws.set_column("A:A", 55)
        sws.set_column("B:B", 18, num_fmt)
        sws.set_column("C:C", 18)

        # Insert figures without an added worksheet heading; the figures themselves
        # intentionally have no chart title or evidence-class symbols/legend.
        for sheet, png in [
            ("Verified_GWP_Contribution", verified_png),
            ("Complete_Exploratory_GWP", complete_exploratory_png),
        ]:
            writer.sheets[sheet].insert_image("F1", str(png), {"x_scale": 0.70, "y_scale": 0.70})

    return output_xlsx, summary


def process_selection_dataframe(
    selections: pd.DataFrame,
    snapshot: dict[str, Any],
    output_dir: str | Path,
    base_name: str,
    *,
    property_evidence: pd.DataFrame | None = None,
    external_evidence: pd.DataFrame | None = None,
):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = _safe_name(base_name)

    results = calculate_rows(selections, snapshot)
    results = enrich_row_uncertainty(results)
    uncertainty, convergence, raw_mc = monte_carlo_with_convergence(results)

    verified_contribution, verified_note = build_contribution_table(results, scope="verified")
    complete_exploratory_contribution, complete_exploratory_note = build_contribution_table(results, scope="complete_exploratory")
    impact_unit = snapshot.get("impact_category_ref_unit") or "kg CO2 eq"
    failure_count = int((results["calculation_status"] != "CALCULATED").sum())

    verified_png = make_contribution_png(
        verified_contribution,
        outdir / f"{base}_Verified_GWP_Contribution.png",
        impact_unit,
        unresolved_count=0,
    )
    complete_exploratory_png = make_contribution_png(
        complete_exploratory_contribution,
        outdir / f"{base}_Complete_Exploratory_GWP_Contribution.png",
        impact_unit,
        unresolved_count=failure_count,
    )

    xlsx, summary = write_final_workbook(
        results,
        selections,
        property_evidence if property_evidence is not None else pd.DataFrame(),
        external_evidence if external_evidence is not None else pd.DataFrame(),
        uncertainty,
        convergence,
        verified_contribution,
        complete_exploratory_contribution,
        verified_note,
        complete_exploratory_note,
        snapshot,
        outdir / f"{base}_Final_A1-A3_GWP.xlsx",
        verified_png,
        complete_exploratory_png,
    )

    raw_mc_path = _write_raw_mc_gz(raw_mc, outdir / f"{base}_Monte_Carlo_Raw.csv.gz")
    package = outdir / f"{base}_Results_Package.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in (xlsx, verified_png, complete_exploratory_png, raw_mc_path):
            z.write(path, Path(path).name)

    return {
        "results": results,
        "summary": summary,
        "uncertainty": uncertainty,
        "convergence": convergence,
        "verified_contribution": verified_contribution,
        "complete_exploratory_contribution": complete_exploratory_contribution,
        "xlsx": xlsx,
        "verified_png": verified_png,
        "complete_exploratory_png": complete_exploratory_png,
        "complete_png": complete_exploratory_png,
        "png": complete_exploratory_png,
        "raw_mc": raw_mc_path,
        "zip": package,
    }
