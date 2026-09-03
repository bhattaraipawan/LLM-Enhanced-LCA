"""Calculate final building GWP from openLCA and external A1-A3 factors.

Approved ELCD rows are calculated locally in openLCA. ELCD-unmatched rows may
use Class-3 External Verified evidence from either the strict target-geography
phase or the subsequent relaxed-geography/source-supported phase. If both fail,
Class 4 contains the explicitly labelled Unverified Fallback Estimate. Physical
unit conversions remain deterministic; any unverified conversion property is
kept outside the verified subtotal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import gzip
import json

import pandas as pd
import olca_schema as o

from .local_openlca import connect, current_catalog_hash, find_quantitative_reference
from .unit_conversion import convert_quantity
from .material_taxonomy import classify_material, BIOGENIC_STORAGE_EXCLUDED_FAMILIES, choose_resolution_material
from .uncertainty import (
    UNCERTAINTY_METHOD_VERSION,
    MONTE_CARLO_SEED,
    MONTE_CARLO_SCHEDULE,
    MONTE_CARLO_MIN_FINAL_RUNS,
    MONTE_CARLO_CONVERGENCE_TOLERANCE,
    enrich_row_uncertainty,
    monte_carlo_with_convergence,
)


def _norm(s: Any) -> str:
    return "" if s is None else str(s).strip().lower()


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return _norm(v) in {"1", "true", "yes", "y"}


def _clean_optional_number(v: Any):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str) and not v.strip():
        return None
    return float(v)



def _external_relaxed_factor_safe(material: Any, ef: float, ref_unit: Any) -> tuple[bool, str]:
    """Final local structural guard for relaxed externally sourced factors."""
    name = _norm(material)
    ref = _norm(ref_unit).replace("³", "3").replace("²", "2")
    if ef <= 0:
        return False, "Non-positive relaxed-phase external A1-A3 factor rejected; credits require verified evidence."
    is_metal = any(x in name for x in ("steel", "iron", "nail", "binding wire", "rebar", "galvanized", "galvanised", "cgi", "corrugated"))
    if is_metal and ref not in {"kg", "t", "ton", "tonne"}:
        return False, "Metal-product relaxed external factor must use a mass reference unit."
    return True, "ok"

def _process_type_name(process) -> str:
    pt = getattr(process, "process_type", None)
    value = getattr(pt, "value", None)
    if value is not None:
        return str(value)
    return "" if pt is None else str(pt)


def _is_lci_result(process) -> bool:
    return _process_type_name(process).strip().upper().endswith("LCI_RESULT")


def _is_gwp_category(category_ref) -> bool:
    name = _norm(getattr(category_ref, "name", None))
    return any(term in name for term in ("climate change", "global warming", "gwp"))


def resolve_method_and_category(client, method_query: str, category_query: str):
    methods = client.get_descriptors(o.ImpactMethod)
    method_matches = [m for m in methods if method_query.lower() in _norm(m.name) or method_query.lower() == _norm(m.id)]
    if len(method_matches) != 1:
        names = [f"{m.name} [{m.id}]" for m in method_matches[:20]]
        raise ValueError(f"Impact method query must identify exactly one method; found {len(method_matches)}: {names}")
    method_ref = method_matches[0]
    method = client.get(o.ImpactMethod, uid=method_ref.id)
    categories = getattr(method, "impact_categories", None) or []
    cat_matches = [c for c in categories if category_query.lower() in _norm(c.name) or category_query.lower() == _norm(c.id)]
    if len(cat_matches) != 1:
        names = [f"{c.name} [{c.id}]" for c in cat_matches[:30]]
        raise ValueError(f"Impact category query must identify exactly one category; found {len(cat_matches)}: {names}")
    return method_ref, cat_matches[0]


def process_reference(client, process_uuid: str):
    process = client.get(o.Process, uid=process_uuid)
    if process is None:
        raise ValueError(f"Selected process UUID not found in active openLCA database: {process_uuid}")
    qref = find_quantitative_reference(process)
    if qref is None:
        raise ValueError(f"Process has no quantitative reference exchange: {process_uuid}")
    unit = getattr(qref, "unit", None)
    flow_property = getattr(qref, "flow_property", None)
    if unit is None or not getattr(unit, "name", None):
        raise ValueError(f"Quantitative reference exchange has no unit: {process_uuid}")
    return process, qref, unit, flow_property


def _find_target_tech_flow(result, process_uuid: str, reference_flow_uuid: str | None):
    tech_flows = result.get_tech_flows()
    for tf in tech_flows:
        provider = getattr(tf, "provider", None)
        flow = getattr(tf, "flow", None)
        if getattr(provider, "id", None) == process_uuid and reference_flow_uuid and getattr(flow, "id", None) == reference_flow_uuid:
            return tf
    provider_matches = [tf for tf in tech_flows if getattr(getattr(tf, "provider", None), "id", None) == process_uuid]
    return provider_matches[0] if provider_matches else None


def calculate_ef(client, process_uuid: str, method_ref, category_ref):
    process, qref, unit, flow_property = process_reference(client, process_uuid)
    setup = o.CalculationSetup(target=o.as_ref(process), impact_method=method_ref, amount=1.0, unit=unit, flow_property=flow_property)
    result = client.calculate(setup)
    try:
        state = result.wait_until_ready()
        if getattr(state, "error", None):
            raise RuntimeError(str(state.error))
        process_type = _process_type_name(process)
        if _is_lci_result(process):
            reference_flow_uuid = getattr(getattr(qref, "flow", None), "id", None)
            target_tf = _find_target_tech_flow(result, process_uuid, reference_flow_uuid)
            if target_tf is None:
                raise RuntimeError("Could not identify selected LCI_RESULT target process; direct impact cannot be read safely.")
            value = result.get_direct_impact_of(category_ref, target_tf)
            impact_basis = "DIRECT_LCI_RESULT"
        else:
            value = result.get_total_impact_value_of(category_ref)
            impact_basis = "TOTAL_LINKED_SYSTEM"
        amount = getattr(value, "amount", None)
        if amount is None:
            raise RuntimeError("openLCA returned no amount for selected impact category.")
        return float(amount), unit.name, process.name, process_type, impact_basis
    finally:
        result.dispose()


def _conversion_inputs(rec):
    density = _clean_optional_number(rec.get("density_kg_m3"))
    if density is None:
        density = _clean_optional_number(rec.get("resolved_density_kg_m3"))
    thickness = _clean_optional_number(rec.get("thickness_mm"))
    if thickness is None:
        thickness = _clean_optional_number(rec.get("resolved_thickness_mm"))
    mass_per_item = _clean_optional_number(rec.get("mass_per_item_kg"))
    if mass_per_item is None:
        mass_per_item = _clean_optional_number(rec.get("resolved_mass_per_item_kg"))
    return density, thickness, mass_per_item


def _property_is_traceable(rec) -> bool:
    status = _norm(rec.get("property_verification"))
    return status not in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate", "unresolved", "input_or_model_failure"}


def calculate_selection_file(input_xlsx, output_xlsx, method_query, category_query, *, port=8080, allow_catalog_mismatch=False):
    inp = Path(input_xlsx)
    df = pd.read_excel(inp, sheet_name="Selections")
    if df.empty:
        raise ValueError("Selections sheet is empty.")

    property_evidence = None
    external_evidence = None
    try:
        property_evidence = pd.read_excel(inp, sheet_name="Property_Evidence")
    except Exception:
        pass
    try:
        external_evidence = pd.read_excel(inp, sheet_name="External_EF_Evidence")
    except Exception:
        pass

    client = connect(port)
    current_hash, current_count = current_catalog_hash(client)
    expected_hashes = set(df["catalog_content_sha256"].dropna().astype(str))
    if len(expected_hashes) != 1:
        raise ValueError("Selection workbook does not contain exactly one catalog_content_sha256.")
    expected_hash = next(iter(expected_hashes))
    if current_hash != expected_hash and not allow_catalog_mismatch:
        raise RuntimeError(
            "The active openLCA process catalog differs from the catalog used in Colab. "
            f"Expected {expected_hash}, current {current_hash}. Reactivate the same database/re-export/re-match, "
            "or use --allow-catalog-mismatch only after deliberate review."
        )

    method_ref, category_ref = resolve_method_and_category(client, method_query, category_query)
    category_is_gwp = _is_gwp_category(category_ref)
    ef_cache = {}
    out = []

    for rec in df.to_dict(orient="records"):
        row = dict(rec)
        row.update({
            "calculation_status": None,
            "calculation_engine": None,
            "calculation_message": None,
            "selected_process_name_live": None,
            "process_type_live": None,
            "impact_basis": None,
            "process_reference_unit_live": None,
            "quantity_in_reference_unit": None,
            "conversion_method": None,
            "conversion_property_verification": rec.get("property_verification"),
            "conversion_property_status": rec.get("property_resolution_status"),
            "conversion_property_source_url": rec.get("property_source_url"),
            "emission_factor_source_class": None,
            "emission_factor_verification": None,
            "emission_factor_source_url": None,
            "emission_factor_source_geography": None,
            "emission_factor_source_year": None,
            "emission_factor_boundary": None,
            "emission_factor_match_type": None,
            "emission_factor_proxy_basis": None,
            "emission_factor_product_identity_status": None,
            "emission_factor_product_identity_reason": None,
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
            "emission_factor_per_reference_unit": None,
            "gwp_total": None,
            "impact_unit": getattr(category_ref, "ref_unit", None),
            "impact_method_name": method_ref.name,
            "impact_method_id": method_ref.id,
            "impact_category_name": category_ref.name,
            "impact_category_id": category_ref.id,
        })

        structured_valid = _as_bool(rec.get("structured_output_valid"))
        approved = _as_bool(rec.get("production_approved"))
        effective_match_type = _norm(rec.get("production_match_type"))
        effective_uid = rec.get("production_selected_process_uuid")
        ext_status = _norm(rec.get("external_ef_resolution_status"))
        ext_ready = ext_status in {"resolved_external_verified", "resolved_provisional_source_supported", "resolved_unverified_fallback_estimate", "resolved_llm_unverified_estimate", "resolved_conservative_analog_estimate"}

        # Malformed Qwen output can never enter the ELCD path. A separately
        # verified external factor may still be used because it was resolved
        # independently from the original BOM description and audited evidence.
        if not structured_valid and not ext_ready:
            row["calculation_status"] = "REVIEW_REQUIRED_MODEL_OUTPUT"
            out.append(row)
            continue

        material_text, _material_basis = choose_resolution_material(
            rec.get("original_material") or rec.get("Material"),
            rec.get("normalized_material"),
        )
        family = classify_material(material_text)
        is_biogenic_policy = family in BIOGENIC_STORAGE_EXCLUDED_FAMILIES
        row["material_family_final"] = family
        row["biogenic_storage_excluded"] = bool(is_biogenic_policy)

        if approved and effective_match_type in {"direct", "proxy"} and effective_uid and not is_biogenic_policy:
            source_mode = "OPENLCA"
        elif ext_ready:
            source_mode = "EXTERNAL"
        else:
            row["calculation_status"] = "REVIEW_REQUIRED_NO_FACTOR"
            row["calculation_message"] = rec.get("safety_reason") or rec.get("external_ef_reason")
            out.append(row)
            continue

        density, thickness, mass_per_item = _conversion_inputs(rec)

        try:
            if source_mode == "OPENLCA":
                uid = str(effective_uid)
                if uid not in ef_cache:
                    ef_cache[uid] = calculate_ef(client, uid, method_ref, category_ref)
                ef, ref_unit, live_name, process_type, impact_basis = ef_cache[uid]
                row["selected_process_name_live"] = live_name
                row["process_type_live"] = process_type
                row["impact_basis"] = impact_basis
                row["process_reference_unit_live"] = ref_unit
                row["calculation_engine"] = "OPENLCA"
                row["emission_factor_source_class"] = "DATABASE_DIRECT" if effective_match_type == "direct" else "DATABASE_PROXY"
                row["emission_factor_verification"] = "DATABASE_TRACEABLE"
                row["emission_factor_boundary"] = "SELECTED_LCIA_OF_ELCD_PROCESS"
                ef_traceable = True
            else:
                if not category_is_gwp:
                    row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_ONLY_VALID_FOR_GWP"
                    out.append(row)
                    continue
                ef = _clean_optional_number(rec.get("external_ef_value"))
                ref_unit = rec.get("external_ef_reference_unit")
                if ef is None or not ref_unit:
                    row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_INVALID"
                    out.append(row)
                    continue
                row["process_reference_unit_live"] = ref_unit
                verification = _norm(rec.get("external_ef_verification"))
                if is_biogenic_policy:
                    indicator = str(rec.get("external_ef_indicator") or "").upper().replace("_", "-")
                    if float(ef) <= 0:
                        row["calculation_status"] = "REVIEW_REQUIRED_BIOGENIC_FACTOR_NONPOSITIVE"
                        row["calculation_message"] = "Biogenic storage excluded; bio-based EF must be strictly positive."
                        out.append(row)
                        continue
                    if verification not in {"external_verified", "provisional_source_supported", "unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                        row["calculation_status"] = "REVIEW_REQUIRED_EXTERNAL_FACTOR_INVALID"
                        row["calculation_message"] = "Unsupported external-factor verification class."
                        out.append(row)
                        continue
                    if "GWP-GHG" not in indicator and "GWP-FOSSIL" not in indicator:
                        row["calculation_status"] = "REVIEW_REQUIRED_BIOGENIC_INDICATOR"
                        row["calculation_message"] = "Bio-based factors must explicitly report GWP-GHG or GWP-fossil when storage credits are excluded."
                        out.append(row)
                        continue
                    row["impact_basis"] = "EXTERNAL_A1_A3_GWP_GHG_NO_BIOGENIC_STORAGE"
                else:
                    row["impact_basis"] = "EXTERNAL_A1_A3_GWP_FACTOR"
                row["calculation_engine"] = "EXTERNAL_FACTOR"
                if verification == "external_verified":
                    row["emission_factor_source_class"] = "EXTERNAL_VERIFIED"
                    ef_traceable = True
                elif verification == "provisional_source_supported":
                    safe, safe_message = _external_relaxed_factor_safe(
                        rec.get("original_material") or rec.get("normalized_material"),
                        float(ef), ref_unit,
                    )
                    if not safe:
                        row["calculation_status"] = "INPUT_OR_MODEL_FAILURE_INVALID_RELAXED_EXTERNAL_EF"
                        row["calculation_message"] = safe_message
                        out.append(row)
                        continue
                    row["emission_factor_source_class"] = "EXTERNAL_VERIFIED"
                    ef_traceable = True
                elif verification in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                    if float(ef) <= 0:
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
                row["emission_factor_verification"] = ("EXTERNAL_VERIFIED" if verification in {"external_verified", "provisional_source_supported"} else rec.get("external_ef_verification"))
                row["emission_factor_fallback_method"] = rec.get("external_ef_fallback_method")
                row["emission_factor_source_url"] = rec.get("external_ef_source_url")
                row["emission_factor_source_geography"] = rec.get("external_ef_source_geography")
                row["emission_factor_source_year"] = rec.get("external_ef_source_year")
                row["emission_factor_boundary"] = rec.get("external_ef_boundary")
                row["emission_factor_match_type"] = rec.get("external_ef_match_type")
                row["emission_factor_proxy_basis"] = rec.get("external_ef_proxy_basis")
                row["emission_factor_product_identity_status"] = rec.get("external_ef_product_identity_status")
                row["emission_factor_product_identity_reason"] = rec.get("external_ef_product_identity_reason")

            conv = convert_quantity(
                _clean_optional_number(rec.get("quantity")), rec.get("unit"), ref_unit,
                density_kg_m3=density, thickness_mm=thickness, mass_per_item_kg=mass_per_item,
                conversion_factor_to_ref_unit=_clean_optional_number(rec.get("conversion_factor_to_ref_unit")),
            )
            row["conversion_method"] = conv.method
            if not conv.ok:
                row["calculation_status"] = "REVIEW_REQUIRED_UNIT_CONVERSION"
                row["calculation_message"] = conv.message
                out.append(row)
                continue

            row["quantity_in_reference_unit"] = conv.quantity_in_ref_unit
            row["emission_factor_per_reference_unit"] = ef
            row["gwp_total"] = float(conv.quantity_in_ref_unit) * float(ef)

            prop_ver = _norm(rec.get("property_verification"))
            if prop_ver in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"} or _norm(row.get("emission_factor_verification")) in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                row["calculation_traceability"] = "USES_UNVERIFIED_FALLBACK_ESTIMATE"
            elif prop_ver in {"traceable_web_relaxed", "provisional_source_supported"}:
                row["calculation_traceability"] = "USES_EXTERNAL_VERIFIED_RELAXED_PROPERTY"
            elif _norm(rec.get("external_ef_source_class")) == "external_verified_relaxed":
                row["calculation_traceability"] = "USES_EXTERNAL_VERIFIED_RELAXED_EF"
            elif prop_ver in {"traceable_web", "project_or_bom", "project_input", "bom_extracted"}:
                row["calculation_traceability"] = "TRACEABLE_OR_PROJECT_PROPERTY"
            else:
                row["calculation_traceability"] = "NO_EXTERNAL_PROPERTY_NEEDED"

            row["property_fallback_method_final"] = rec.get("property_fallback_method")
            fallback_methods = [str(x) for x in (row.get("emission_factor_fallback_method"), row.get("property_fallback_method_final")) if x not in (None, "", "nan")]
            row["unverified_fallback_method"] = ";".join(dict.fromkeys(fallback_methods)) or None
            property_traceable = _property_is_traceable(rec)
            row["fully_traceable_row"] = bool(ef_traceable and property_traceable)
            ef_ver = _norm(row.get("emission_factor_verification"))
            if ef_ver in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"} or prop_ver in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
                evidence_class = "UNVERIFIED_FALLBACK_ESTIMATE"
            elif source_mode == "EXTERNAL" or prop_ver in {"traceable_web_relaxed", "provisional_source_supported"} or _norm(rec.get("external_ef_source_class")) == "external_verified_relaxed":
                evidence_class = "EXTERNAL_VERIFIED"
            elif not ef_traceable or not property_traceable:
                evidence_class = "INPUT_OR_MODEL_FAILURE"
            elif source_mode == "OPENLCA" and effective_match_type == "direct":
                evidence_class = "ELCD_DIRECT"
            elif source_mode == "OPENLCA" and effective_match_type == "proxy":
                evidence_class = "ELCD_PROXY"
            else:
                evidence_class = "INPUT_OR_MODEL_FAILURE"
            row["result_evidence_class"] = evidence_class
            row["included_in_verified_calculation"] = evidence_class in {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"}
            row["included_in_evidence_supported_screening"] = row["included_in_verified_calculation"]  # legacy alias
            row["included_in_complete_exploratory_screening"] = evidence_class in {"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED", "UNVERIFIED_FALLBACK_ESTIMATE"}
            row["included_in_complete_screening"] = row["included_in_complete_exploratory_screening"]
            row["calculation_status"] = "CALCULATED"
        except Exception as exc:
            row["calculation_status"] = "REVIEW_REQUIRED_CALCULATION"
            row["calculation_message"] = str(exc)
        out.append(row)

    results = pd.DataFrame(out)
    results = enrich_row_uncertainty(results)
    uncertainty, convergence, raw_mc = monte_carlo_with_convergence(results)

    calculated = results[results["calculation_status"] == "CALCULATED"].copy()
    verified = calculated[calculated["result_evidence_class"].isin({"ELCD_DIRECT", "ELCD_PROXY", "EXTERNAL_VERIFIED"})].copy()
    fallback = calculated[calculated["result_evidence_class"] == "UNVERIFIED_FALLBACK_ESTIMATE"].copy()
    failures = results[~results["calculation_status"].eq("CALCULATED")].copy()

    impact_unit = getattr(category_ref, "ref_unit", None)
    verified_total = float(verified["gwp_total"].sum()) if not verified.empty else 0.0
    fallback_total = float(fallback["gwp_total"].sum()) if not fallback.empty else 0.0
    exploratory_total = float(calculated["gwp_total"].sum()) if not calculated.empty else 0.0
    failure_count = int(len(failures))
    exploratory_complete = failure_count == 0
    traceable_share = (verified_total / exploratory_total * 100.0) if exploratory_total > 0 else 0.0
    numeric_coverage = (len(calculated) / len(results) * 100.0) if len(results) else 0.0

    summary_rows = [
        ["Verified A1-A3 GWP subtotal (classes 1-3)", verified_total, impact_unit],
        ["Unverified fallback GWP contribution (class 4)", fallback_total, impact_unit],
        ["Complete exploratory GWP estimate (classes 1-4)" if exploratory_complete else "Calculated exploratory subtotal (input/model failures remain)", exploratory_total, impact_unit],
        ["Verified share of complete exploratory GWP", traceable_share, "%"],
        ["Numerical row coverage", numeric_coverage, "% of BOM rows"],
        ["ELCD Direct rows", int((results["result_evidence_class"] == "ELCD_DIRECT").sum()), "rows"],
        ["ELCD Proxy rows", int((results["result_evidence_class"] == "ELCD_PROXY").sum()), "rows"],
        ["External Verified rows", int((results["result_evidence_class"] == "EXTERNAL_VERIFIED").sum()), "rows"],
        ["Unverified Fallback Estimate rows", int((results["result_evidence_class"] == "UNVERIFIED_FALLBACK_ESTIMATE").sum()), "rows"],
        ["Input/model failure rows", failure_count, "rows"],
        ["Exploratory screening complete", bool(exploratory_complete), "boolean"],
    ]
    for _, r in uncertainty.iterrows():
        scope = str(r["scope"])
        prefix = scope
        summary_rows.extend([
            [f"{prefix} Monte Carlo median", float(r["median"]), impact_unit],
            [f"{prefix} Monte Carlo 2.5th percentile", float(r["p2_5"]), impact_unit],
            [f"{prefix} Monte Carlo 97.5th percentile", float(r["p97_5"]), impact_unit],
        ])
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "unit"])

    metadata = pd.DataFrame([
        ["calculated_at_utc", datetime.now(timezone.utc).isoformat()],
        ["active_catalog_content_sha256", current_hash],
        ["active_process_count", current_count],
        ["selection_catalog_content_sha256", expected_hash],
        ["catalog_match", current_hash == expected_hash],
        ["impact_method_name", method_ref.name],
        ["impact_method_id", method_ref.id],
        ["impact_category_name", category_ref.name],
        ["impact_category_id", category_ref.id],
        ["impact_category_ref_unit", impact_unit],
        ["result_design", "Verified subtotal (classes 1-3) and complete exploratory estimate (classes 1-4) are reported separately."],
        ["uncertainty_method_version", UNCERTAINTY_METHOD_VERSION],
        ["uncertainty_hierarchy", "source-reported quantitative uncertainty -> accepted-source dispersion -> adapted pedigree matrix; relaxed Class-3 source-supported evidence uses source dispersion when available and conservative pedigree otherwise"],
        ["uncertainty_distribution", "Independent factor/property lognormal uncertainty propagated at row level."],
        ["monte_carlo_seed", MONTE_CARLO_SEED],
        ["monte_carlo_schedule", json.dumps(MONTE_CARLO_SCHEDULE)],
        ["monte_carlo_min_final_runs", MONTE_CARLO_MIN_FINAL_RUNS],
        ["monte_carlo_convergence_tolerance", MONTE_CARLO_CONVERGENCE_TOLERANCE],
        ["external_factor_required_boundary", "A1-A3"],
        ["lci_result_calculation_rule", "DIRECT characterized impact of selected target process"],
        ["unit_process_calculation_rule", "TOTAL linked-system characterized impact"],
        ["final_evidence_hierarchy", "1 ELCD_DIRECT; 2 ELCD_PROXY; 3 EXTERNAL_VERIFIED; 4 UNVERIFIED_FALLBACK_ESTIMATE"],
    ], columns=["field", "value"])

    evidence_cols = [c for c in [
        "item_id", "original_material", "normalized_material", "quantity", "unit",
        "production_match_type", "production_selected_process_uuid", "production_selected_process_name",
        "safety_status", "safety_reason", "external_ef_resolution_status",
        "emission_factor_source_class", "emission_factor_verification", "emission_factor_per_reference_unit",
        "process_reference_unit_live", "emission_factor_source_url", "emission_factor_source_geography",
        "emission_factor_source_year", "emission_factor_boundary", "external_ef_indicator",
        "external_ef_evidence_quote", "external_ef_reason", "external_ef_fallback_method", "emission_factor_fallback_method",
        "property_lookup_needed", "property_lookup_reason", "property_required", "property_fallback_method", "property_fallback_method_final", "unverified_fallback_method",
        "property_verification", "property_source_url",
        "conversion_method", "quantity_in_reference_unit", "result_evidence_class",
        "included_in_verified_calculation", "included_in_evidence_supported_screening",
        "included_in_complete_exploratory_screening", "included_in_complete_screening", "gwp_total",
        "factor_uncertainty_gsd", "factor_uncertainty_method", "property_uncertainty_gsd",
        "property_uncertainty_method", "row_uncertainty_gsd", "calculation_status", "calculation_message",
    ] if c in results.columns]
    evidence_audit = results[evidence_cols + [c for c in results.columns if c not in evidence_cols]].copy()
    uncertainty_cols = [c for c in results.columns if c.startswith("gwp_") or "uncertainty" in c or c.startswith("pedigree_")]
    id_cols = [c for c in ("item_id", "original_material", "normalized_material", "result_evidence_class", "gwp_total") if c in results.columns]
    uncertainty_by_row = results[id_cols + [c for c in uncertainty_cols if c not in id_cols]].copy()

    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    raw_mc_path = output_xlsx.with_name(output_xlsx.stem + "_Monte_Carlo_Raw.csv.gz")
    with gzip.open(raw_mc_path, "wt", encoding="utf-8", newline="") as f:
        raw_mc.to_csv(f, index=False)

    sheets = [
        ("Summary", summary),
        ("Verified_Calculation", verified),
        ("Complete_Exploratory_Screening", results),
        ("Evidence_Audit", evidence_audit),
        ("Uncertainty_Summary", uncertainty),
        ("Uncertainty_By_Row", uncertainty_by_row),
        ("MC_Convergence", convergence),
        ("Input_Model_Failures", failures),
        ("Selections_Audit", df),
        ("Run_Metadata", metadata),
    ]
    if property_evidence is not None and not property_evidence.empty:
        sheets.append(("Property_Evidence", property_evidence))
    if external_evidence is not None and not external_evidence.empty:
        sheets.append(("External_EF_Evidence", external_evidence))

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
        from openpyxl.styles import Font
        for name, _frame in sheets:
            ws = writer.book[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)

    return output_xlsx, results, summary
