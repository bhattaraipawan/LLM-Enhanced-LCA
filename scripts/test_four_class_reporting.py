from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from property_resolver import conversion_requirements
from colab_calculate import _final_evidence_class, build_contribution_table, build_summary


def main():
    # Unit requirements remain deterministic.
    assert conversion_requirements("m3", "m3") == []
    assert conversion_requirements("kg", "t") == []
    assert conversion_requirements("m3", "kg") == ["density_kg_m3"]
    assert conversion_requirements("m2", "kg") == ["thickness_mm", "density_kg_m3"]

    # Strict and relaxed external evidence are both publication-facing Class 3.
    assert _final_evidence_class({
        "emission_factor_verification":"EXTERNAL_VERIFIED",
        "emission_factor_source_class":"EXTERNAL_VERIFIED",
        "property_verification":"NOT_NEEDED",
    }, ef_traceable=True, property_traceable=True) == "EXTERNAL_VERIFIED"
    assert _final_evidence_class({
        "emission_factor_verification":"EXTERNAL_VERIFIED",
        "emission_factor_source_class":"EXTERNAL_VERIFIED",
        "external_ef_source_class":"EXTERNAL_VERIFIED_RELAXED",
        "property_verification":"TRACEABLE_WEB_RELAXED",
    }, ef_traceable=True, property_traceable=True) == "EXTERNAL_VERIFIED"
    assert _final_evidence_class({
        "emission_factor_verification":"UNVERIFIED_FALLBACK_ESTIMATE",
        "emission_factor_source_class":"LLM_UNVERIFIED_ESTIMATE",
        "property_verification":"NOT_NEEDED",
    }, ef_traceable=False, property_traceable=True) == "UNVERIFIED_FALLBACK_ESTIMATE"

    # Chart rule: every individual material <0.1% is grouped into Other.
    results = pd.DataFrame([
        {"calculation_status":"CALCULATED","result_evidence_class":"ELCD_DIRECT","original_material":"Major","normalized_material":"Major","gwp_total":1000.0},
        {"calculation_status":"CALCULATED","result_evidence_class":"EXTERNAL_VERIFIED","original_material":"Tiny A","normalized_material":"Tiny A","gwp_total":0.4},
        {"calculation_status":"CALCULATED","result_evidence_class":"ELCD_PROXY","original_material":"Tiny B","normalized_material":"Tiny B","gwp_total":0.3},
        {"calculation_status":"CALCULATED","result_evidence_class":"UNVERIFIED_FALLBACK_ESTIMATE","original_material":"Fallback","normalized_material":"Fallback","gwp_total":2.0},
    ])
    verified, note = build_contribution_table(results, scope="verified")
    assert "Other" in set(verified["material"])
    assert abs(float(verified.loc[verified["material"]=="Other","gwp_total"].iloc[0]) - 0.7) < 1e-12
    assert "less than 0.1%" in note
    complete, _ = build_contribution_table(results, scope="complete_exploratory")
    assert "Fallback" in set(complete["material"])

    summary = build_summary(results, pd.DataFrame(columns=["scope","median","p2_5","p97_5"]), "kg CO2 eq")
    metrics = set(summary["metric"])
    assert "Verified A1-A3 GWP subtotal (classes 1-3)" in metrics
    assert "Complete exploratory GWP estimate (classes 1-4)" in metrics
    assert all("Evidence-supported" not in x for x in metrics)

    print("four-class final reporting checks: PASS")


if __name__ == "__main__":
    main()
