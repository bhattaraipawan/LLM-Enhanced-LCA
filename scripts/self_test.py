"""Fast deterministic regression checks for the guarded four-class workflow.
No model download, web access, or openLCA server is required.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from guardrails import emission_factor_plausible, property_plausible, emission_factor_cap, property_range
from evidence_consensus import robust_model_ensemble_consensus
from external_ef_resolver import ExternalEFResolver, _target_geography_supported as ef_target_geo_ok
from property_resolver import WebPropertyResolver, _target_geography_supported as prop_target_geo_ok
from colab_calculate import calculate_rows, build_contribution_table, build_summary
from uncertainty import enrich_row_uncertainty, monte_carlo_with_convergence


def ef_candidate(v, *, analog=None, unit="kg"):
    # v is supplied in the requested unit basis by the synthetic matcher.
    d={"found":True,"central_value":v,"lower_value":v*0.7,"upper_value":v*1.3,
       "reference_unit":unit,"boundary":"A1-A3","indicator":"GWP-total",
       "geography_assumption":"global","estimation_basis":"independent engineering lens","rationale":"unit basis checked"}
    if analog is None: d["product_interpretation"]="requested construction product"
    else: d["analog_material"]=analog
    return d


def _raw_intensity_from_per_kg(v, unit):
    # Synthetic test conversion only; production code does not use these values.
    if unit == "kg": return v
    if unit == "g": return v / 1000.0
    if unit == "t": return v * 1000.0
    raise AssertionError(unit)

class TerminalSingleMatcher:
    def generate_with_system(self, system, payload, **kwargs):
        if "ONE terminal exploratory A1-A3 GWP estimate" in system:
            unit=payload.get("requested_reference_unit","kg")
            fam=str(payload.get("deterministic_material_family", "UNKNOWN"))
            if fam == "GALVANIZED_FLAT_STEEL":
                interpretation="corrugated galvanized iron roofing sheet made from zinc-coated steel sheet"
                rationale="synthetic ferrous zinc-coated roofing product; offline regression test"
            else:
                interpretation="requested construction product"
                rationale="synthetic requested product; offline regression test"
            d=ef_candidate(2.0, unit=unit)
            d["product_interpretation"]=interpretation
            d["rationale"]=rationale
            return json.dumps(d)
        if "last numerical fallback" in system.lower():
            unit=payload.get("requested_reference_unit","kg")
            return json.dumps({
                "found":True,"central_value":2.0,"reference_unit":unit,
                "boundary":"A1-A3","indicator":"GWP-total",
                "product_interpretation":"requested construction product",
                "rationale":"synthetic parser-recovery test"})
        raise AssertionError("unexpected prompt")


def prop_candidate(v, analog=None):
    d={"found":True,"central_value":v,"lower_value":v*0.8,"upper_value":v*1.2,"unit":"kg/m3",
       "estimation_basis":"independent engineering lens","rationale":"bulk-volume basis checked"}
    if analog is None: d["material_interpretation"]="requested bulk material"
    else: d["analog_material"]=analog
    return d

class PropertyEnsembleMatcher:
    def generate_with_system(self, system, payload, **kwargs):
        if "ONE independent exploratory physical-property estimate" in system:
            vals=[1500,1600,1700,100000,160000]
            return json.dumps(prop_candidate(vals[int(payload.get("candidate_index",1))-1]))
        raise AssertionError("unexpected prompt")

class ProvisionalModuleMatcher:
    def generate_with_system(self, system, payload, **kwargs):
        c=payload["evidence_candidates"][0]
        return json.dumps({"records":[{
            "source_result_id":c["result_id"],"found":True,"value_mode":"SUM_A1_A2_A3",
            "impact_value":None,"a1_value":0.10,"a2_value":0.02,"a3_value":0.30,
            "impact_unit":"kg CO2e","declared_quantity":1.0,"declared_unit":"kg",
            "boundary":"A1-A3 product stage","indicator":"GWP-total","source_year":"2025",
            "evidence_quote":"Generic panel EPD Declared unit 1 kg A1 0.10 kg CO2e A2 0.02 kg CO2e A3 0.30 kg CO2e GWP-total product stage A1-A3 2025",
            "reason":"explicit module values"}]})


def main():
    # There are no material-specific numerical caps/ranges anymore.
    assert emission_factor_cap("any material", "kg") is None
    assert property_range("any material", "density_kg_m3") is None
    assert emission_factor_plausible("Stonecrete block", 300.0, "kg")[0]  # magnitude is NOT judged here
    assert property_plausible("Natural Gravel", "density_kg_m3", 160000.0)[0]

    # Class 3 geography is strict; non-target/global evidence is reserved for Class 4.
    nepal_src={"title":"Nepal construction product EPD","url":"https://example.com/epd","snippet":"Product manufactured in Nepal. A1-A3 GWP.","excerpt":""}
    india_src={"title":"Construction product EPD","url":"https://example.in/epd","snippet":"Product manufactured in India. A1-A3 GWP.","excerpt":""}
    global_src={"title":"Global construction dataset","url":"https://example.com/data","snippet":"Worldwide average product-stage data.","excerpt":""}
    assert ef_target_geo_ok(nepal_src,"Nepal")
    assert not ef_target_geo_ok(india_src,"Nepal")
    assert not ef_target_geo_ok(global_src,"Nepal")
    assert prop_target_geo_ok(nepal_src,"Nepal")
    assert not prop_target_geo_ok(india_src,"Nepal")

    # Class 4 may sum explicit A1+A2+A3 source values without inventing a factor.
    r4=ExternalEFResolver(ProvisionalModuleMatcher())
    cands=[{"result_id":"S1","title":"Generic panel EPD","url":"https://example.com/panel-epd",
        "snippet":"Generic panel EPD Declared unit 1 kg A1 0.10 kg CO2e A2 0.02 kg CO2e A3 0.30 kg CO2e GWP-total product stage A1-A3 2025",
        "excerpt":"Generic panel EPD Declared unit 1 kg A1 0.10 kg CO2e A2 0.02 kg CO2e A3 0.30 kg CO2e GWP-total product stage A1-A3 2025",
        "query":"Generic panel","tier":"GLOBAL","match_type":"DIRECT_PRODUCT","proxy_basis":None}]
    p4=r4._extract_relaxed_candidates("Generic panel","GLOBAL",cands)
    assert len(p4)==1 and abs(float(p4[0]["ef_value"])-0.42)<1e-12
    assert p4[0]["calculation_basis"]=="PYTHON_SUM_OF_EXPLICIT_SOURCE_A1_A2_A3"

    # Generic log-space consensus removes high-disagreement values.
    c=robust_model_ensemble_consensus([0.3,0.4,0.5,100,300])
    assert c.accepted and abs(c.central_value-0.4)<1e-12 and c.retained_count==3
    c2=robust_model_ensemble_consensus([0.1,1,10,100,1000])
    assert not c2.accepted

    # Class 4: after source/database routes fail, one terminal model-only value may
    # be retained, but it remains explicitly unverified and uses no repeated-value
    # consensus as validation. The synthetic CGI interpretation must match the
    # deterministic galvanized-flat-steel family.
    r=ExternalEFResolver(TerminalSingleMatcher()); r._resolve_one=lambda *a,**k:(None,[])
    out,_=r.resolve_record({"ID":"1","Material":"CGI Sheets","Quantity":1,"Unit":"kg","production_approved":False,"structured_output_valid":True})
    assert out["external_ef_resolution_status"]=="RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE"
    assert abs(float(out["external_ef_value"])-2.0)<1e-12
    assert "TERMINAL_SINGLE_LLM_VALUE_SEMANTIC_VETO" in str(out["external_ef_fallback_method"])
    assert "INTERPRETATION_FIRST_IDENTITY_PASS" in str(out["external_ef_guardrail_status"])

    # Density/property uses the identical material-agnostic ensemble principle.
    p=WebPropertyResolver(PropertyEnsembleMatcher()); p._resolve_one_property=lambda *a,**k:(None,[])
    prop,_=p.resolve_record({"ID":"3","Material":"unknown granular material","Quantity":1,"Unit":"m3",
                             "production_approved":True,"production_selected_process_ref_unit":"kg"})
    assert prop["property_resolution_status"]=="RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE"
    assert abs(float(prop["resolved_density_kg_m3"])-1600.0)<1e-12

    # Class 4 remains exploratory only.
    sel=pd.DataFrame([{"catalog_content_sha256":"abc","item_id":"x","original_material":"sheet product",
        "normalized_material":"sheet product","quantity":10.0,"unit":"kg","structured_output_valid":True,
        "production_approved":False,"production_match_type":"review required",
        "external_ef_resolution_status":"RESOLVED_UNVERIFIED_FALLBACK_ESTIMATE","external_ef_value":2.0,
        "external_ef_reference_unit":"kg","external_ef_verification":"UNVERIFIED_FALLBACK_ESTIMATE",
        "external_ef_indicator":"GWP-total","external_ef_lower_value":1.0,"external_ef_upper_value":4.0,
        "property_verification":"NOT_NEEDED"}])
    snap={"catalog_content_sha256":"abc","impact_category_ref_unit":"kg CO2 eq","factors":{}}
    rows=calculate_rows(sel,snap)
    assert rows.iloc[0]["result_evidence_class"]=="UNVERIFIED_FALLBACK_ESTIMATE"
    assert bool(rows.iloc[0]["included_in_complete_exploratory_screening"])
    assert not bool(rows.iloc[0]["included_in_evidence_supported_screening"])
    rows=enrich_row_uncertainty(rows); unc,_,_=monte_carlo_with_convergence(rows)
    _,note=build_contribution_table(rows,scope="complete_exploratory")
    assert "classes 1-4" in note
    summary=build_summary(rows,unc,"kg CO2 eq").set_index("metric")["value"].to_dict()
    assert int(summary["Unverified Fallback Estimate rows"])==1
    print("Guarded four-class production self-test: PASS")

if __name__=="__main__": main()
