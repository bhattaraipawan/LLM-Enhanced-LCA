"""Transparent uncertainty assignment and Monte Carlo propagation for LLM-LCA.

The implementation follows a hierarchy suitable for screening LCA:
1. source-reported quantitative uncertainty when it is explicitly supported;
2. empirical dispersion across multiple independently accepted factors for the
   same product/tier/reference unit;
3. an adapted ecoinvent pedigree-matrix calculation when quantitative source
   uncertainty is unavailable;
4. source-supported provisional multi-source dispersion when available;
5. otherwise a conservative adapted pedigree-matrix estimate.

The ecoinvent pedigree matrix expresses additional uncertainty as variances of
an underlying normal distribution for lognormal data.  These variances are added
and converted to a geometric standard deviation (GSD) with exp(sqrt(variance)).
For characterized construction-product factors, a small basic log-variance of
0.0006 is used as a transparent screening minimum, adapted from the ecoinvent
basic uncertainty value for semi-finished products.  This is an uncertainty
model for this study; it is not claimed to be uncertainty metadata supplied by
ELCD.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd

UNCERTAINTY_METHOD_VERSION = "1.1"
STUDY_REFERENCE_YEAR = 2026
TARGET_GEOGRAPHY = "Nepal"
BASIC_LOG_VARIANCE = 0.0006
MONTE_CARLO_SEED = 42
MONTE_CARLO_SCHEDULE = (1000, 5000, 10000, 25000, 50000)
MONTE_CARLO_MIN_FINAL_RUNS = 10000
MONTE_CARLO_CONVERGENCE_TOLERANCE = 0.01

# ecoinvent v3 Data Quality Guidelines, Table 10.5: variances of the
# underlying normal distributions used for pedigree uncertainty.
PEDIGREE_VARIANCES = {
    "reliability": {1: 0.0000, 2: 0.0006, 3: 0.0020, 4: 0.0080, 5: 0.0400},
    "completeness": {1: 0.0000, 2: 0.0001, 3: 0.0006, 4: 0.0020, 5: 0.0080},
    "temporal": {1: 0.0000, 2: 0.0002, 3: 0.0020, 4: 0.0080, 5: 0.0400},
    "geographical": {1: 0.0000, 2: 0.000025, 3: 0.0001, 4: 0.0006, 5: 0.0020},
    "technological": {1: 0.0000, 2: 0.0006, 3: 0.0080, 4: 0.0400, 5: 0.1200},
}


def _clean(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _norm(v: Any) -> str:
    return _clean(v).lower()


def _num(v: Any) -> float | None:
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
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def gsd_from_cv(cv: float) -> float | None:
    """Convert coefficient of variation (fraction, not percent) to lognormal GSD."""
    cv = float(cv)
    if cv < 0 or not math.isfinite(cv):
        return None
    sigma = math.sqrt(math.log1p(cv * cv))
    return math.exp(sigma)


def gsd_from_interval(lower: float, upper: float, confidence: float = 0.95) -> float | None:
    """Derive lognormal GSD from a central multiplicative confidence interval."""
    lower = float(lower)
    upper = float(upper)
    confidence = float(confidence)
    if lower <= 0 or upper <= 0 or lower >= upper or not (0 < confidence < 1):
        return None
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    if z <= 0:
        return None
    sigma = (math.log(upper) - math.log(lower)) / (2.0 * z)
    return math.exp(max(sigma, 0.0))


def gsd_from_peer_values(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and float(v) > 0 and math.isfinite(float(v))]
    if len(vals) < 2:
        return None
    logs = np.log(np.asarray(vals, dtype=float))
    sigma = float(np.std(logs, ddof=1))
    if not math.isfinite(sigma):
        return None
    return math.exp(max(sigma, 0.0))


def _parse_json_numbers(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        seq = value
    else:
        s = _clean(value)
        if not s:
            return []
        try:
            seq = json.loads(s)
        except Exception:
            return []
    out: list[float] = []
    for x in seq:
        if isinstance(x, dict):
            x = x.get("ef_value")
        n = _num(x)
        if n is not None and n > 0:
            out.append(n)
    return out


def _temporal_score(year: Any, study_year: int = STUDY_REFERENCE_YEAR) -> int:
    y = _num(year)
    if y is None:
        return 5
    diff = abs(int(study_year) - int(y))
    if diff < 3:
        return 1
    if diff < 6:
        return 2
    if diff < 10:
        return 3
    if diff < 15:
        return 4
    return 5


def _geography_score(geography: Any, target: str = TARGET_GEOGRAPHY) -> int:
    g = _norm(geography)
    t = _norm(target)
    if not g:
        return 5
    if t and t in g:
        return 1
    if "south asia" in g:
        return 2
    if any(x in g for x in ("india", "bangladesh", "sri lanka", "pakistan", "bhutan")):
        return 3
    if "asia" in g or "global" in g or "world" in g:
        return 4
    return 5


def pedigree_scores_for_row(rec: dict[str, Any]) -> tuple[dict[str, int], str]:
    """Assign transparent pedigree scores from evidence metadata.

    The mapping intentionally avoids material-specific numerical expectations.
    It reflects evidence quality/representativeness only.
    """
    source = _norm(rec.get("emission_factor_source_class"))
    verification = _norm(rec.get("emission_factor_verification"))
    external_source_class = _norm(rec.get("external_ef_source_class"))
    match_type = _norm(rec.get("production_match_type"))
    external_match = _norm(rec.get("external_ef_match_type"))

    relaxed_external = (source == "provisional_source_supported" or verification == "provisional_source_supported" or external_source_class == "external_verified_relaxed")
    if relaxed_external:
        year = rec.get("emission_factor_source_year_final") or rec.get("external_ef_source_year")
        geography = rec.get("emission_factor_source_geography_final") or rec.get("external_ef_source_geography")
        scores = {
            "reliability": 4,
            "completeness": 4,
            "temporal": _temporal_score(year),
            "geographical": _geography_score(geography),
            "technological": 4,
        }
        return scores, (
            "External Verified relaxed-phase factor: conservative reliability/completeness/technology "
            "scores with temporal/geographic representativeness derived from retrieved evidence metadata."
        )

    # Reliability: verified program/database evidence is strongest; sources that
    # are traceable but not independently program-verified are one level lower.
    if source.startswith("database_"):
        reliability = 2
    elif external_source_class in {"verified_epd_program", "government", "nepal_government", "technical_lca_database"}:
        reliability = 1
    elif external_source_class in {"manufacturer_epd", "peer_reviewed_publication", "academic"}:
        reliability = 2
    else:
        reliability = 3

    # Completeness cannot usually be inferred quantitatively from web snippets.
    # Database direct values receive score 2, proxies 3; product-specific external
    # declarations are conservatively treated as one/few-site evidence (score 4).
    if source == "database_direct":
        completeness = 2
    elif source == "database_proxy":
        completeness = 3
    elif source == "external_verified":
        completeness = 4
    else:
        completeness = 4

    year = rec.get("emission_factor_source_year_final") or rec.get("external_ef_source_year")
    temporal = _temporal_score(year)

    geography = (
        rec.get("emission_factor_source_geography_final")
        or rec.get("external_ef_source_geography")
        or rec.get("production_selected_process_location")
    )
    geographical = _geography_score(geography)

    if source == "database_direct":
        technological = 1
    elif source == "database_proxy":
        technological = 3
    elif source == "external_verified":
        technological = 4 if external_match == "product_proxy" else 2
    else:
        technological = 4 if match_type == "proxy" else 3

    scores = {
        "reliability": reliability,
        "completeness": completeness,
        "temporal": temporal,
        "geographical": geographical,
        "technological": technological,
    }
    reason = (
        f"Pedigree scores derived from source class={source or verification or 'unknown'}, "
        f"external source class={external_source_class or 'n/a'}, year={_clean(year) or 'unknown'}, "
        f"geography={_clean(geography) or 'unknown'}, process/external match={external_match or match_type or 'unknown'}."
    )
    return scores, reason


def pedigree_gsd(scores: dict[str, int], basic_variance: float = BASIC_LOG_VARIANCE) -> tuple[float, float]:
    variance = float(basic_variance)
    for key in ("reliability", "completeness", "temporal", "geographical", "technological"):
        score = int(scores[key])
        variance += PEDIGREE_VARIANCES[key][score]
    sigma = math.sqrt(max(variance, 0.0))
    return math.exp(sigma), variance


def _factor_gsd(rec: dict[str, Any]) -> tuple[float, str, str, dict[str, int] | None]:
    verification = _norm(rec.get("emission_factor_verification"))
    relaxed_external = (verification == "provisional_source_supported" or _norm(rec.get("emission_factor_source_class")) == "provisional_source_supported" or _norm(rec.get("external_ef_source_class")) == "external_verified_relaxed" or _norm(rec.get("external_ef_verification_tier")) == "relaxed")

    fallback_method = _norm(rec.get("emission_factor_fallback_method") or rec.get("external_ef_fallback_method"))
    if verification in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"} or _norm(rec.get("emission_factor_source_class")) in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
        lower = _num(rec.get("external_ef_lower_value"))
        upper = _num(rec.get("external_ef_upper_value"))
        if lower is not None and upper is not None and 0 < lower < upper:
            gsd = gsd_from_interval(lower, upper, 0.95)
            if gsd is not None:
                label = "UNVERIFIED_FALLBACK_RANGE" if not fallback_method else f"UNVERIFIED_{fallback_method.upper()}_RANGE"
                return gsd, label, "Derived lognormal GSD from the Class-4 unverified fallback envelope; this is exploratory rather than source-verified uncertainty.", None
        scores = {"reliability": 5, "completeness": 5, "temporal": 5, "geographical": 5, "technological": 5}
        gsd, variance = pedigree_gsd(scores)
        return gsd, "UNVERIFIED_FALLBACK_MAX_PEDIGREE", f"No valid Class-4 fallback range was available; maximum adapted pedigree scores used, log-variance={variance:.8f}.", scores

    # Explicit uncertainty extracted from a fully verified traceable source.
    if not relaxed_external:
        gsd = _num(rec.get("external_ef_uncertainty_gsd"))
        if gsd is not None and gsd >= 1:
            return gsd, "SOURCE_REPORTED_GSD", "GSD explicitly reported and verified in retrieved source evidence.", None

        cv = _num(rec.get("external_ef_uncertainty_cv"))
        if cv is not None:
            cv_fraction = cv / 100.0 if cv > 1 else cv
            gsd = gsd_from_cv(cv_fraction)
            if gsd is not None:
                return gsd, "SOURCE_REPORTED_CV", f"Converted source-reported CV={cv_fraction:.6g} to lognormal GSD.", None

        lower = _num(rec.get("external_ef_uncertainty_lower_value"))
        upper = _num(rec.get("external_ef_uncertainty_upper_value"))
        conf = _num(rec.get("external_ef_uncertainty_confidence_level"))
        if lower is not None and upper is not None:
            confidence = (conf / 100.0 if conf and conf > 1 else conf) if conf is not None else 0.95
            gsd = gsd_from_interval(lower, upper, confidence or 0.95)
            if gsd is not None:
                return gsd, "SOURCE_REPORTED_INTERVAL", f"Derived lognormal GSD from source-reported central {100*(confidence or 0.95):.1f}% interval.", None

    # Dispersion among independently accepted/supporting source values is usable
    # for both verified and provisional pathways because every peer value is
    # anchored to retrieved evidence rather than model memory.
    peers = _parse_json_numbers(rec.get("external_ef_peer_values_json"))
    gsd = gsd_from_peer_values(peers)
    if gsd is not None:
        label = "RELAXED_EXTERNAL_MULTI_SOURCE_DISPERSION" if relaxed_external else "MULTI_SOURCE_DISPERSION"
        return gsd, label, f"Calculated GSD from {len(peers)} independently retrieved factors on the same normalized reference basis.", None

    scores, score_reason = pedigree_scores_for_row(rec)
    gsd, variance = pedigree_gsd(scores)
    return gsd, "ADAPTED_PEDIGREE_MATRIX", f"{score_reason} Combined log-variance={variance:.8f}.", scores


def _property_gsd(rec: dict[str, Any]) -> tuple[float, str, str]:
    method = _norm(rec.get("conversion_method"))
    uses_property = method in {
        "volume_to_mass", "mass_to_volume", "mass_to_area", "mass_to_count",
        "area_to_volume", "volume_to_area", "area_to_mass", "count_to_mass",
    }
    if not uses_property:
        return 1.0, "NO_STOCHASTIC_PROPERTY", "No uncertain physical property is required by this conversion."

    verification = _norm(rec.get("property_verification"))
    if verification in {"project_or_bom", "project_input", "bom_extracted"}:
        return 1.0, "PROJECT_OR_BOM_PROPERTY", "Project/BOM property is treated as fixed because no project-specific uncertainty was supplied."

    fallback_method = _norm(rec.get("property_fallback_method_final") or rec.get("property_fallback_method"))
    if verification in {"unverified_fallback_estimate", "llm_unverified_estimate", "conservative_analog_estimate"}:
        lower = _num(rec.get("property_lower_value"))
        upper = _num(rec.get("property_upper_value"))
        if lower is not None and upper is not None and 0 < lower < upper:
            gsd = gsd_from_interval(lower, upper, 0.95)
            if gsd is not None:
                label = "UNVERIFIED_FALLBACK_PROPERTY_RANGE" if not fallback_method else f"UNVERIFIED_{fallback_method.upper()}_PROPERTY_RANGE"
                return gsd, label, "Derived property GSD from the Class-4 unverified fallback envelope; this is exploratory rather than source-verified uncertainty."
        scores = {"reliability": 5, "completeness": 5, "temporal": 5, "geographical": 5, "technological": 5}
        gsd, variance = pedigree_gsd(scores)
        return gsd, "UNVERIFIED_FALLBACK_PROPERTY_MAX_PEDIGREE", f"No valid Class-4 property range was available; maximum adapted pedigree scores used, log-variance={variance:.8f}."

    if verification in {"provisional_source_supported", "traceable_web_relaxed"}:
        peers = _parse_json_numbers(rec.get("property_peer_values_json"))
        gsd = gsd_from_peer_values(peers)
        if gsd is not None:
            return gsd, "RELAXED_EXTERNAL_PROPERTY_MULTI_SOURCE_DISPERSION", f"Calculated property GSD from {len(peers)} independent source-supported values."
        temporal = _temporal_score(rec.get("property_source_year"))
        geographical = _geography_score(rec.get("property_source_geography"))
        scores = {
            "reliability": 4,
            "completeness": 4,
            "temporal": temporal,
            "geographical": geographical,
            "technological": 4,
        }
        gsd, variance = pedigree_gsd(scores)
        return gsd, "RELAXED_EXTERNAL_PROPERTY_ADAPTED_PEDIGREE", f"External Verified relaxed-phase property; conservative pedigree log-variance={variance:.8f}."

    # Traceable web property: derive representativeness scores from its metadata.
    temporal = _temporal_score(rec.get("property_source_year"))
    geographical = _geography_score(rec.get("property_source_geography"))
    scores = {
        "reliability": 2,
        "completeness": 4,
        "temporal": temporal,
        "geographical": geographical,
        "technological": 2,
    }
    gsd, variance = pedigree_gsd(scores)
    return gsd, "PROPERTY_ADAPTED_PEDIGREE", f"Traceable conversion property pedigree log-variance={variance:.8f}."


def enrich_row_uncertainty(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    new_cols = [
        "factor_uncertainty_gsd", "factor_uncertainty_method", "factor_uncertainty_reason",
        "property_uncertainty_gsd", "property_uncertainty_method", "property_uncertainty_reason",
        "row_uncertainty_gsd", "uncertainty_method_version",
        "pedigree_reliability", "pedigree_completeness", "pedigree_temporal",
        "pedigree_geographical", "pedigree_technological",
        "gwp_p2_5", "gwp_p5", "gwp_p50", "gwp_p95", "gwp_p97_5",
    ]
    for c in new_cols:
        if c not in out.columns:
            out[c] = np.nan if c.startswith(("factor_uncertainty_gsd", "property_uncertainty_gsd", "row_uncertainty_gsd", "pedigree_", "gwp_")) else None

    mask = out["calculation_status"].eq("CALCULATED")
    for idx in out.index[mask]:
        rec = out.loc[idx].to_dict()
        factor_gsd, factor_method, factor_reason, scores = _factor_gsd(rec)
        prop_gsd, prop_method, prop_reason = _property_gsd(rec)
        sigma = math.sqrt(math.log(max(factor_gsd, 1.0)) ** 2 + math.log(max(prop_gsd, 1.0)) ** 2)
        row_gsd = math.exp(sigma)
        point = float(out.at[idx, "gwp_total"])
        q = [math.exp(z * sigma) for z in (-1.959963984540054, -1.6448536269514722, 0.0, 1.6448536269514722, 1.959963984540054)]
        vals = [point * m for m in q]

        out.at[idx, "factor_uncertainty_gsd"] = factor_gsd
        out.at[idx, "factor_uncertainty_method"] = factor_method
        out.at[idx, "factor_uncertainty_reason"] = factor_reason
        out.at[idx, "property_uncertainty_gsd"] = prop_gsd
        out.at[idx, "property_uncertainty_method"] = prop_method
        out.at[idx, "property_uncertainty_reason"] = prop_reason
        out.at[idx, "row_uncertainty_gsd"] = row_gsd
        out.at[idx, "uncertainty_method_version"] = UNCERTAINTY_METHOD_VERSION
        if scores:
            out.at[idx, "pedigree_reliability"] = scores["reliability"]
            out.at[idx, "pedigree_completeness"] = scores["completeness"]
            out.at[idx, "pedigree_temporal"] = scores["temporal"]
            out.at[idx, "pedigree_geographical"] = scores["geographical"]
            out.at[idx, "pedigree_technological"] = scores["technological"]
        for c, v in zip(("gwp_p2_5", "gwp_p5", "gwp_p50", "gwp_p95", "gwp_p97_5"), vals):
            out.at[idx, c] = v
    return out


def _scope_stats(draws: np.ndarray, point: float, scope: str, n: int) -> dict[str, Any]:
    q = np.percentile(draws[:n], [2.5, 5, 50, 95, 97.5])
    return {
        "scope": scope,
        "point_estimate": float(point),
        "mean": float(np.mean(draws[:n])),
        "p2_5": float(q[0]),
        "p5": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "p97_5": float(q[4]),
    }


def _relative_change(a: float, b: float) -> float:
    denom = max(abs(float(b)), 1e-12)
    return abs(float(a) - float(b)) / denom


def monte_carlo_with_convergence(results: pd.DataFrame):
    """Run reproducible row-specific lognormal Monte Carlo with convergence.

    Returns ``(summary_df, convergence_df, raw_draws_df)``. Factor and physical-
    property uncertainty are sampled independently in log space.  The raw table
    stores, for every run and calculated BOM row, the sampled emission factor,
    sampled quantity in the factor reference unit, and resulting line-item GWP.
    This makes every value entering every Monte Carlo building total auditable.

    One seeded random stream is generated up to the maximum schedule size and
    smaller convergence trials use deterministic prefixes of that same stream.
    """
    calc = results[results["calculation_status"] == "CALCULATED"].copy()
    if calc.empty:
        empty_summary = pd.DataFrame(columns=[
            "scope", "point_estimate", "mean", "p2_5", "p5", "median", "p95", "p97_5",
            "simulations", "seed", "converged", "convergence_tolerance",
        ])
        return empty_summary, pd.DataFrame(), pd.DataFrame()

    points = calc["gwp_total"].astype(float).to_numpy()
    base_efs = calc["emission_factor_per_reference_unit"].astype(float).to_numpy()
    base_qty = calc["quantity_in_reference_unit"].astype(float).to_numpy()
    factor_sigmas = np.log(calc["factor_uncertainty_gsd"].astype(float).clip(lower=1.0).to_numpy())
    property_sigmas = np.log(calc["property_uncertainty_gsd"].astype(float).clip(lower=1.0).to_numpy())

    max_n = max(MONTE_CARLO_SCHEDULE)
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    # Independent factor/property draws.  Their product is equivalent to the
    # combined row-level lognormal variance used for analytical row quantiles.
    z_factor = rng.standard_normal(size=(max_n, len(calc)))
    z_property = rng.standard_normal(size=(max_n, len(calc)))
    factor_multiplier = np.exp(z_factor * factor_sigmas[None, :])
    property_multiplier = np.exp(z_property * property_sigmas[None, :])
    sampled_ef = base_efs[None, :] * factor_multiplier
    sampled_qty = base_qty[None, :] * property_multiplier
    sampled_rows = sampled_ef * sampled_qty

    def _bool_mask(name: str, default: bool) -> np.ndarray:
        series = calc[name] if name in calc.columns else pd.Series(default, index=calc.index)
        vals = []
        for v in series.tolist():
            if isinstance(v, (bool, np.bool_)):
                vals.append(bool(v))
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append(default)
            else:
                vals.append(str(v).strip().lower() in {"1", "true", "yes", "y"})
        return np.asarray(vals, dtype=bool)

    verified_mask = _bool_mask("included_in_verified_calculation", False)
    exploratory_mask = _bool_mask("included_in_complete_exploratory_screening", True)
    verified_total = sampled_rows[:, verified_mask].sum(axis=1) if verified_mask.any() else np.zeros(max_n)
    exploratory_total = sampled_rows[:, exploratory_mask].sum(axis=1) if exploratory_mask.any() else np.zeros(max_n)
    verified_point = float(points[verified_mask].sum()) if verified_mask.any() else 0.0
    exploratory_point = float(points[exploratory_mask].sum()) if exploratory_mask.any() else 0.0
    exploratory_complete = bool(results.get("calculation_status", pd.Series(index=results.index, dtype=str)).eq("CALCULATED").all())
    exploratory_scope = "Complete exploratory GWP (classes 1-4)" if exploratory_complete else "Exploratory GWP subtotal (input/model failures remain)"

    convergence_rows: list[dict[str, Any]] = []
    prev_by_scope: dict[str, dict[str, Any]] = {}
    selected_n = max_n
    converged = False
    for n in MONTE_CARLO_SCHEDULE:
        scopes = [
            _scope_stats(verified_total, verified_point, "Verified A1-A3 GWP (classes 1-3)", n),
            _scope_stats(exploratory_total, exploratory_point, exploratory_scope, n),
        ]
        max_change = 0.0
        have_previous = True
        for st in scopes:
            prev = prev_by_scope.get(st["scope"])
            if prev is None:
                have_previous = False
                change = None
            else:
                change = max(
                    _relative_change(st["p2_5"], prev["p2_5"]),
                    _relative_change(st["median"], prev["median"]),
                    _relative_change(st["p97_5"], prev["p97_5"]),
                )
                max_change = max(max_change, change)
            convergence_rows.append({
                "scope": st["scope"],
                "simulations": n,
                "p2_5": st["p2_5"],
                "median": st["median"],
                "p97_5": st["p97_5"],
                "max_relative_change_vs_previous": change,
            })
            prev_by_scope[st["scope"]] = st
        if have_previous and n >= MONTE_CARLO_MIN_FINAL_RUNS and max_change <= MONTE_CARLO_CONVERGENCE_TOLERANCE:
            selected_n = n
            converged = True
            break

    summary_rows = []
    for draws, point, scope in [
        (verified_total, verified_point, "Verified A1-A3 GWP (classes 1-3)"),
        (exploratory_total, exploratory_point, exploratory_scope),
    ]:
        st = _scope_stats(draws, point, scope, selected_n)
        st.update({
            "simulations": selected_n,
            "seed": MONTE_CARLO_SEED,
            "converged": converged,
            "convergence_tolerance": MONTE_CARLO_CONVERGENCE_TOLERANCE,
        })
        summary_rows.append(st)

    # Stable short labels based on item ID; duplicate IDs are disambiguated
    # deterministically.  Three columns per row expose the exact MC inputs and
    # output contribution: sampled EF, sampled reference quantity, sampled GWP.
    labels: list[str] = []
    used: dict[str, int] = {}
    for i, (_, rec) in enumerate(calc.iterrows(), start=1):
        raw_label = _clean(rec.get("item_id") or rec.get("ID") or f"row{i}")
        label = "item_" + "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in raw_label)[:40]
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label}_{used[label]}"
        labels.append(label)

    raw = pd.DataFrame({"run": np.arange(1, selected_n + 1)})
    for j, label in enumerate(labels):
        raw[f"{label}__ef"] = sampled_ef[:selected_n, j]
        raw[f"{label}__quantity_ref"] = sampled_qty[:selected_n, j]
        raw[f"{label}__gwp"] = sampled_rows[:selected_n, j]
    raw["verified_total_gwp"] = verified_total[:selected_n]
    raw["complete_exploratory_total_gwp"] = exploratory_total[:selected_n]

    return pd.DataFrame(summary_rows), pd.DataFrame(convergence_rows), raw
