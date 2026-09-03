"""Robust consensus utilities for source-supported provisional evidence.

The functions in this module are intentionally material-agnostic. They never
encode expected embodied-carbon factors, densities, or product-specific values.
They only combine positive numerical observations that have already been
extracted from retrieved evidence and passed the relevant identity/unit checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

CONSENSUS_METHOD_VERSION = "1.0"
MODIFIED_Z_THRESHOLD = 3.5


@dataclass(frozen=True)
class ConsensusResult:
    central_value: float
    retained_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    method: str
    input_count: int
    retained_count: int


def _positive_finite(values: Sequence[float]) -> tuple[list[float], list[int]]:
    clean: list[float] = []
    original_indices: list[int] = []
    for i, raw in enumerate(values):
        try:
            value = float(raw)
        except Exception:
            continue
        if math.isfinite(value) and value > 0:
            clean.append(value)
            original_indices.append(i)
    return clean, original_indices


def robust_positive_consensus(values: Sequence[float]) -> ConsensusResult | None:
    """Return a median-based consensus with log-space robust outlier filtering.

    No material-specific plausibility range is used. Outliers are identified
    only from disagreement within the retrieved evidence itself.

    Rules:
    - one observation -> use that source-supported value;
    - two observations -> use their ordinary median, no outlier claim;
    - three or more -> use modified z-scores in log space (MAD); when MAD is
      exactly zero because a majority of observations are identical, retain the
      repeated consensus values and flag disagreeing observations.
    - filtering is abandoned if it would leave fewer than two observations in a
      multi-source set; this avoids manufacturing consensus from one surviving
      point.
    """
    clean, original_indices = _positive_finite(values)
    n = len(clean)
    if n == 0:
        return None
    if n == 1:
        return ConsensusResult(
            central_value=float(clean[0]),
            retained_indices=(original_indices[0],),
            outlier_indices=(),
            method="SINGLE_SOURCE_MEDIAN",
            input_count=1,
            retained_count=1,
        )

    arr = np.asarray(clean, dtype=float)
    keep = np.ones(n, dtype=bool)
    method = "MULTI_SOURCE_MEDIAN_NO_OUTLIER_FILTER"

    if n >= 3:
        logs = np.log(arr)
        med = float(np.median(logs))
        abs_dev = np.abs(logs - med)
        mad = float(np.median(abs_dev))
        eps = 1e-12
        if mad > eps:
            modified_z = 0.6744897501960817 * abs_dev / mad
            candidate_keep = modified_z <= MODIFIED_Z_THRESHOLD
            if int(candidate_keep.sum()) >= 2:
                keep = candidate_keep
                method = "LOG_MAD_FILTERED_MEDIAN"
        else:
            # A zero MAD means at least half the observations sit exactly on the
            # log-median. When two or more observations form that repeated
            # consensus and other values disagree, the disagreeing values are
            # source-set outliers. If all observations are essentially equal,
            # keep everything.
            consensus_mask = abs_dev <= eps
            if int(consensus_mask.sum()) >= 2 and int(consensus_mask.sum()) < n:
                keep = consensus_mask
                method = "REPEATED_MEDIAN_FILTERED"
            else:
                method = "MULTI_SOURCE_MEDIAN_ZERO_MAD"

    retained_local = np.where(keep)[0].tolist()
    if n >= 2 and len(retained_local) < 2:
        retained_local = list(range(n))
        keep = np.ones(n, dtype=bool)
        method = "MULTI_SOURCE_MEDIAN_FILTER_ABORTED"

    retained_values = arr[keep]
    central = float(np.median(retained_values))
    retained_original = tuple(original_indices[i] for i in retained_local)
    outlier_original = tuple(
        original_indices[i] for i in range(n) if not bool(keep[i])
    )
    return ConsensusResult(
        central_value=central,
        retained_indices=retained_original,
        outlier_indices=outlier_original,
        method=method,
        input_count=n,
        retained_count=len(retained_original),
    )


def canonical_factor_basis(value: float, reference_unit: str) -> tuple[float, str] | None:
    """Canonicalize mass-reference emission factors for evidence comparison.

    The returned value is still kg CO2e per returned reference unit. Non-mass
    reference units are left unchanged. This conversion is dimensional only and
    contains no material-specific assumptions.
    """
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    u = str(reference_unit or "").strip().lower().replace("³", "3").replace("²", "2")
    aliases = {"ton": "t", "tonne": "t", "metrictonne": "t", "unit": "item", "piece": "item", "each": "item"}
    u = aliases.get(u, u)
    if u == "t":
        return value / 1000.0, "kg"
    if u == "g":
        return value * 1000.0, "kg"
    if u in {"kg", "m3", "m2", "item"}:
        return value, u
    return None

# Material-agnostic ensemble settings for model-only fallback estimates.
# These are global statistical criteria, not material-specific environmental or
# physical-property limits.
MODEL_ENSEMBLE_SIZE = 5
MODEL_ENSEMBLE_MIN_AGREEMENT = 3
MODEL_ENSEMBLE_MAX_RETAINED_RATIO = 4.0
MODEL_ENSEMBLE_METHOD_VERSION = "1.0"


@dataclass(frozen=True)
class EnsembleConsensusResult:
    accepted: bool
    central_value: float | None
    retained_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    method: str
    input_count: int
    retained_count: int
    retained_ratio: float | None
    reason: str


def robust_model_ensemble_consensus(
    values: Sequence[float],
    *,
    min_agreement: int = MODEL_ENSEMBLE_MIN_AGREEMENT,
    max_retained_ratio: float = MODEL_ENSEMBLE_MAX_RETAINED_RATIO,
) -> EnsembleConsensusResult:
    """Consensus gate for five model-generated candidates, with no material priors.

    The calculation is performed in log space so multiplicative disagreement is
    treated symmetrically. First, modified-z/MAD filtering removes clear outliers.
    If the retained set is still too dispersed, the tightest contiguous cluster
    containing at least ``min_agreement`` estimates is examined. A consensus is
    accepted only when at least the requested number agree within the same global
    multiplicative spread. No GWP, density, thickness, or item-mass value is
    encoded here.
    """
    clean, original_indices = _positive_finite(values)
    n = len(clean)
    if n < min_agreement:
        return EnsembleConsensusResult(False, None, (), tuple(original_indices),
            "INSUFFICIENT_VALID_CANDIDATES", n, 0, None,
            f"need_at_least_{min_agreement}_positive_finite_candidates")

    arr = np.asarray(clean, dtype=float)
    logs = np.log(arr)
    med = float(np.median(logs))
    abs_dev = np.abs(logs - med)
    mad = float(np.median(abs_dev))
    keep = np.ones(n, dtype=bool)
    method = "LOG_MAD"
    eps = 1e-12
    if mad > eps:
        modified_z = 0.6744897501960817 * abs_dev / mad
        keep = modified_z <= MODIFIED_Z_THRESHOLD
    else:
        same = abs_dev <= eps
        if int(same.sum()) >= min_agreement:
            keep = same
            method = "REPEATED_LOG_MEDIAN"

    def _ratio(indices: list[int]) -> float:
        vals = arr[indices]
        return float(vals.max() / vals.min()) if len(vals) else math.inf

    retained_local = np.where(keep)[0].tolist()
    ratio = _ratio(retained_local) if retained_local else math.inf

    # If the first robust pass is too small or too dispersed, find the tightest
    # multiplicative cluster of >= min_agreement candidates. This catches cases
    # such as [0.3, 0.4, 0.5, 100, 300] without any material-specific ceiling.
    if len(retained_local) < min_agreement or ratio > max_retained_ratio:
        order = np.argsort(arr)
        best: tuple[float, list[int]] | None = None
        for width in range(min_agreement, n + 1):
            for start in range(0, n - width + 1):
                idx = order[start:start + width].tolist()
                r = _ratio(idx)
                if best is None or r < best[0] or (math.isclose(r, best[0]) and len(idx) > len(best[1])):
                    best = (r, idx)
        if best is not None and best[0] <= max_retained_ratio:
            ratio, retained_local = best
            method = "TIGHTEST_LOG_CLUSTER"
        else:
            return EnsembleConsensusResult(False, None, (), tuple(original_indices),
                "NO_COHERENT_CLUSTER", n, 0,
                None if best is None else float(best[0]),
                f"no_{min_agreement}_candidate_cluster_within_{max_retained_ratio:g}x_spread")

    retained_set = set(retained_local)
    retained_original = tuple(original_indices[i] for i in retained_local)
    outlier_original = tuple(original_indices[i] for i in range(n) if i not in retained_set)
    central = float(np.median(arr[retained_local]))
    return EnsembleConsensusResult(True, central, retained_original, outlier_original,
        method, n, len(retained_original), float(ratio), "ok")
