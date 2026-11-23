# Drift detection: PSI and two-sample Kolmogorov–Smirnov.
"""Drift detection.

Two complementary checks between a reference distribution (training
data) and a current distribution (serving data):

* **PSI — Population Stability Index.** The reference range is split
  into quantile bins; PSI sums ``(actual - expected) * ln(actual /
  expected)`` over bins. Rule of thumb (from credit-risk practice):
  PSI < 0.1 = no significant change, 0.1–0.25 = moderate shift worth
  investigating, > 0.25 = significant shift, retrain / investigate.
* **KS — two-sample Kolmogorov–Smirnov statistic.** The maximum
  absolute difference between the two empirical CDFs. 0 = identical
  distributions, 1 = completely disjoint. No binning, so it catches
  shifts PSI's bins might hide.

Both are implemented from scratch (numpy only) so the math is visible
and testable — no scipy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPSILON = 1e-4  # smoothing so log(0) never appears in PSI


@dataclass
class DriftResult:
    feature: str
    psi: float
    ks_statistic: float
    n_reference: int
    n_current: int
    verdict: str  # "stable" | "moderate" | "significant"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "verdict": self.verdict,
        }


def _verdict(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples.

    Bin edges are the quantiles of the *reference* distribution, so the
    bins are fixed by training data and the current data is measured
    against them. Empty bins are smoothed with ``EPSILON``.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("PSI needs non-empty reference and current samples")

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if len(edges) < 2:
        return 0.0  # constant feature: no drift possible
    # Make the outer edges inclusive so no value falls outside the bins.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / len(ref), EPSILON)
    cur_pct = np.maximum(cur_counts / len(cur), EPSILON)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF difference)."""
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("KS needs non-empty reference and current samples")

    combined = np.sort(np.concatenate([ref, cur]))
    cdf_ref = np.searchsorted(ref, combined, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, combined, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def check_feature_drift(
    feature: str,
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> DriftResult:
    """PSI + KS for one feature, with a verdict."""
    value = psi(reference, current, n_bins=n_bins)
    ks = ks_statistic(reference, current)
    return DriftResult(
        feature=feature,
        psi=value,
        ks_statistic=ks,
        n_reference=int(len(np.asarray(reference))),
        n_current=int(len(np.asarray(current))),
        verdict=_verdict(value),
    )


def drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> dict:
    """Drift report over several features.

    Returns ``{"features": [...], "overall_verdict": ...}`` where the
    overall verdict is the worst per-feature verdict.
    """
    results = []
    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            raise ValueError(f"feature {feature!r} missing from one of the dataframes")
        results.append(
            check_feature_drift(
                feature,
                reference_df[feature].to_numpy(dtype=float),
                current_df[feature].to_numpy(dtype=float),
                n_bins=n_bins,
            )
        )
    order = {"stable": 0, "moderate": 1, "significant": 2}
    overall = max(results, key=lambda r: order[r.verdict]).verdict if results else "stable"
    return {
        "features": [r.to_dict() for r in results],
        "overall_verdict": overall,
    }
