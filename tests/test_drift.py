"""Tests for drift detection: PSI and KS math."""

import numpy as np
import pandas as pd
import pytest

from feature_store.drift import check_feature_drift, drift_report, ks_statistic, psi


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(100, 15, size=1_000)
    assert psi(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_small_shift_is_stable():
    rng = np.random.default_rng(1)
    ref = rng.normal(100, 15, size=5_000)
    cur = rng.normal(101, 15, size=5_000)  # tiny mean shift
    value = psi(ref, cur)
    assert value < 0.1


def test_psi_large_shift_is_significant():
    rng = np.random.default_rng(2)
    ref = rng.normal(100, 15, size=5_000)
    cur = rng.normal(160, 15, size=5_000)  # large mean shift
    assert psi(ref, cur) > 0.25


def test_psi_is_symmetric_in_spirit_and_non_negative():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, size=2_000)
    cur = rng.normal(1, 1, size=2_000)
    assert psi(ref, cur) >= 0
    assert psi(cur, ref) >= 0


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError):
        psi(np.array([]), np.array([1.0, 2.0]))


def test_ks_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(50, 10, size=500)
    assert ks_statistic(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_ks_disjoint_distributions_is_one():
    ref = np.array([1.0, 2.0, 3.0])
    cur = np.array([100.0, 200.0, 300.0])
    assert ks_statistic(ref, cur) == pytest.approx(1.0)


def test_ks_partial_overlap_is_between_zero_and_one():
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, size=1_000)
    cur = rng.normal(0.5, 1, size=1_000)
    value = ks_statistic(ref, cur)
    assert 0.0 < value < 1.0


def test_drift_report_verdicts_and_overall():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({
        "amount": rng.normal(100, 15, size=5_000),
        "count": rng.normal(5, 1, size=5_000),
    })
    current = pd.DataFrame({
        "amount": rng.normal(100, 15, size=5_000),   # unchanged
        "count": rng.normal(9, 1, size=5_000),       # shifted hard
    })
    report = drift_report(reference, current, ["amount", "count"])
    by_feature = {f["feature"]: f for f in report["features"]}
    assert by_feature["amount"]["verdict"] == "stable"
    assert by_feature["count"]["verdict"] == "significant"
    assert report["overall_verdict"] == "significant"


def test_check_feature_drift_returns_counts():
    rng = np.random.default_rng(6)
    result = check_feature_drift(
        "amount", rng.normal(0, 1, 100), rng.normal(0, 1, 200)
    )
    assert result.n_reference == 100
    assert result.n_current == 200
    assert result.verdict in {"stable", "moderate", "significant"}
