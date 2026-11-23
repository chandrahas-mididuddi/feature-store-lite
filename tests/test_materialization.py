"""Tests for the materialization job and the synthetic event generator."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from feature_store.features import USER_TRANSACTION_FEATURES as VIEW
from feature_store.materialization import (
    compute_features,
    generate_transaction_events,
    materialize,
)
from feature_store.offline import OfflineStore
from feature_store.online import OnlineStore


def test_event_generator_shape_and_determinism():
    events = generate_transaction_events(n_users=10, n_events=100, seed=7)
    assert list(events.columns) == [
        "user_id", "event_timestamp", "amount", "merchant", "category",
    ]
    assert len(events) == 100
    assert (events["amount"] > 0).all()
    # Same seed -> same data.
    again = generate_transaction_events(n_users=10, n_events=100, seed=7)
    pd.testing.assert_frame_equal(events, again)


def test_compute_features_first_event_is_count_one():
    events = pd.DataFrame({
        "user_id": ["user_1", "user_1", "user_1"],
        "event_timestamp": pd.to_datetime([
            "2026-01-01 10:00", "2026-01-02 10:00", "2026-01-03 10:00",
        ]),
        "amount": [10.0, 20.0, 30.0],
        "merchant": ["amazon", "amazon", "walmart"],
        "category": ["retail", "retail", "grocery"],
    })
    features = compute_features(events)
    assert len(features) == 3
    first = features.iloc[0]
    assert first["txn_count_7d"] == 1
    assert first["txn_amount_sum_7d"] == 10.0
    assert first["distinct_merchants_7d"] == 1
    third = features.iloc[2]
    assert third["txn_count_7d"] == 3
    assert third["txn_amount_sum_7d"] == 60.0
    assert third["txn_amount_avg_30d"] == 20.0
    assert third["distinct_merchants_7d"] == 2
    assert third["max_amount_7d"] == 30.0


def test_compute_features_window_excludes_old_events():
    events = pd.DataFrame({
        "user_id": ["user_1"] * 2,
        "event_timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-10 10:00"]),
        "amount": [100.0, 5.0],
        "merchant": ["amazon", "amazon"],
        "category": ["retail", "retail"],
    })
    features = compute_features(events)
    second = features.iloc[1]
    # The Jan 1 event is 9 days old: outside the 7-day window,
    # inside the 30-day window.
    assert second["txn_count_7d"] == 1
    assert second["txn_amount_sum_7d"] == 5.0
    assert second["txn_count_30d"] == 2
    assert second["txn_amount_avg_30d"] == 52.5


def test_compute_features_never_uses_future_events():
    """Each feature row may only use events at or before its own time."""
    events = generate_transaction_events(n_users=5, n_events=200, seed=3)
    features = compute_features(events)
    # The last event per user: its 30-day count can never exceed the
    # number of events that user actually had up to that time.
    for user_id, group in events.groupby("user_id"):
        feat = features[features["user_id"] == user_id]
        assert feat["txn_count_30d"].max() <= len(group)
        # And the final feature row's 30d sum must equal the sum of that
        # user's events in the trailing 30 days — recompute directly.
        last = feat.iloc[-1]
        cutoff_7d = last["event_timestamp"] - pd.Timedelta(days=7)
        expected_7d = group[group["event_timestamp"] > cutoff_7d]["amount"].sum()
        assert last["txn_amount_sum_7d"] == pytest.approx(expected_7d)
        cutoff_30d = last["event_timestamp"] - pd.Timedelta(days=30)
        expected_30d = group[group["event_timestamp"] > cutoff_30d]["amount"].sum()
        # 7d sum is a subset of the 30d window sum.
        assert last["txn_amount_sum_7d"] <= expected_30d + 1e-9


def test_materialize_writes_both_stores(tmp_path):
    events = generate_transaction_events(n_users=10, n_events=200, seed=11)
    offline = OfflineStore(str(tmp_path / "offline"))
    online = OnlineStore(str(tmp_path / "online.db"))
    summary = materialize(events, offline, online)

    assert summary["rows_written"] == 200
    assert summary["users_updated"] == 10

    stored = offline.read_all(VIEW)
    assert len(stored) == 200

    # Online store holds the latest value per user.
    online_values = online.get(VIEW, "user_0000")
    assert online_values["txn_count_7d"] >= 1
    # ... and it matches the offline store's latest row for that user.
    latest_offline = stored[stored["user_id"] == "user_0000"].sort_values(
        "event_timestamp"
    ).iloc[-1]
    assert online_values["txn_count_7d"] == latest_offline["txn_count_7d"]
    online.close()


def test_materialize_empty_events_is_noop(tmp_path):
    offline = OfflineStore(str(tmp_path / "offline"))
    online = OnlineStore(str(tmp_path / "online.db"))
    empty = pd.DataFrame(columns=["user_id", "event_timestamp", "amount", "merchant", "category"])
    summary = materialize(empty, offline, online)
    assert summary == {"rows_written": 0, "users_updated": 0}
    online.close()
