"""Tests for the offline store — point-in-time correctness is the
core guarantee, so these tests are the most important in the project.
"""

import pandas as pd
import pytest

from feature_store.features import USER_TRANSACTION_FEATURES as VIEW
from feature_store.offline import OfflineStore

VIEW_COLS = [VIEW.entity, "event_timestamp", "created_timestamp"] + VIEW.feature_names()


def _feature_rows(entity, times_values):
    """Build feature rows: (timestamp, txn_count_7d) pairs for one entity."""
    rows = []
    for ts, count in times_values:
        row = {
            VIEW.entity: entity,
            "event_timestamp": pd.Timestamp(ts),
            "created_timestamp": pd.Timestamp(ts),
            "txn_count_7d": count,
            "txn_amount_sum_7d": float(count * 10),
            "txn_amount_avg_30d": 10.0,
            "txn_count_30d": count,
            "distinct_merchants_7d": 1,
            "max_amount_7d": 10.0,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=VIEW_COLS)


def test_point_in_time_returns_value_as_of_query_time(tmp_path):
    store = OfflineStore(str(tmp_path))
    store.write(VIEW, _feature_rows("user_1", [
        ("2026-01-01 10:00", 1),
        ("2026-01-05 10:00", 3),
        ("2026-01-10 10:00", 5),
    ]))
    entity_df = pd.DataFrame({
        "user_id": ["user_1"],
        "event_timestamp": [pd.Timestamp("2026-01-06 12:00")],
    })
    result = store.get_historical_features(VIEW, entity_df)
    # Jan 6 is after the Jan 5 row and before the Jan 10 row: value is 3.
    assert result.iloc[0]["txn_count_7d"] == 3


def test_point_in_time_never_leaks_future_data(tmp_path):
    """The leakage test: if the implementation ever returns a value
    computed from data after the query timestamp, this fails."""
    store = OfflineStore(str(tmp_path))
    store.write(VIEW, _feature_rows("user_1", [
        ("2026-01-01 10:00", 1),
        ("2026-01-05 10:00", 3),
        ("2026-01-10 10:00", 5),
    ]))
    # Query one minute BEFORE the Jan 5 row exists.
    entity_df = pd.DataFrame({
        "user_id": ["user_1"],
        "event_timestamp": [pd.Timestamp("2026-01-05 09:59")],
    })
    result = store.get_historical_features(VIEW, entity_df)
    assert result.iloc[0]["txn_count_7d"] == 1  # NOT 3, NOT 5


def test_query_before_any_feature_value_returns_null(tmp_path):
    store = OfflineStore(str(tmp_path))
    store.write(VIEW, _feature_rows("user_1", [("2026-01-05 10:00", 3)]))
    entity_df = pd.DataFrame({
        "user_id": ["user_1"],
        "event_timestamp": [pd.Timestamp("2026-01-01 10:00")],
    })
    result = store.get_historical_features(VIEW, entity_df)
    assert pd.isna(result.iloc[0]["txn_count_7d"])


def test_point_in_time_is_per_entity(tmp_path):
    store = OfflineStore(str(tmp_path))
    df = pd.concat([
        _feature_rows("user_1", [("2026-01-01 10:00", 1), ("2026-01-10 10:00", 9)]),
        _feature_rows("user_2", [("2026-01-03 10:00", 2)]),
    ])
    store.write(VIEW, df)
    entity_df = pd.DataFrame({
        "user_id": ["user_1", "user_2", "user_3"],
        "event_timestamp": [pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-06")],
    })
    result = store.get_historical_features(VIEW, entity_df)
    assert result.iloc[0]["txn_count_7d"] == 1   # user_1: Jan 1 row
    assert result.iloc[1]["txn_count_7d"] == 2   # user_2: Jan 3 row
    assert pd.isna(result.iloc[2]["txn_count_7d"])  # user_3: unknown entity


def test_exact_timestamp_boundary_is_inclusive(tmp_path):
    store = OfflineStore(str(tmp_path))
    store.write(VIEW, _feature_rows("user_1", [("2026-01-05 10:00", 3)]))
    entity_df = pd.DataFrame({
        "user_id": ["user_1"],
        "event_timestamp": [pd.Timestamp("2026-01-05 10:00")],  # exactly the row's time
    })
    result = store.get_historical_features(VIEW, entity_df)
    assert result.iloc[0]["txn_count_7d"] == 3


def test_write_appends_and_latest_wins(tmp_path):
    store = OfflineStore(str(tmp_path))
    store.write(VIEW, _feature_rows("user_1", [("2026-01-01 10:00", 1)]))
    store.write(VIEW, _feature_rows("user_1", [("2026-01-02 10:00", 2)]))
    latest = store.get_latest_features(VIEW, ["user_1"])
    assert len(latest) == 1
    assert latest.iloc[0]["txn_count_7d"] == 2


def test_empty_store_returns_empty_frame(tmp_path):
    store = OfflineStore(str(tmp_path))
    entity_df = pd.DataFrame({
        "user_id": ["user_1"],
        "event_timestamp": [pd.Timestamp("2026-01-06")],
    })
    result = store.get_historical_features(VIEW, entity_df)
    assert len(result) == 1
    assert pd.isna(result.iloc[0]["txn_count_7d"])


def test_write_rejects_missing_columns(tmp_path):
    store = OfflineStore(str(tmp_path))
    bad = pd.DataFrame({"user_id": ["user_1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        store.write(VIEW, bad)
