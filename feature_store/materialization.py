# Materialization: synthetic events -> computed features -> both stores.
"""Materialization job.

Pipeline::

    raw transaction events (synthetic generator or a CSV)
        -> rolling window aggregations per user
        -> offline store (Parquet, one row per user per event time)
        -> online store (latest values per user, with TTL)

Features are computed *as of* each event time: a feature row timestamped
``t`` only uses transactions with ``timestamp <= t``. That is what keeps
the offline store point-in-time correct — the materialization job never
writes a value that "knows the future".
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .features import USER_TRANSACTION_FEATURES, FeatureView
from .offline import OfflineStore
from .online import OnlineStore

MERCHANTS = [
    "amazon", "walmart", "target", "starbucks", "uber", "netflix",
    "whole-foods", "shell", "delta", "apple",
]

CATEGORIES = ["grocery", "travel", "dining", "retail", "transport", "entertainment"]


def generate_transaction_events(
    n_users: int = 200,
    n_events: int = 5_000,
    start: datetime = datetime(2026, 1, 1),
    days: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic transaction events.

    Amounts are log-normal (a few large purchases, many small ones) and
    a small fraction of users are "high activity" so the feature
    distributions have a realistic long tail.
    """
    rng = np.random.default_rng(seed)
    user_ids = [f"user_{i:04d}" for i in range(n_users)]
    # Activity is skewed: 10% of users generate ~40% of events.
    weights = np.array([5.0 if i < n_users // 10 else 1.0 for i in range(n_users)])
    weights = weights / weights.sum()

    users = rng.choice(user_ids, size=n_events, p=weights)
    timestamps = start + pd.to_timedelta(
        rng.uniform(0, days * 24 * 3600, size=n_events), unit="s"
    )
    amounts = np.round(rng.lognormal(mean=3.5, sigma=1.0, size=n_events), 2)
    merchants = rng.choice(MERCHANTS, size=n_events)
    categories = rng.choice(CATEGORIES, size=n_events)

    events = pd.DataFrame(
        {
            "user_id": users,
            "event_timestamp": pd.to_datetime(timestamps),
            "amount": amounts,
            "merchant": merchants,
            "category": categories,
        }
    )
    return events.sort_values(["user_id", "event_timestamp"]).reset_index(drop=True)


def compute_features(
    events: pd.DataFrame, view: FeatureView = USER_TRANSACTION_FEATURES
) -> pd.DataFrame:
    """Compute rolling features per user, as of each event time.

    For every event row we emit one feature row timestamped at that
    event, using only transactions up to and including that event.
    Windows: 7 days and 30 days, trailing from the event timestamp.
    """
    events = events.sort_values(["user_id", "event_timestamp"]).reset_index(drop=True)
    rows: list[dict] = []

    for user_id, group in events.groupby("user_id", sort=False):
        times = group["event_timestamp"].to_numpy()
        amounts = group["amount"].to_numpy(dtype=float)
        merchants = group["merchant"].to_numpy()

        for i, ts in enumerate(times):
            window_7d = times > (ts - np.timedelta64(7, "D"))
            window_30d = times > (ts - np.timedelta64(30, "D"))
            # Only events at or before the current event count (i + 1).
            idx_7d = np.where(window_7d[: i + 1])[0]
            idx_30d = np.where(window_30d[: i + 1])[0]
            amounts_7d = amounts[idx_7d]
            amounts_30d = amounts[idx_30d]
            rows.append(
                {
                    "user_id": user_id,
                    "event_timestamp": pd.Timestamp(ts),
                    "created_timestamp": pd.Timestamp.now("UTC"),
                    "txn_count_7d": int(len(idx_7d)),
                    "txn_amount_sum_7d": float(amounts_7d.sum()) if len(amounts_7d) else 0.0,
                    "txn_amount_avg_30d": float(amounts_30d.mean()) if len(amounts_30d) else 0.0,
                    "txn_count_30d": int(len(idx_30d)),
                    "distinct_merchants_7d": int(len(set(merchants[idx_7d]))) if len(idx_7d) else 0,
                    "max_amount_7d": float(amounts_7d.max()) if len(amounts_7d) else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["user_id", "event_timestamp"]).reset_index(drop=True)


def materialize(
    events: pd.DataFrame,
    offline_store: OfflineStore,
    online_store: OnlineStore,
    view: FeatureView = USER_TRANSACTION_FEATURES,
) -> dict:
    """Run one materialization pass.

    1. Compute features from ``events`` (point-in-time safe).
    2. Append them to the offline store.
    3. Push the latest value per user to the online store.

    Returns a small summary dict.
    """
    features_df = compute_features(events, view)
    if features_df.empty:
        return {"rows_written": 0, "users_updated": 0}

    n_written = offline_store.write(view, features_df)

    latest = features_df.sort_values(["user_id", "event_timestamp"]).groupby(
        "user_id", as_index=False
    ).tail(1)
    for _, row in latest.iterrows():
        online_store.set(
            view,
            str(row["user_id"]),
            {name: row[name] for name in view.feature_names()},
        )

    return {"rows_written": n_written, "users_updated": int(len(latest))}


def run_demo(data_dir: str = "data") -> dict:
    """End-to-end demo: generate events, materialize, return summary."""
    os.makedirs(data_dir, exist_ok=True)
    events_path = os.path.join(data_dir, "transactions.parquet")
    events = generate_transaction_events()
    events.to_parquet(events_path, index=False)

    offline = OfflineStore(os.path.join(data_dir, "offline"))
    online = OnlineStore(os.path.join(data_dir, "online.db"))
    summary = materialize(events, offline, online)
    summary["events_path"] = events_path
    summary["events"] = len(events)
    online.close()
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run_demo(), indent=2, default=str))
