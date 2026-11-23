"""End-to-end demo: generate events -> materialize -> query -> drift check.

Run from the repo root::

    python scripts/demo.py

It prints a point-in-time query result, an online lookup, and a drift
report (reference vs. a shifted serving sample).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from feature_store.drift import drift_report
from feature_store.features import USER_TRANSACTION_FEATURES as VIEW
from feature_store.materialization import generate_transaction_events, materialize
from feature_store.offline import OfflineStore
from feature_store.online import OnlineStore


def main() -> None:
    data_dir = "data"
    offline = OfflineStore(os.path.join(data_dir, "offline"))
    online = OnlineStore(os.path.join(data_dir, "online.db"))

    events = generate_transaction_events(n_users=200, n_events=5_000)
    summary = materialize(events, offline, online)
    print(f"Materialized {summary['rows_written']} feature rows "
          f"for {summary['users_updated']} users from {len(events)} events.\n")

    # 1. Point-in-time query: features for user_0000 as of 2026-01-15.
    entity_df = pd.DataFrame({
        "user_id": ["user_0000"],
        "event_timestamp": [pd.Timestamp("2026-01-15")],
    })
    pit = offline.get_historical_features(VIEW, entity_df)
    print("Point-in-time features for user_0000 as of 2026-01-15:")
    print(pit[VIEW.feature_names()].to_string(index=False), "\n")

    # 2. Online lookup: latest values for the same user.
    print("Online features for user_0000 (latest):")
    print(json.dumps(online.get(VIEW, "user_0000"), indent=2), "\n")

    # 3. Drift: reference = all stored txn_amount_sum_7d values,
    #    current = same distribution shifted up by 50% (simulated drift).
    stored = offline.read_all(VIEW)
    rng = np.random.default_rng(0)
    reference = stored[["txn_amount_sum_7d", "txn_count_7d"]]
    current = reference.copy()
    current["txn_amount_sum_7d"] = current["txn_amount_sum_7d"] * 1.5 + rng.normal(0, 5, len(current))
    report = drift_report(reference, current, ["txn_amount_sum_7d", "txn_count_7d"])
    print("Drift report (reference vs. +50% shifted serving data):")
    print(json.dumps(report, indent=2))

    online.close()


if __name__ == "__main__":
    main()
