# Offline store: Parquet-backed, point-in-time correct feature retrieval.
"""Offline store.

Feature values are stored as Parquet files, one file per feature view,
with columns::

    <entity>          entity id (e.g. user_id)
    event_timestamp   when the feature value became true
    created_timestamp when the row was written (ingestion time)
    <feature_1> ...   feature values

Point-in-time correctness
-------------------------
Given a query ``(entity, timestamp)``, the correct feature value is the
*latest* row for that entity whose ``event_timestamp`` is **<=** the query
timestamp. Rows with a later ``event_timestamp`` must never be used —
that would be training/serving skew (leakage): the model would see
information that did not exist yet at prediction time.

This module implements that with a per-entity as-of lookup
(``numpy.searchsorted`` on the sorted event timestamps), which is the
same semantics as ``pandas.merge_asof`` but explicit and testable.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from .features import FEATURE_VIEWS, FeatureView


class OfflineStore:
    """Parquet-backed offline feature store."""

    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _path(self, view: FeatureView) -> str:
        return os.path.join(self.base_path, f"{view.name}.parquet")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def write(self, view: FeatureView, df: pd.DataFrame) -> int:
        """Append feature rows for ``view`` to its Parquet file.

        ``df`` must contain the entity column, ``event_timestamp`` and
        ``created_timestamp`` (both datetime-like), plus one column per
        feature in the view. Returns the number of rows written.
        """
        required = {view.entity, "event_timestamp", "created_timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        missing_features = set(view.feature_names()) - set(df.columns)
        if missing_features:
            raise ValueError(f"missing feature columns: {sorted(missing_features)}")

        df = df.copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df["created_timestamp"] = pd.to_datetime(df["created_timestamp"])
        # Keep a canonical column order and sort by entity + time so that
        # reads can use a fast as-of lookup.
        cols = [view.entity, "event_timestamp", "created_timestamp"] + view.feature_names()
        df = df[cols].sort_values([view.entity, "event_timestamp"])

        path = self._path(view)
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.sort_values([view.entity, "event_timestamp"])
        df.to_parquet(path, index=False)
        return len(df)

    def read_all(self, view: FeatureView) -> pd.DataFrame:
        """Return every stored row for ``view`` (empty frame if none)."""
        path = self._path(view)
        if not os.path.exists(path):
            return pd.DataFrame(
                columns=[view.entity, "event_timestamp", "created_timestamp"]
                + view.feature_names()
            )
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Point-in-time reads
    # ------------------------------------------------------------------
    def get_historical_features(
        self,
        view: FeatureView,
        entity_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Point-in-time join.

        ``entity_df`` has the entity column and ``event_timestamp``
        (the "as of" time for each row). Returns ``entity_df`` with one
        column per feature, holding the value as of that timestamp —
        i.e. the latest feature row with ``event_timestamp <=`` the
        query timestamp, per entity. Rows with no feature value yet
        get ``NaN``/``<NA>``.

        This is the core correctness guarantee of the store: a query
        can never see a feature value computed from data that arrived
        after the query timestamp.
        """
        if view.entity not in entity_df.columns:
            raise ValueError(f"entity_df must contain {view.entity!r}")
        if "event_timestamp" not in entity_df.columns:
            raise ValueError("entity_df must contain 'event_timestamp'")

        stored = self.read_all(view)
        result = entity_df.copy()
        result["event_timestamp"] = pd.to_datetime(result["event_timestamp"])
        for name in view.feature_names():
            result[name] = np.nan

        if stored.empty:
            return result

        stored = stored.sort_values([view.entity, "event_timestamp"])
        # Group stored rows by entity once, then as-of lookup per query row.
        grouped = {
            entity: group
            for entity, group in stored.groupby(view.entity, sort=False)
        }

        for idx, row in result.iterrows():
            group = grouped.get(row[view.entity])
            if group is None:
                continue
            times = group["event_timestamp"].to_numpy()
            # Index of the last stored row at or before the query time.
            pos = int(np.searchsorted(times, np.datetime64(row["event_timestamp"]), side="right")) - 1
            if pos < 0:
                continue  # no feature value existed yet at this time
            latest = group.iloc[pos]
            for name in view.feature_names():
                result.at[idx, name] = latest[name]

        return result

    def get_latest_features(
        self, view: FeatureView, entity_ids: list
    ) -> pd.DataFrame:
        """Latest known value per entity (no timestamp constraint)."""
        stored = self.read_all(view)
        if stored.empty:
            return pd.DataFrame(columns=[view.entity] + view.feature_names())
        stored = stored.sort_values([view.entity, "event_timestamp"])
        latest = stored.groupby(view.entity, as_index=False).tail(1)
        return latest[latest[view.entity].isin(entity_ids)].reset_index(drop=True)


def get_view(name: str) -> FeatureView:
    try:
        return FEATURE_VIEWS[name]
    except KeyError:
        raise KeyError(f"unknown feature view {name!r}; known: {sorted(FEATURE_VIEWS)}")
