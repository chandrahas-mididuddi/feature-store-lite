# Online store: in-memory + SQLite-backed key-value store with TTL.
"""Online store.

Serving path: given an entity id, return the latest feature values —
fast, no scan. Values are kept in two layers:

* an in-memory dict (the hot cache), and
* a SQLite table (durable, survives a restart).

Every value carries an ``expires_at`` timestamp derived from the feature
view's TTL. Reads drop expired values instead of returning stale data.

A ``RedisStore`` class with the same interface is also provided. It is a
Redis-*compatible* interface backed by the same SQLite store, so the code
runs with no Redis server installed; swapping in a real ``redis-py``
client only means replacing the backend object, not the call sites.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .features import FeatureView


def _now_ts() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class StoredValue:
    value: Any
    updated_at: float  # epoch seconds
    expires_at: Optional[float]  # None = never expires


class OnlineStore:
    """In-memory + SQLite online feature store with TTL expiry."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], StoredValue] = {}
        # One connection per store; check_same_thread=False + a lock so
        # the FastAPI service can safely share it across threads.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS online_features (
                view_name   TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                feature     TEXT NOT NULL,
                value_json  TEXT NOT NULL,
                updated_at  REAL NOT NULL,
                expires_at  REAL,
                PRIMARY KEY (view_name, entity_id, feature)
            )
            """
        )
        self._conn.commit()
        self._load_cache_from_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_cache_from_db(self) -> None:
        now = _now_ts()
        for view_name, entity_id, feature, value_json, updated_at, expires_at in self._conn.execute(
            "SELECT view_name, entity_id, feature, value_json, updated_at, expires_at "
            "FROM online_features"
        ):
            if expires_at is not None and expires_at <= now:
                continue  # skip stale rows on load
            self._cache[(view_name, entity_id, feature)] = StoredValue(
                json.loads(value_json), updated_at, expires_at
            )

    def _is_expired(self, stored: StoredValue) -> bool:
        return stored.expires_at is not None and stored.expires_at <= _now_ts()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def set(
        self,
        view: FeatureView,
        entity_id: str,
        values: dict[str, Any],
        event_timestamp: Optional[datetime] = None,
    ) -> None:
        """Store ``values`` for ``entity_id`` under ``view``.

        ``expires_at`` is now + the view TTL (or never, if TTL <= 0).
        ``event_timestamp`` is accepted for API symmetry with the
        offline store but the online store always keeps the *latest*
        write; point-in-time reads belong to the offline store.
        """
        now = _now_ts()
        expires_at = now + view.ttl_seconds if view.ttl_seconds > 0 else None
        with self._lock:
            for feature, value in values.items():
                stored = StoredValue(value, now, expires_at)
                self._cache[(view.name, str(entity_id), feature)] = stored
                self._conn.execute(
                    """
                    INSERT INTO online_features
                        (view_name, entity_id, feature, value_json, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(view_name, entity_id, feature) DO UPDATE SET
                        value_json=excluded.value_json,
                        updated_at=excluded.updated_at,
                        expires_at=excluded.expires_at
                    """,
                    (view.name, str(entity_id), feature, json.dumps(value), now, expires_at),
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get(
        self, view: FeatureView, entity_id: str, features: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Latest values for ``entity_id``. Expired values are dropped
        (and removed from both layers) rather than returned stale."""
        names = features or view.feature_names()
        out: dict[str, Any] = {}
        with self._lock:
            for name in names:
                key = (view.name, str(entity_id), name)
                stored = self._cache.get(key)
                if stored is None:
                    continue
                if self._is_expired(stored):
                    del self._cache[key]
                    self._conn.execute(
                        "DELETE FROM online_features WHERE view_name=? AND entity_id=? AND feature=?",
                        (view.name, str(entity_id), name),
                    )
                    continue
                out[name] = stored.value
            self._conn.commit()
        return out

    def get_many(self, view: FeatureView, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {eid: self.get(view, eid) for eid in entity_ids}

    def delete(self, view: FeatureView, entity_id: str) -> None:
        with self._lock:
            for name in view.feature_names():
                self._cache.pop((view.name, str(entity_id), name), None)
            self._conn.execute(
                "DELETE FROM online_features WHERE view_name=? AND entity_id=?",
                (view.name, str(entity_id)),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class RedisStore:
    """Redis-compatible interface backed by the SQLite online store.

    Drop-in replacement for the parts of ``redis-py`` this project
    uses (``get`` / ``set`` with TTL / ``delete`` / ``exists``), so the
    serving code can be written against a Redis-style API today and
    pointed at a real Redis cluster later without changing call sites.
    No Redis server is required: values live in the ``OnlineStore``
    passed in (or a fresh in-memory one).
    """

    def __init__(self, online_store: Optional[OnlineStore] = None):
        self._store = online_store or OnlineStore()

    @staticmethod
    def _view_for(key: str) -> FeatureView:
        # Keys are namespaced "view_name:entity_id:feature"; resolve the
        # view from the registry so TTL rules still apply.
        from .features import FEATURE_VIEWS

        view_name = key.split(":", 1)[0]
        return FEATURE_VIEWS[view_name]

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str, str]:
        parts = key.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"expected key 'view:entity:feature', got {key!r}")
        return parts[0], parts[1], parts[2]

    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """SET key value [EX seconds]. ``ex`` overrides the view TTL."""
        view_name, entity_id, feature = self._parse_key(key)
        view = self._view_for(key)
        # Honour an explicit TTL by temporarily using a view copy.
        effective = view
        if ex is not None:
            from dataclasses import replace

            effective = replace(view, ttl_seconds=ex)
        self._store.set(effective, entity_id, {feature: value})
        return True

    def get(self, key: str) -> Any:
        view_name, entity_id, feature = self._parse_key(key)
        view = self._view_for(key)
        result = self._store.get(view, entity_id, [feature])
        return result.get(feature)

    def delete(self, key: str) -> int:
        view_name, entity_id, feature = self._parse_key(key)
        view = self._view_for(key)
        before = self._store.get(view, entity_id, [feature])
        if feature not in before:
            return 0
        self._store.delete(view, entity_id)
        return 1

    def exists(self, key: str) -> bool:
        return self.get(key) is not None
