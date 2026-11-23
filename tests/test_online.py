"""Tests for the online store: TTL expiry, persistence, Redis interface."""

import time
from dataclasses import replace

import pytest

from feature_store.features import USER_TRANSACTION_FEATURES as VIEW
from feature_store.online import OnlineStore, RedisStore


def _values(count=1, total=10.0):
    return {
        "txn_count_7d": count,
        "txn_amount_sum_7d": total,
        "txn_amount_avg_30d": 10.0,
        "txn_count_30d": count,
        "distinct_merchants_7d": 1,
        "max_amount_7d": 10.0,
    }


def test_set_then_get_returns_latest_values():
    store = OnlineStore()
    store.set(VIEW, "user_1", _values(count=2, total=25.0))
    got = store.get(VIEW, "user_1")
    assert got["txn_count_7d"] == 2
    assert got["txn_amount_sum_7d"] == 25.0
    store.close()


def test_later_write_overwrites_earlier():
    store = OnlineStore()
    store.set(VIEW, "user_1", _values(count=1))
    store.set(VIEW, "user_1", _values(count=4))
    assert store.get(VIEW, "user_1")["txn_count_7d"] == 4
    store.close()


def test_ttl_expiry_drops_stale_values():
    short_ttl = replace(VIEW, ttl_seconds=1)
    store = OnlineStore()
    store.set(short_ttl, "user_1", _values(count=3))
    assert store.get(short_ttl, "user_1")["txn_count_7d"] == 3
    time.sleep(1.2)
    # Expired: must NOT return the stale value.
    assert store.get(short_ttl, "user_1") == {}
    store.close()


def test_no_ttl_means_no_expiry():
    no_ttl = replace(VIEW, ttl_seconds=0)
    store = OnlineStore()
    store.set(no_ttl, "user_1", _values(count=3))
    time.sleep(0.1)
    assert store.get(no_ttl, "user_1")["txn_count_7d"] == 3
    store.close()


def test_unknown_entity_returns_empty_dict():
    store = OnlineStore()
    assert store.get(VIEW, "nobody") == {}
    store.close()


def test_sqlite_persists_across_reopen(tmp_path):
    db_path = str(tmp_path / "online.db")
    store = OnlineStore(db_path)
    store.set(VIEW, "user_1", _values(count=7))
    store.close()
    reopened = OnlineStore(db_path)
    assert reopened.get(VIEW, "user_1")["txn_count_7d"] == 7
    reopened.close()


def test_redis_interface_get_set_delete():
    redis = RedisStore()
    key = f"{VIEW.name}:user_1:txn_count_7d"
    assert redis.get(key) is None
    redis.set(key, 5)
    assert redis.get(key) == 5
    assert redis.exists(key) is True
    assert redis.delete(key) == 1
    assert redis.get(key) is None
    assert redis.exists(key) is False


def test_redis_interface_respects_expiry():
    redis = RedisStore()
    key = f"{VIEW.name}:user_1:txn_count_7d"
    redis.set(key, 5, ex=1)
    assert redis.get(key) == 5
    time.sleep(1.2)
    assert redis.get(key) is None
