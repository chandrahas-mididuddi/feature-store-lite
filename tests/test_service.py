"""API tests for the FastAPI service (in-process, via TestClient)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import feature_store.service as service
from feature_store.features import USER_TRANSACTION_FEATURES as VIEW
from feature_store.materialization import generate_transaction_events, materialize
from feature_store.offline import OfflineStore
from feature_store.online import OnlineStore


@pytest.fixture()
def client(tmp_path):
    """A TestClient wired to fresh stores in a temp dir, pre-loaded
    with a small materialized dataset."""
    offline = OfflineStore(str(tmp_path / "offline"))
    online = OnlineStore(str(tmp_path / "online.db"))
    events = generate_transaction_events(n_users=5, n_events=100, seed=21)
    materialize(events, offline, online)
    service._offline = offline
    service._online = online
    with TestClient(service.app) as c:
        yield c
    online.close()
    service._offline = None
    service._online = None


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "user_transaction_features" in response.json()["views"]


def test_online_endpoint_returns_latest_features(client):
    response = client.get(
        "/features/online",
        params={"view": VIEW.name, "entity_id": "user_0000"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["entity_id"] == "user_0000"
    assert body["features"]["txn_count_7d"] >= 1


def test_online_endpoint_404_for_unknown_entity(client):
    response = client.get(
        "/features/online",
        params={"view": VIEW.name, "entity_id": "user_9999"},
    )
    assert response.status_code == 404


def test_online_endpoint_404_for_unknown_view(client):
    response = client.get(
        "/features/online",
        params={"view": "no_such_view", "entity_id": "user_0000"},
    )
    assert response.status_code == 404


def test_historical_endpoint_is_point_in_time_correct(client):
    # user_0000's first event is in early January; query it as of a
    # time before that event and the features must be null.
    response = client.post(
        "/features/historical",
        json={
            "view": VIEW.name,
            "rows": [
                {"entity_id": "user_0000", "event_timestamp": "2025-12-01T00:00:00"},
                {"entity_id": "user_0000", "event_timestamp": "2026-03-01T00:00:00"},
            ],
        },
    )
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert rows[0]["features"]["txn_count_7d"] is None  # before any event
    assert rows[1]["features"]["txn_count_7d"] is not None  # after events


def test_drift_endpoint_stable_and_shifted(client):
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 15, size=2_000).tolist()
    same = rng.normal(100, 15, size=2_000).tolist()
    shifted = rng.normal(160, 15, size=2_000).tolist()

    stable = client.post(
        "/drift/report",
        json={"features": ["amount"], "reference": {"amount": ref}, "current": {"amount": same}},
    )
    assert stable.status_code == 200
    assert stable.json()["overall_verdict"] == "stable"

    drifted = client.post(
        "/drift/report",
        json={"features": ["amount"], "reference": {"amount": ref}, "current": {"amount": shifted}},
    )
    assert drifted.status_code == 200
    assert drifted.json()["overall_verdict"] == "significant"


def test_drift_endpoint_rejects_missing_feature(client):
    response = client.post(
        "/drift/report",
        json={"features": ["nope"], "reference": {"amount": [1.0]}, "current": {"amount": [1.0]}},
    )
    assert response.status_code == 400
