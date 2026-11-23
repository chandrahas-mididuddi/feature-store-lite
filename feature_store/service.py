# FastAPI service: online lookup, point-in-time lookup, drift report.
"""HTTP API for feature-store-lite.

Endpoints
---------
GET  /health
GET  /features/online?view=<name>&entity_id=<id>
     Latest online features for one entity.
POST /features/historical
     Point-in-time features for a list of (entity, timestamp) rows.
POST /drift/report
     PSI + KS drift report between a reference and a current sample
     supplied as JSON columns.

Run with::

    uvicorn feature_store.service:app --reload
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .drift import drift_report
from .features import FEATURE_VIEWS
from .offline import OfflineStore, get_view
from .online import OnlineStore

DATA_DIR = os.environ.get("FEATURE_STORE_DATA_DIR", "data")

app = FastAPI(
    title="feature-store-lite",
    description="Lightweight feature store: point-in-time offline retrieval, "
    "online serving with TTL, and drift detection.",
    version="0.1.0",
)

# Stores are created lazily so importing the module (e.g. in tests)
# never touches the filesystem.
_offline: Optional[OfflineStore] = None
_online: Optional[OnlineStore] = None


def offline_store() -> OfflineStore:
    global _offline
    if _offline is None:
        _offline = OfflineStore(os.path.join(DATA_DIR, "offline"))
    return _offline


def online_store() -> OnlineStore:
    global _online
    if _online is None:
        _online = OnlineStore(os.path.join(DATA_DIR, "online.db"))
    return _online


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HistoricalRow(BaseModel):
    entity_id: str
    event_timestamp: datetime


class HistoricalRequest(BaseModel):
    view: str = "user_transaction_features"
    rows: list[HistoricalRow] = Field(..., min_length=1)


class DriftRequest(BaseModel):
    features: list[str]
    reference: dict[str, list[float]]
    current: dict[str, list[float]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "views": sorted(FEATURE_VIEWS)}


@app.get("/features/online")
def get_online_features(view: str, entity_id: str) -> dict:
    try:
        feature_view = get_view(view)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    values = online_store().get(feature_view, entity_id)
    if not values:
        raise HTTPException(
            status_code=404,
            detail=f"no online features for entity {entity_id!r} in view {view!r}",
        )
    return {"view": view, "entity_id": entity_id, "features": values}


@app.post("/features/historical")
def get_historical_features(request: HistoricalRequest) -> dict:
    try:
        feature_view = get_view(request.view)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    entity_df = pd.DataFrame(
        {
            feature_view.entity: [r.entity_id for r in request.rows],
            "event_timestamp": [r.event_timestamp for r in request.rows],
        }
    )
    result = offline_store().get_historical_features(feature_view, entity_df)
    records: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        records.append(
            {
                "entity_id": row[feature_view.entity],
                "event_timestamp": row["event_timestamp"].isoformat(),
                "features": {
                    name: (None if pd.isna(row[name]) else row[name])
                    for name in feature_view.feature_names()
                },
            }
        )
    return {"view": request.view, "rows": records}


@app.post("/drift/report")
def get_drift_report(request: DriftRequest) -> dict:
    try:
        reference_df = pd.DataFrame(request.reference)
        current_df = pd.DataFrame(request.current)
        return drift_report(reference_df, current_df, request.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
