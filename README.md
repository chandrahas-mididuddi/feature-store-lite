# feature-store-lite

A lightweight feature store for ML, built as a portfolio project. It covers the three things a feature store actually has to get right:

1. **Point-in-time correct offline retrieval** — training features that never leak future data.
2. **Online serving** — latest features per entity, with TTL expiry.
3. **Drift detection** — PSI + Kolmogorov–Smirnov between training and serving distributions.

Plus a small FastAPI service and a materialization job that computes features from raw transaction events.

> **Honest scope note:** this is a single-node portfolio project. It is not a replacement for [Feast](https://feast.dev/), Tecton, or a managed feature store at scale — no distributed storage, no streaming ingestion, no registry service, no access control. It exists to demonstrate the core concepts correctly, with tests proving the correctness guarantees.

## Architecture

```
raw transaction events (synthetic generator)
        │
        ▼
materialization job ──► offline store (Parquet, per feature view)
        │                       ▲  point-in-time join (as-of lookup)
        │                       │
        └───────────────► online store (in-memory + SQLite, TTL)
                                ▲  GET /features/online
                                │
                         FastAPI service ──► POST /features/historical
                                            POST /drift/report
```

| Component | File | What it does |
|---|---|---|
| Feature definitions | `feature_store/features.py` | Entity + feature schema, types, TTL, source |
| Offline store | `feature_store/offline.py` | Parquet storage, point-in-time joins |
| Online store | `feature_store/online.py` | In-memory + SQLite KV store with TTL; `RedisStore` is a Redis-compatible interface backed by SQLite (no Redis server needed) |
| Materialization | `feature_store/materialization.py` | Synthetic event generator + rolling-window feature computation + batch job writing to both stores |
| Drift detection | `feature_store/drift.py` | PSI and two-sample KS, implemented from scratch (numpy only) |
| API | `feature_store/service.py` | FastAPI: `/features/online`, `/features/historical`, `/drift/report` |

## Point-in-time correctness

The failure mode this project is built around is **training/serving skew**: if a training row for time `t` uses a feature computed from data that arrived *after* `t`, the model learns from information it will never have at inference time, and offline metrics silently overstate real performance.

The guarantee here:

- Every feature row is timestamped with the `event_timestamp` it was true as of.
- A point-in-time query `(entity, t)` returns the **latest feature row with `event_timestamp <= t`** — per entity, via an as-of lookup (`numpy.searchsorted` on sorted timestamps).
- The materialization job computes each feature row using **only events at or before that row's timestamp**.
- A query before any feature value exists returns null, not a neighbouring value.

This is covered by tests, including a dedicated leakage test that fails if future data ever leaks into a query (`tests/test_offline.py::test_point_in_time_never_leaks_future_data`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic events, materialize features, query, drift-check
python scripts/demo.py

# Start the API
uvicorn feature_store.service:app --reload
```

## API examples

```bash
# Latest online features for one user
curl "http://localhost:8000/features/online?view=user_transaction_features&entity_id=user_0000"

# Point-in-time features (as of the given timestamps — no leakage)
curl -X POST http://localhost:8000/features/historical -H "Content-Type: application/json" -d '{
  "view": "user_transaction_features",
  "rows": [
    {"entity_id": "user_0000", "event_timestamp": "2026-01-15T00:00:00"},
    {"entity_id": "user_0001", "event_timestamp": "2026-02-01T00:00:00"}
  ]
}'

# Drift report between a reference and a current sample
curl -X POST http://localhost:8000/drift/report -H "Content-Type: application/json" -d '{
  "features": ["txn_amount_sum_7d"],
  "reference": {"txn_amount_sum_7d": [120.5, 98.0, 210.3]},
  "current":   {"txn_amount_sum_7d": [180.2, 160.7, 320.1]}
}'
```

## Drift detection

- **PSI** — reference quantiles define the bins; PSI < 0.1 = stable, 0.1–0.25 = moderate shift, > 0.25 = significant shift.
- **KS statistic** — max difference between the empirical CDFs (0 = identical, 1 = disjoint). No binning, so it catches shifts PSI's bins might hide.

## Example feature view

`user_transaction_features` (entity `user_id`, TTL 7 days), computed from raw transaction events:

| Feature | Meaning |
|---|---|
| `txn_count_7d` | Transactions in the last 7 days |
| `txn_amount_sum_7d` | Total spend in the last 7 days |
| `txn_amount_avg_30d` | Average transaction amount, 30 days |
| `txn_count_30d` | Transactions in the last 30 days |
| `distinct_merchants_7d` | Distinct merchants, 7 days |
| `max_amount_7d` | Largest single transaction, 7 days |

## Tests

```bash
python -m pytest -q
```

The suite covers point-in-time correctness (including the leakage test), TTL expiry, materialization, PSI/KS math, and the API. CI runs it on Python 3.11 and 3.12 via GitHub Actions (`.github/workflows/ci.yml`).

## Docker

```bash
docker build -t feature-store-lite .
docker run -p 8000:8000 feature-store-lite
```

## Limitations

- Single node; Parquet files + SQLite are the storage layer. No partitioning strategy for very large tables, no streaming ingestion (batch materialization only).
- No feature registry service, no versioning of feature definitions, no access control or auth on the API.
- The Redis interface is a compatibility layer over SQLite, not a real Redis client — swap in `redis-py` for production use.
- Synthetic data only; the generator is for the demo, not a claim about real transaction behaviour.
