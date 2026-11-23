"""feature-store-lite: a lightweight feature store for ML.

Point-in-time correct offline retrieval (Parquet), an online store with
TTL (in-memory + SQLite, Redis-compatible interface), a materialization
job, and PSI/KS drift detection — with a small FastAPI service on top.
"""

from .drift import check_feature_drift, drift_report, ks_statistic, psi
from .features import FEATURE_VIEWS, Feature, FeatureType, FeatureView, USER_TRANSACTION_FEATURES
from .materialization import compute_features, generate_transaction_events, materialize
from .offline import OfflineStore
from .online import OnlineStore, RedisStore

__all__ = [
    "FEATURE_VIEWS",
    "Feature",
    "FeatureType",
    "FeatureView",
    "USER_TRANSACTION_FEATURES",
    "OfflineStore",
    "OnlineStore",
    "RedisStore",
    "compute_features",
    "generate_transaction_events",
    "materialize",
    "check_feature_drift",
    "drift_report",
    "ks_statistic",
    "psi",
]

__version__ = "0.1.0"
