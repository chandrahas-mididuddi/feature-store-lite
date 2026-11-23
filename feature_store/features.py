# Feature definitions for feature-store-lite
"""Feature definitions: the schema layer.

A ``Feature`` describes one computed value attached to an entity.
A ``FeatureView`` groups features that share an entity, a source table,
and a TTL (how long a computed value stays valid in the online store).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeatureType(str, Enum):
    """Supported feature value types."""

    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"


# Map feature types to pandas dtypes used in the offline store.
PANDAS_DTYPES = {
    FeatureType.INT: "Int64",
    FeatureType.FLOAT: "float64",
    FeatureType.STRING: "string",
    FeatureType.BOOL: "boolean",
}


@dataclass(frozen=True)
class Feature:
    """A single feature in a feature view.

    Attributes:
        name: Feature name, unique within its feature view.
        dtype: Value type (int / float / string / bool).
        description: Human-readable description of what the feature means.
    """

    name: str
    dtype: FeatureType
    description: str = ""


@dataclass
class FeatureView:
    """A group of features computed together for one entity type.

    Attributes:
        name: Feature view name, e.g. ``"user_transaction_features"``.
        entity: Entity the features describe, e.g. ``"user_id"``.
        features: The features in this view.
        ttl_seconds: How long a value stays valid in the online store.
            0 (or negative) means "never expires".
        source: Name of the source table / event stream the features
            are computed from, e.g. ``"transactions"``.
        description: What this view is for.
        online: Whether this view is materialized to the online store.
    """

    name: str
    entity: str
    features: list[Feature]
    ttl_seconds: int = 86_400  # 24 hours by default
    source: str = ""
    description: str = ""
    online: bool = True

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def get_feature(self, name: str) -> Feature:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"feature {name!r} not in feature view {self.name!r}")


# ---------------------------------------------------------------------------
# Example feature view: transaction behaviour per user.
# These are the features the materialization job computes from raw
# transaction events, and the ones the demo / API serve.
# ---------------------------------------------------------------------------

USER_TRANSACTION_FEATURES = FeatureView(
    name="user_transaction_features",
    entity="user_id",
    source="transactions",
    ttl_seconds=7 * 86_400,  # 7 days
    description=(
        "Rolling transaction behaviour per user, computed from raw "
        "transaction events. Used for fraud-risk scoring."
    ),
    features=[
        Feature("txn_count_7d", FeatureType.INT,
                "Number of transactions in the last 7 days"),
        Feature("txn_amount_sum_7d", FeatureType.FLOAT,
                "Total transaction amount in the last 7 days"),
        Feature("txn_amount_avg_30d", FeatureType.FLOAT,
                "Average transaction amount in the last 30 days"),
        Feature("txn_count_30d", FeatureType.INT,
                "Number of transactions in the last 30 days"),
        Feature("distinct_merchants_7d", FeatureType.INT,
                "Distinct merchants in the last 7 days"),
        Feature("max_amount_7d", FeatureType.FLOAT,
                "Largest single transaction in the last 7 days"),
    ],
)

#: All feature views registered in this project.
FEATURE_VIEWS: dict[str, FeatureView] = {
    USER_TRANSACTION_FEATURES.name: USER_TRANSACTION_FEATURES,
}
