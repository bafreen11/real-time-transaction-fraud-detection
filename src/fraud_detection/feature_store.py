"""Feast objects and helpers for the recipient-account feature view.

The repository config lives in feature_repo/feature_store.yaml; the
objects are defined here so the training script, the scoring service
and `feast apply` all use the same definitions.
"""

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

from fraud_detection.config import FEATURE_REPO_DIR, RECIPIENT_STATS_PATH
from fraud_detection.features.behavioral import RECIPIENT_STATE_FIELDS

FEATURE_VIEW_NAME = "recipient_stats"

recipient = Entity(
    name="recipient",
    join_keys=["account_id"],
    value_type=ValueType.STRING,
    description="Customer account receiving the transaction (nameDest)",
)

recipient_stats_source = FileSource(
    name="recipient_stats_source",
    path=RECIPIENT_STATS_PATH.as_posix(),
    timestamp_field="event_timestamp",
)

recipient_stats_view = FeatureView(
    name=FEATURE_VIEW_NAME,
    entities=[recipient],
    # The whole PaySim simulation spans 31 days.
    ttl=timedelta(days=60),
    schema=[
        Field(name="account_id", dtype=String),
        Field(name="txn_count", dtype=Int64),
        Field(name="amount_sum", dtype=Float64),
        Field(name="last_step", dtype=Int64),
        Field(name="day_txn_count", dtype=Int64),
    ],
    online=True,
    source=recipient_stats_source,
)

FEATURE_REFS = [
    f"{FEATURE_VIEW_NAME}:{field}" for field in RECIPIENT_STATE_FIELDS
]


def get_store(apply_definitions=False):
    """Open the Feast store. apply_definitions registers the entity and
    feature view first (needed on a fresh registry, e.g. in a container).
    """
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    if apply_definitions:
        store.apply([recipient, recipient_stats_view])
    return store


def get_historical_state(store, account_ids, lookup_timestamps):
    """Point-in-time recipient state for training rows.

    Returns a DataFrame aligned with the inputs (same order and length).
    """
    entity_df = pd.DataFrame({
        "account_id": pd.Series(account_ids).astype(str).to_numpy(),
        "event_timestamp": pd.to_datetime(lookup_timestamps, utc=True),
        "row_position": range(len(account_ids)),
    })

    state = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    # The file offline store drops entity rows with no earlier feature
    # row (an account's first transaction). Put them back as empty
    # state, which is what "no history" means.
    return (
        state.set_index("row_position")
        .reindex(range(len(entity_df)))
        .reset_index(drop=True)[RECIPIENT_STATE_FIELDS]
    )


def get_online_state(store, account_ids):
    """Latest recipient state from the online store, one row per id."""
    response = store.get_online_features(
        features=FEATURE_REFS,
        entity_rows=[{"account_id": str(a)} for a in account_ids],
    ).to_df()
    return response[RECIPIENT_STATE_FIELDS]


def write_online_state(store, rows):
    """Write recipient states to the online store.

    rows: list of dicts with account_id, event_timestamp and
    RECIPIENT_STATE_FIELDS.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    store.write_to_online_store(FEATURE_VIEW_NAME, df)
