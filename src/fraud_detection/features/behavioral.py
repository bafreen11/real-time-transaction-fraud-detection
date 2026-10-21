"""Recipient-account behavioral features backed by the Feast store.

PaySim senders almost never repeat (99.7% of nameOrig values appear
once), but recipients do, so account history is tracked per recipient
(nameDest). Merchant recipients ("M...") only receive PAYMENTs, which
are never fraud, so only customer recipients ("C...") are tracked.

Feast stores the raw running state of each recipient account
(RECIPIENT_STATE_FIELDS). The model features are derived from that
state and the incoming transaction by recipient_features(), which is
used by training and by the scoring service alike, so both see the
exact same numbers.

Event order: transactions are processed in file order. Each row gets
a unique timestamp (its step's start hour plus its position within
the step in milliseconds). A transaction sees the state produced by
every earlier row, including earlier rows of the same step, which is
also what a sequential stream consumer sees.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from fraud_detection.config import PAYSIM_START_TIME

# Running state per recipient account, stored in Feast.
RECIPIENT_STATE_FIELDS = [
    "txn_count",
    "amount_sum",
    "last_step",
    "day_txn_count",
]

# Model features derived from the state.
BEHAVIORAL_FEATURES = [
    "dest_txn_count_prev",
    "dest_txn_count_same_day",
    "dest_steps_since_last_txn",
    "dest_amount_mean_prev",
    "amount_to_dest_avg",
]


def is_tracked_recipient(account_ids):
    """True for customer recipients, whose history is kept in Feast."""
    return pd.Series(account_ids).astype(str).str.startswith("C").to_numpy()


def step_day(step):
    return (np.asarray(step) - 1) // 24


def add_event_timestamps(df):
    """Add transaction_id and event/lookup timestamps to raw PaySim rows.

    df must hold the complete dataset in original file order.
    event_timestamp is when the transaction happens (and when the
    recipient snapshot it produces becomes valid). lookup_timestamp is
    one millisecond earlier, so a point-in-time join returns the state
    from all earlier rows but not from the row itself.
    """
    df = df.copy()
    df["transaction_id"] = np.arange(len(df), dtype=np.int64)

    position_in_step = df.groupby("step").cumcount()

    df["event_timestamp"] = (
        pd.Timestamp(PAYSIM_START_TIME)
        + pd.to_timedelta(df["step"] - 1, unit="h")
        + pd.to_timedelta(position_in_step, unit="ms")
    )
    df["lookup_timestamp"] = df["event_timestamp"] - timedelta(milliseconds=1)
    return df


def build_recipient_snapshots(df):
    """Recipient state after every transaction, one row per transaction.

    df is the output of add_event_timestamps(). Returns the Feast offline
    source: account_id, event_timestamp and RECIPIENT_STATE_FIELDS.
    """
    tracked = df.loc[
        is_tracked_recipient(df["nameDest"]),
        ["nameDest", "step", "amount", "event_timestamp"],
    ]

    by_account = tracked.groupby("nameDest", sort=False)
    day = step_day(tracked["step"])

    return pd.DataFrame({
        "account_id": tracked["nameDest"].to_numpy(),
        "event_timestamp": tracked["event_timestamp"].to_numpy(),
        "txn_count": (by_account.cumcount() + 1).to_numpy(np.int64),
        "amount_sum": by_account["amount"].cumsum().to_numpy(np.float64),
        "last_step": tracked["step"].to_numpy(np.int64),
        "day_txn_count": (
            tracked.groupby([tracked["nameDest"], day], sort=False)
            .cumcount() + 1
        ).to_numpy(np.int64),
    })


def recipient_features(transactions, state):
    """Derive BEHAVIORAL_FEATURES from transactions and recipient state.

    transactions: DataFrame with step and amount.
    state: DataFrame aligned row-for-row with RECIPIENT_STATE_FIELDS;
    missing values mean the account has no history.
    """
    step = transactions["step"].to_numpy(np.int64)
    amount = transactions["amount"].to_numpy(np.float64)

    count = state["txn_count"].fillna(0).to_numpy(np.float64)
    amount_sum = state["amount_sum"].fillna(0).to_numpy(np.float64)
    last_step = state["last_step"].fillna(-1).to_numpy(np.int64)
    day_count = state["day_txn_count"].fillna(0).to_numpy(np.float64)

    has_history = count > 0
    mean_prev = np.where(has_history, amount_sum / np.maximum(count, 1), 0.0)
    same_day = has_history & (step_day(last_step) == step_day(step))

    return pd.DataFrame({
        "dest_txn_count_prev": count,
        "dest_txn_count_same_day": np.where(same_day, day_count, 0.0),
        "dest_steps_since_last_txn": np.where(
            has_history, step - last_step, -1
        ).astype(np.float64),
        "dest_amount_mean_prev": mean_prev,
        "amount_to_dest_avg": amount / (mean_prev + 1),
    }, index=transactions.index)


def update_recipient_state(state, step, amount):
    """State after one more transaction; the streaming counterpart of
    build_recipient_snapshots(). state is a dict (empty for a new account).
    """
    count = state.get("txn_count") or 0
    last_step = state.get("last_step")
    same_day = (
        count > 0
        and last_step is not None
        and step_day(last_step) == step_day(step)
    )

    return {
        "txn_count": int(count) + 1,
        "amount_sum": float(state.get("amount_sum") or 0.0) + float(amount),
        "last_step": int(step),
        "day_txn_count": (
            int(state.get("day_txn_count") or 0) + 1 if same_day else 1
        ),
    }
