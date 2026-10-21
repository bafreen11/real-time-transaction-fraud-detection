"""Feature code shared by training and serving."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import (
    FEATURE_COLUMNS_PATH,
    STREAM_XGBOOST_MODEL_PATH,
    XGBOOST_MODEL_PATH,
    XGBOOST_VALIDATED_MODEL_PATH,
)
from fraud_detection.features.behavioral import (
    RECIPIENT_STATE_FIELDS,
    add_event_timestamps,
    build_recipient_snapshots,
    is_tracked_recipient,
    recipient_features,
    update_recipient_state,
)
from fraud_detection.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    prepare_transaction,
)
from fraud_detection.features.stream import (
    STREAM_FEATURE_COLUMNS,
    build_stream_features,
)

TRANSACTION = {
    "step": 26,
    "amount": 1000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 200.0,
    "newbalanceDest": 1200.0,
}


def empty_state(rows=1):
    return pd.DataFrame(np.nan, index=range(rows), columns=RECIPIENT_STATE_FIELDS)


# --------------------------------------------------
# Baseline features (12)
# --------------------------------------------------

def test_engineered_feature_values():
    row = create_features(pd.DataFrame([TRANSACTION])).iloc[0]

    assert row["origin_balance_error"] == 0         # 5000 - 1000 - 4000
    assert row["destination_balance_error"] == 0    # 200 + 1000 - 1200
    assert row["amount_to_origin_balance"] == pytest.approx(1000 / 5001)
    assert row["amount_to_destination_balance"] == pytest.approx(1000 / 201)
    assert row["step_day"] == 1                     # step 26 = day 1, hour 1
    assert row["step_hour"] == 1


def test_prepare_transaction_keeps_training_order():
    X = prepare_transaction(TRANSACTION, FEATURE_COLUMNS)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (1, 12)


@pytest.mark.parametrize(
    "path", [XGBOOST_MODEL_PATH, XGBOOST_VALIDATED_MODEL_PATH]
)
def test_saved_baseline_models_use_feature_contract(path):
    assert joblib.load(path)["feature_columns"] == FEATURE_COLUMNS


def test_feature_columns_json_matches_code():
    assert json.loads(FEATURE_COLUMNS_PATH.read_text()) == FEATURE_COLUMNS


# --------------------------------------------------
# Recipient behavioral features (Feast state)
# --------------------------------------------------

def test_only_customer_recipients_are_tracked():
    assert list(is_tracked_recipient(["C123", "M456"])) == [True, False]


def test_streaming_state_update_matches_offline_snapshots():
    """update_recipient_state (service) must equal
    build_recipient_snapshots (training), or features would drift."""
    df = add_event_timestamps(pd.DataFrame({
        "step":     [1, 1, 3, 30, 30, 31],
        "amount":   [100.0, 50.0, 25.0, 10.0, 70.0, 5.0],
        "nameDest": ["C1", "M9", "C1", "C2", "C1", "C1"],
    }))
    offline = build_recipient_snapshots(df)[RECIPIENT_STATE_FIELDS]

    state, streamed = {}, []
    for account, step, amount in zip(df["nameDest"], df["step"], df["amount"]):
        if not is_tracked_recipient([account])[0]:
            continue
        state[account] = update_recipient_state(
            state.get(account, {}), step, amount
        )
        streamed.append(state[account])

    np.testing.assert_allclose(
        pd.DataFrame(streamed)[RECIPIENT_STATE_FIELDS].to_numpy(float),
        offline.to_numpy(float),
    )


def test_recipient_features_without_history():
    txn = pd.DataFrame([{"step": 5, "amount": 300.0}])
    features = recipient_features(txn, empty_state()).iloc[0]

    assert features["dest_txn_count_prev"] == 0
    assert features["dest_txn_count_same_day"] == 0
    assert features["dest_steps_since_last_txn"] == -1
    assert features["dest_amount_mean_prev"] == 0
    assert features["amount_to_dest_avg"] == 300.0


def test_recipient_features_with_history():
    txn = pd.DataFrame([{"step": 10, "amount": 400.0}])
    state = pd.DataFrame([{
        "txn_count": 3, "amount_sum": 300.0, "last_step": 8, "day_txn_count": 2,
    }])
    features = recipient_features(txn, state).iloc[0]

    assert features["dest_txn_count_prev"] == 3
    assert features["dest_txn_count_same_day"] == 2  # steps 8 and 10: same day
    assert features["dest_steps_since_last_txn"] == 2
    assert features["dest_amount_mean_prev"] == 100.0
    assert features["amount_to_dest_avg"] == pytest.approx(400 / 101)


# --------------------------------------------------
# Stream model features (19)
# --------------------------------------------------

def test_build_stream_features_columns_and_type_flags():
    txn = pd.DataFrame([{
        **TRANSACTION, "type": "TRANSFER", "nameOrig": "C10", "nameDest": "C20",
    }])
    X = build_stream_features(txn, empty_state())

    assert list(X.columns) == STREAM_FEATURE_COLUMNS
    assert len(STREAM_FEATURE_COLUMNS) == 19
    assert X.loc[0, "type_TRANSFER"] == 1
    assert X.loc[0, "type_CASH_OUT"] == 0


def test_stream_model_uses_stream_feature_contract():
    artifact = joblib.load(STREAM_XGBOOST_MODEL_PATH)
    assert artifact["feature_columns"] == STREAM_FEATURE_COLUMNS
    assert 0 <= artifact["gnn_weight"] <= 1
    assert 0 < artifact["threshold"] < 1
