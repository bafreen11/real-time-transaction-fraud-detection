from typing import Literal

import pandas as pd

# The six raw numeric PaySim fields of a transaction.
RAW_TRANSACTION_FIELDS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Fields a transaction must provide for scoring (the API inputs).
TRANSACTION_FIELDS = RAW_TRANSACTION_FIELDS + ["type"]

# The five PaySim transaction types.
TransactionType = Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

# Features created by create_features().
ENGINEERED_FEATURES = [
    "origin_balance_error",
    "destination_balance_error",
    "amount_to_origin_balance",
    "amount_to_destination_balance",
    "step_day",
    "step_hour",
]

# PaySim fraud only occurs in TRANSFER and CASH_OUT transactions, so
# those two types get indicator features (also created by
# create_features). Adopted after scripts/xgboost/compare_transaction_type.py
# showed better validation results than the 12 features without them.
TYPE_FEATURES = ["type_TRANSFER", "type_CASH_OUT"]

# Exact feature names and order used to train the XGBoost models.
FEATURE_COLUMNS = RAW_TRANSACTION_FIELDS + ENGINEERED_FEATURES + TYPE_FEATURES


def create_features(df):
    """Create features for PaySim fraud detection."""

    df = df.copy()

    # 1. Sender balance difference
    df["origin_balance_error"] = (
        df["oldbalanceOrg"]
        - df["amount"]
        - df["newbalanceOrig"]
    )

    # 2. Recipient balance difference
    df["destination_balance_error"] = (
        df["oldbalanceDest"]
        + df["amount"]
        - df["newbalanceDest"]
    )

    # 3. Amount relative to sender's old balance
    df["amount_to_origin_balance"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )

    # 4. Amount relative to recipient's old balance
    df["amount_to_destination_balance"] = (
        df["amount"] / (df["oldbalanceDest"] + 1)
    )

    # 5. Extract simulated time features
    df["step_day"] = (df["step"] - 1) // 24
    df["step_hour"] = (df["step"] - 1) % 24

    # 6. Transaction type indicators
    for type_feature in TYPE_FEATURES:
        df[type_feature] = (
            df["type"] == type_feature.removeprefix("type_")
        ).astype("int64")

    return df


def prepare_transaction(transaction, feature_columns):
    """Build a one-row model input from a dict of TRANSACTION_FIELDS."""
    df = create_features(pd.DataFrame([transaction]))
    return df[feature_columns]
