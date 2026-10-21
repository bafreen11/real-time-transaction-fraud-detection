"""Reproducible data splits for the baseline XGBoost track.

Every baseline script (train, validate, evaluate, threshold analysis,
confusion matrix, MLflow tracking) must use the same 10% stratified
sample and 80/20 split, so the logic lives here once.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    BASELINE_SAMPLE_FRACTION,
    BASELINE_TEST_FRACTION,
    RANDOM_STATE,
    SHARED_TEST_FRACTION,
    SHARED_TRAIN_LEGITIMATE_SAMPLE,
)


def baseline_sample(df):
    """Return the seeded 10% stratified sample of the dataset."""
    df_sample, _ = train_test_split(
        df,
        train_size=BASELINE_SAMPLE_FRACTION,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE
    )
    return df_sample


def baseline_train_test_split(X, y):
    """Split the baseline sample into 80% train and 20% test."""
    return train_test_split(
        X,
        y,
        test_size=BASELINE_TEST_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE
    )


def baseline_split(df, feature_columns):
    """Sample the dataset and return X_train, X_test, y_train, y_test."""
    df_sample = baseline_sample(df)

    X = df_sample[feature_columns]
    y = df_sample["isFraud"]

    return baseline_train_test_split(X, y)


def shared_split(df):
    """Return (train_sample, test_df) for the shared XGBoost/GNN track.

    The full raw dataset is split 80/20 (stratified). The training side
    keeps every fraud row plus a seeded sample of legitimate rows; the
    test side is kept whole. Extra columns (such as a transaction id)
    do not change which rows land where.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=SHARED_TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=df["isFraud"],
    )

    # Keep the full held-out test partition.
    # Sample the training partition for manageable initial training.
    fraud_train = train_df[train_df["isFraud"] == 1]
    legitimate_train = train_df[train_df["isFraud"] == 0]

    legitimate_count = min(
        SHARED_TRAIN_LEGITIMATE_SAMPLE,
        len(legitimate_train),
    )

    legitimate_sample = legitimate_train.sample(
        n=legitimate_count,
        random_state=RANDOM_STATE,
    )

    train_sample = pd.concat(
        [fraud_train, legitimate_sample],
        ignore_index=True,
    )

    return train_sample, test_df
