"""Reproducible data splits for the baseline and shared tracks.

Every baseline script (train, validate, evaluate, threshold analysis,
evaluation figures, MLflow tracking) must use the same 10% stratified
sample and 80/20 split, and the shared-track scripts the same shared
split and validation hold-out, so the logic lives here once.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    PAYSIM_LEGITIMATE_ROWS,
    BASELINE_SAMPLE_FRACTION,
    BASELINE_TEST_FRACTION,
    BASELINE_VALIDATION_FRACTION_OF_TRAIN,
    RANDOM_STATE,
    SHARED_TEST_FRACTION,
    SHARED_TRAIN_LEGITIMATE_SAMPLE,
    SHARED_VALIDATION_FRACTION,
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


def baseline_train_val_test_split(X, y):
    """Split the baseline sample 70/10/20 into train, validation, test.

    The test rows are exactly those of baseline_train_test_split, so
    every baseline model is scored on the same test set. Returns
    X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train_val, X_test, y_train_val, y_test = baseline_train_test_split(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=BASELINE_VALIDATION_FRACTION_OF_TRAIN,
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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


def shared_validation_split(labels):
    """Return (fit_positions, validation_positions) for shared_train.csv.

    A stratified hold-out of the shared training rows. The shared
    XGBoost and GNN both train on the fit rows only and score the same
    validation rows, which then choose the GNN training strategy, the
    blend weight and the threshold. shared_test.csv is not used for
    any of these choices.
    """
    positions = np.arange(len(labels))
    fit_positions, validation_positions = train_test_split(
        positions,
        test_size=SHARED_VALIDATION_FRACTION,
        stratify=np.asarray(labels),
        random_state=RANDOM_STATE,
    )
    return np.sort(fit_positions), np.sort(validation_positions)


def shared_validation_weights(labels):
    """Sample weights that restore the natural fraud rate.

    shared_train.csv keeps every fraud row of the training partition
    but only SHARED_TRAIN_LEGITIMATE_SAMPLE of its legitimate rows (a
    uniform random sample), so its validation rows are ~6% fraud
    against 0.13% in reality. Precision, F1 and average precision
    depend on that rate, so validation metrics weight each legitimate
    row by the number of legitimate partition rows it stands for (~51).
    """
    partition_legitimate = PAYSIM_LEGITIMATE_ROWS - round(
        PAYSIM_LEGITIMATE_ROWS * SHARED_TEST_FRACTION
    )
    legitimate_weight = partition_legitimate / SHARED_TRAIN_LEGITIMATE_SAMPLE
    return np.where(np.asarray(labels) == 1, 1.0, legitimate_weight)
