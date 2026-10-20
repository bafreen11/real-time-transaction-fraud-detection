"""Reproducible data splits for the baseline XGBoost track.

Every baseline script (train, validate, evaluate, threshold analysis,
confusion matrix, MLflow tracking) must use the same 10% stratified
sample and 80/20 split, so the logic lives here once.
"""

from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    BASELINE_SAMPLE_FRACTION,
    BASELINE_TEST_FRACTION,
    RANDOM_STATE,
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
