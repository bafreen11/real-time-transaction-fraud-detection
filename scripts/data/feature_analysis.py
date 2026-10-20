import numpy as np
import pandas as pd

from fraud_detection.config import RAW_DATA_PATH, REPORTS_DIR
from fraud_detection.data.loading import load_data
from fraud_detection.features.engineering import (
    ENGINEERED_FEATURES,
    create_features,
)


# STEP 1: Load data and create features
df = load_data(RAW_DATA_PATH)
df_features = create_features(df)

print("\nDataset shape:", df_features.shape)


# STEP 2: Check fraud distribution
print("\n--- FRAUD DISTRIBUTION ---")

fraud_counts = df_features["isFraud"].value_counts().sort_index()

print(fraud_counts)

print("\nFraud percentage:")
print(
    (df_features["isFraud"].value_counts(normalize=True).sort_index() * 100)
    .round(4)
)


# STEP 3: Compare transaction types
print("\n--- FRAUD BY TRANSACTION TYPE ---")

type_analysis = (
    df_features.groupby("type")["isFraud"]
    .agg(total_transactions="count", fraud_transactions="sum",
         fraud_rate="mean")
)

type_analysis["fraud_rate_percent"] = (
    type_analysis["fraud_rate"] * 100
)

print(type_analysis.round(4))


# STEP 4: Compare engineered features
print("\n--- FEATURE COMPARISON BY FRAUD STATUS ---")

features = ENGINEERED_FEATURES

feature_comparison = (
    df_features.groupby("isFraud")[features]
    .agg(["mean", "median"])
    .round(4)
)

print(feature_comparison)


# STEP 5: Check missing and infinite values
print("\n--- FEATURE QUALITY CHECK ---")

print("\nMissing values:")
print(df_features[features].isnull().sum())

print("\nInfinite values:")
print(
    pd.Series(
        np.isinf(df_features[features].to_numpy()).sum(axis=0),
        index=features
    )
)


# STEP 6: Analyze transaction amounts
print("\n--- TRANSACTION AMOUNT BY FRAUD STATUS ---")

amount_analysis = (
    df_features.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print(amount_analysis)


# STEP 7: Save summary reports
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

type_analysis.to_csv(REPORTS_DIR / "fraud_by_type.csv")
feature_comparison.to_csv(REPORTS_DIR / "feature_comparison.csv")
amount_analysis.to_csv(REPORTS_DIR / "amount_analysis.csv")

print("\nAnalysis reports saved to:", REPORTS_DIR)

