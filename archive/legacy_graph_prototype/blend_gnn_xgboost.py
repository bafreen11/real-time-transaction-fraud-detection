
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from ps_feature_engineering import create_features

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

PROJECT_DIR = r"E:\Transactions Fraud Detection"

TRANSACTIONS_PATH = os.path.join(
    PROJECT_DIR, "graph_transactions.csv"
)

GNN_PREDICTIONS_PATH = os.path.join(
    PROJECT_DIR, "gnn_test_predictions.csv"
)

XGBOOST_MODEL_PATH = os.path.join(
    PROJECT_DIR, "fraud_xgboost_model.joblib"
)

# --------------------------------------------------
# 2. Load the saved models and predictions
# --------------------------------------------------

print("Loading transaction data...")

df = pd.read_csv(TRANSACTIONS_PATH)
gnn_test = pd.read_csv(GNN_PREDICTIONS_PATH)

artifact = joblib.load(XGBOOST_MODEL_PATH)

# Your saved XGBoost artifact contains the model
# and its expected feature columns.
xgb_model = artifact["model"]
feature_columns = artifact["feature_columns"]

print("Transactions:", len(df))
print("GNN test predictions:", len(gnn_test))
print("XGBoost model loaded successfully.")

# --------------------------------------------------
# 3. Recreate XGBoost features
# --------------------------------------------------

df_features = create_features(df)

missing = [
    col for col in feature_columns
    if col not in df_features.columns
]

if missing:
    raise ValueError(
        f"Missing XGBoost features: {missing}"
    )

# Select exactly the feature columns used by XGBoost.
X_all = df_features[feature_columns]

# --------------------------------------------------
# 4. Select the exact GNN test transactions
# --------------------------------------------------

test_indices = gnn_test["transaction_index"].astype(int)

X_test = X_all.loc[test_indices]

# Preserve the same row order as the GNN predictions.
y_true = gnn_test["isFraud"].to_numpy()
gnn_scores = gnn_test[
    "gnn_fraud_probability"
].to_numpy()

# --------------------------------------------------
# 5. Generate XGBoost probabilities for those rows
# --------------------------------------------------

xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

print("\nBoth models scored the same transactions.")
print("Test rows:", len(y_true))

# --------------------------------------------------
# 6. Evaluate models and weighted blends
# --------------------------------------------------

results = []

# XGBoost weights to test:
# 1.00 = XGBoost only
# 0.75 = 75% XGBoost + 25% GNN
# 0.50 = equal blend
# 0.25 = 25% XGBoost + 75% GNN
# 0.00 = GNN only

for xgb_weight in [1.00, 0.75, 0.50, 0.25, 0.00]:

    gnn_weight = 1.0 - xgb_weight

    blended_scores = (
        xgb_weight * xgb_scores
        + gnn_weight * gnn_scores
    )

    predictions = (
        blended_scores >= 0.5
    ).astype(int)

    results.append({
        "XGBoost_weight": xgb_weight,
        "GNN_weight": gnn_weight,
        "Average_Precision": average_precision_score(
            y_true, blended_scores
        ),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
    })

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# --------------------------------------------------
# 7. Save the comparison and equal-blend predictions
# --------------------------------------------------

results_df.to_csv(
    os.path.join(PROJECT_DIR, "blend_comparison.csv"),
    index=False,
)

equal_blend_scores = (
    0.5 * xgb_scores + 0.5 * gnn_scores
)

pd.DataFrame({
    "transaction_index": test_indices.to_numpy(),
    "isFraud": y_true,
    "xgboost_probability": xgb_scores,
    "gnn_probability": gnn_scores,
    "blended_probability": equal_blend_scores,
}).to_csv(
    os.path.join(
        PROJECT_DIR, "blended_test_predictions.csv"
    ),
    index=False,
)

print("\nSaved comparison: blend_comparison.csv")
print("Saved predictions: blended_test_predictions.csv")
print("Blending evaluation completed.")
