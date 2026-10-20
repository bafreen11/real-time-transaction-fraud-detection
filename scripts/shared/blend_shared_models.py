import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from fraud_detection.config import (
    SHARED_BLEND_COMPARISON_PATH as COMPARISON_PATH,
    SHARED_BLENDED_PREDICTIONS_PATH as PREDICTIONS_PATH,
    SHARED_GNN_PREDICTIONS_PATH as GNN_PATH,
    SHARED_XGB_PREDICTIONS_PATH as XGB_PATH,
    ensure_output_dirs,
)

print("Loading shared test predictions...")
xgb_df = pd.read_csv(XGB_PATH)
gnn_df = pd.read_csv(GNN_PATH)

# Confirm that both predictions refer to identical test rows.
if len(xgb_df) != len(gnn_df):
    raise ValueError("Prediction files have different row counts.")

if not xgb_df["transaction_index"].equals(
    gnn_df["transaction_index"]
):
    raise ValueError("Transaction indices do not match.")

if not xgb_df["isFraud"].equals(gnn_df["isFraud"]):
    raise ValueError("Test labels do not match.")

y_true = xgb_df["isFraud"].to_numpy()
xgb_scores = xgb_df["xgboost_probability"].to_numpy()
gnn_scores = gnn_df["gnn_fraud_probability"].to_numpy()

print("Matched test transactions:", len(y_true))
print("Test fraud cases:", int(y_true.sum()))

# These are exploratory comparisons.
# Do not select a final weight using the test set alone.
results = []

for xgb_weight in [1.00, 0.75, 0.50, 0.25, 0.00]:
    gnn_weight = 1.0 - xgb_weight

    blended_scores = (
        xgb_weight * xgb_scores
        + gnn_weight * gnn_scores
    )

    predictions = (blended_scores >= 0.5).astype(int)

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

print("\n========== SHARED MODEL COMPARISON ==========")
print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

ensure_output_dirs()
results_df.to_csv(COMPARISON_PATH, index=False)

# Save a 75% XGBoost / 25% GNN blend as a candidate.
# This is not automatically the best model.
candidate_scores = 0.75 * xgb_scores + 0.25 * gnn_scores

pd.DataFrame({
    "transaction_index": xgb_df["transaction_index"],
    "isFraud": y_true,
    "xgboost_probability": xgb_scores,
    "gnn_probability": gnn_scores,
    "blended_probability": candidate_scores,
}).to_csv(PREDICTIONS_PATH, index=False)

print("\nSaved comparison:", COMPARISON_PATH)
print("Saved candidate predictions:", PREDICTIONS_PATH)
print("Shared blending evaluation completed.")
