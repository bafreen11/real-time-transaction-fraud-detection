import pandas as pd

from fraud_detection.config import (
    SHARED_BLEND_COMPARISON_PATH as SELECTION_PATH,
    SHARED_BLEND_CONFUSION_MATRIX_PATH as CONFUSION_MATRIX_PATH,
    SHARED_BLEND_PR_CURVE_PATH as PR_CURVE_PATH,
    SHARED_BLEND_TEST_RESULTS_PATH as TEST_RESULTS_PATH,
    SHARED_BLEND_THRESHOLDS,
    SHARED_BLEND_XGB_WEIGHTS,
    SHARED_BLENDED_PREDICTIONS_PATH as PREDICTIONS_PATH,
    SHARED_GNN_PREDICTIONS_PATH as GNN_TEST_PATH,
    SHARED_GNN_VALIDATION_PREDICTIONS_PATH as GNN_VAL_PATH,
    SHARED_XGB_PREDICTIONS_PATH as XGB_TEST_PATH,
    SHARED_XGB_VALIDATION_PREDICTIONS_PATH as XGB_VAL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import shared_validation_weights
from fraud_detection.evaluation import (
    plot_confusion_counts,
    plot_pr_curves,
    threshold_metrics,
)

# Model selection uses the validation rows held out of shared_train.csv
# (see splits.shared_validation_split). The XGBoost weight and the
# decision threshold are fixed there; the test set is scored once,
# afterwards, and is not used to choose anything.
#
# The validation rows come from shared_train.csv, which is ~6% fraud
# (all fraud, sampled legitimate rows). Their metrics are weighted back
# to the natural 0.13% fraud rate (splits.shared_validation_weights),
# otherwise the chosen threshold would raise far too many alerts on
# real traffic. Weighted counts in the selection report are estimates.


def load_aligned(xgb_path, gnn_path):
    """XGBoost and GNN predictions that refer to identical rows."""
    xgb_df = pd.read_csv(xgb_path)
    gnn_df = pd.read_csv(gnn_path)

    if len(xgb_df) != len(gnn_df):
        raise ValueError("Prediction files have different row counts.")

    if not xgb_df["transaction_index"].equals(
        gnn_df["transaction_index"]
    ):
        raise ValueError("Transaction indices do not match.")

    if not xgb_df["isFraud"].equals(gnn_df["isFraud"]):
        raise ValueError("Labels do not match.")

    return (
        xgb_df["transaction_index"],
        xgb_df["isFraud"].to_numpy(),
        xgb_df["xgboost_probability"].to_numpy(),
        gnn_df["gnn_fraud_probability"].to_numpy(),
    )


def blend(xgb_scores, gnn_scores, xgb_weight):
    return xgb_weight * xgb_scores + (1.0 - xgb_weight) * gnn_scores


def best_threshold(y_true, scores, sample_weight):
    """F1-best threshold (ties: higher threshold, fewer alerts)."""
    rows = [threshold_metrics(y_true, scores, t, sample_weight)
            for t in SHARED_BLEND_THRESHOLDS]
    return max(rows, key=lambda row: (row["f1"], row["threshold"]))


# --------------------------------------------------
# 1. Choose the blend weight and threshold on validation
# --------------------------------------------------

print("Loading validation predictions...")
_, y_val, xgb_val, gnn_val = load_aligned(XGB_VAL_PATH, GNN_VAL_PATH)

print("Validation transactions:", len(y_val))
print("Validation fraud cases:", int(y_val.sum()))

val_weights = shared_validation_weights(y_val)

selection = []

for xgb_weight in SHARED_BLEND_XGB_WEIGHTS:
    scores = blend(xgb_val, gnn_val, xgb_weight)
    for threshold in SHARED_BLEND_THRESHOLDS:
        selection.append({
            "XGBoost_weight": xgb_weight,
            "GNN_weight": round(1.0 - xgb_weight, 2),
            **threshold_metrics(y_val, scores, threshold, val_weights),
        })

selection_df = pd.DataFrame(selection)

# Highest validation F1; ties go to higher average precision, then the
# simpler model (more XGBoost weight), then the higher threshold.
ranked = selection_df.sort_values(
    ["f1", "average_precision", "XGBoost_weight", "threshold"],
    ascending=False,
)
best = ranked.iloc[0]

selected_weight = float(best["XGBoost_weight"])
selected_threshold = float(best["threshold"])

print("\n========== VALIDATION (WEIGHTED): BEST THRESHOLD PER WEIGHT "
      "==========")
# Same tie rule as the overall choice
per_weight = ranked.drop_duplicates("XGBoost_weight").sort_values(
    "XGBoost_weight", ascending=False
)
print(per_weight[[
    "XGBoost_weight", "GNN_weight", "threshold", "average_precision",
    "precision", "recall", "f1",
]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

print(f"\nSelected on validation: XGBoost weight {selected_weight}, "
      f"GNN weight {1 - selected_weight:.2f}, "
      f"threshold {selected_threshold}")

# Single models at their own validation-chosen thresholds, as
# reference rows for the test report (also fixed before testing).
xgb_threshold = best_threshold(y_val, xgb_val, val_weights)["threshold"]
gnn_threshold = best_threshold(y_val, gnn_val, val_weights)["threshold"]

# --------------------------------------------------
# 2. Score the test set once with the fixed choices
# --------------------------------------------------

print("\nLoading shared test predictions...")
test_index, y_test, xgb_test, gnn_test = load_aligned(
    XGB_TEST_PATH, GNN_TEST_PATH
)

print("Test transactions:", len(y_test))
print("Test fraud cases:", int(y_test.sum()))

blend_test = blend(xgb_test, gnn_test, selected_weight)

test_results = pd.DataFrame([
    {"model": "selected_blend", "XGBoost_weight": selected_weight,
     **threshold_metrics(y_test, blend_test, selected_threshold)},
    {"model": "xgboost_only", "XGBoost_weight": 1.0,
     **threshold_metrics(y_test, xgb_test, xgb_threshold)},
    {"model": "gnn_only", "XGBoost_weight": 0.0,
     **threshold_metrics(y_test, gnn_test, gnn_threshold)},
])

print("\n========== SHARED TEST RESULTS (choices fixed on validation) ==========")
print(test_results.to_string(
    index=False, float_format=lambda value: f"{value:.6f}"
))

# --------------------------------------------------
# 3. Save reports, figures and the selected blend's predictions
# --------------------------------------------------

ensure_output_dirs()
selection_df.to_csv(SELECTION_PATH, index=False)
test_results.to_csv(TEST_RESULTS_PATH, index=False)

selected_row = test_results.iloc[0]

# A weight of 1 or 0 is a single model; don't draw it twice.
if selected_weight in (0.0, 1.0):
    selected_name = (
        "XGBoost only" if selected_weight == 1.0 else "GNN only"
    ) + " (selected)"
    curves = [
        (selected_name, y_test, blend_test),
        ("GNN only", y_test, gnn_test) if selected_weight == 1.0
        else ("XGBoost only", y_test, xgb_test),
    ]
else:
    selected_name = (
        f"Blend XGBoost {selected_weight:g} / GNN {1 - selected_weight:g}"
    )
    curves = [
        (selected_name, y_test, blend_test),
        ("XGBoost only", y_test, xgb_test),
        ("GNN only", y_test, gnn_test),
    ]

plot_pr_curves(
    curves,
    PR_CURVE_PATH,
    "Shared track: precision-recall on the shared test set",
    operating_point=(
        f"Threshold {selected_threshold:g} (chosen on validation)",
        selected_row["recall"],
        selected_row["precision"],
    ),
)
plot_confusion_counts(
    selected_row.to_dict(),
    CONFUSION_MATRIX_PATH,
    f"Shared track, selected model: {selected_name.removesuffix(' (selected)')}",
)

pd.DataFrame({
    "transaction_index": test_index,
    "isFraud": y_test,
    "xgboost_probability": xgb_test,
    "gnn_probability": gnn_test,
    "blended_probability": blend_test,
    "predicted_label": (blend_test >= selected_threshold).astype(int),
}).to_csv(PREDICTIONS_PATH, index=False)

print("\nSaved validation selection:", SELECTION_PATH)
print("Saved test results:", TEST_RESULTS_PATH)
print("Saved PR curve:", PR_CURVE_PATH)
print("Saved confusion matrix:", CONFUSION_MATRIX_PATH)
print("Saved selected blend predictions:", PREDICTIONS_PATH)
print("Shared blending finished.")
