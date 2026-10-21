import joblib
import pandas as pd

from fraud_detection.config import (
    CONFUSION_MATRIX_PATH,
    FEATURES_DATA_PATH,
    PR_CURVE_PATH,
    SERVED_MODEL_TEST_RESULTS_PATH,
    XGBOOST_VALIDATED_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import baseline_split
from fraud_detection.evaluation import (
    plot_confusion_counts,
    plot_pr_curves,
    threshold_metrics,
)

# Figures and test metrics for the model the API serves
# (fraud_xgboost_validated_model.joblib) at its validation-chosen
# threshold, on the baseline 20% test split.

print("Loading dataset and served model...")
saved = joblib.load(XGBOOST_VALIDATED_MODEL_PATH)

model = saved["model"]
features = saved["feature_columns"]
threshold = float(saved["threshold"])

df = pd.read_csv(FEATURES_DATA_PATH, usecols=features + ["isFraud"])

# Reconstruct the same seeded 10% sample and test split
X_train, X_test, y_train, y_test = baseline_split(df, features)
del df

probabilities = model.predict_proba(X_test)[:, 1]
metrics = threshold_metrics(y_test, probabilities, threshold)

print("Test rows:", len(y_test), "| fraud:", int(y_test.sum()))
print(pd.Series(metrics).to_string())

ensure_output_dirs()

pd.DataFrame([{
    "model_artifact": XGBOOST_VALIDATED_MODEL_PATH.name,
    "test_rows": len(y_test),
    "test_fraud": int(y_test.sum()),
    **metrics,
}]).to_csv(SERVED_MODEL_TEST_RESULTS_PATH, index=False)

plot_confusion_counts(
    metrics,
    CONFUSION_MATRIX_PATH,
    "Served XGBoost model, baseline test set",
)
plot_pr_curves(
    [("Served XGBoost model", y_test, probabilities)],
    PR_CURVE_PATH,
    "Served XGBoost model: precision-recall on the baseline test set",
    operating_point=(
        f"Chosen threshold {threshold:g}",
        metrics["recall"],
        metrics["precision"],
    ),
)

print("Saved test results to:", SERVED_MODEL_TEST_RESULTS_PATH)
print("Saved confusion matrix to:", CONFUSION_MATRIX_PATH)
print("Saved PR curve to:", PR_CURVE_PATH)
