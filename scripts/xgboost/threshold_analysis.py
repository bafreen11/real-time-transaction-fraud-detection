import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from fraud_detection.config import (
    FEATURES_DATA_PATH,
    THRESHOLD_RESULTS_PATH,
    XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import baseline_split

# Step 1: Load engineered dataset
df = pd.read_csv(FEATURES_DATA_PATH)

# Step 2: Load trained model and feature names
saved = joblib.load(XGBOOST_MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

# Step 3: Recreate the original sample and train-test split
X_train, X_test, y_train, y_test = baseline_split(df, feature_columns)

# Step 4: Predict fraud probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Step 5: Compare different thresholds
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "Fraud_Alerts": int(y_pred.sum())
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

ensure_output_dirs()
results_df.to_csv(THRESHOLD_RESULTS_PATH, index=False)

print("\nResults saved to:", THRESHOLD_RESULTS_PATH)
print("Average Precision (threshold-independent):",
      round(average_precision_score(y_test, y_prob), 6))
