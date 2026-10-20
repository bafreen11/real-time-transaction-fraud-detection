import json

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from fraud_detection.config import (
    BASELINE_SAMPLE_FRACTION,
    BASELINE_TEST_FRACTION,
    FEATURE_COLUMNS_PATH,
    FEATURES_DATA_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    XGBOOST_MODEL_PATH,
)
from fraud_detection.data.splits import baseline_split

# Load the saved model and feature names
saved = joblib.load(XGBOOST_MODEL_PATH)
model = saved["model"]
feature_columns = saved["feature_columns"]

# Load the same dataset and recreate the baseline sample/split
df = pd.read_csv(FEATURES_DATA_PATH)

X_train, X_test, y_train, y_test = baseline_split(df, feature_columns)

# Re-evaluate the saved model on the same split
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    "average_precision": float(
        average_precision_score(y_test, y_prob)
    ),
    "fraud_precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "fraud_recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "fraud_f1": float(
        f1_score(y_test, y_pred, zero_division=0)
    )
}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name="xgboost_baseline_sample"):
    mlflow.log_param("model_type", "XGBClassifier")
    mlflow.log_param("dataset", "PaySim")
    mlflow.log_param("sample_fraction", BASELINE_SAMPLE_FRACTION)
    mlflow.log_param("test_fraction", BASELINE_TEST_FRACTION)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("threshold", 0.5)
    mlflow.log_param("feature_count", len(feature_columns))

    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    mlflow.log_artifact(str(XGBOOST_MODEL_PATH))

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)

    mlflow.log_artifact(str(FEATURE_COLUMNS_PATH))

    print("MLflow run logged successfully.")
    print("Metrics:", metrics)
    print("Run ID:", mlflow.active_run().info.run_id)

print("\nStart the MLflow dashboard from the project root using:")
print("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
