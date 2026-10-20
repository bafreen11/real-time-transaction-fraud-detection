import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from fraud_detection.config import (
    RANDOM_STATE,
    SHARED_TEST_PATH as TEST_PATH,
    SHARED_TRAIN_PATH as TRAIN_PATH,
    SHARED_XGB_PREDICTIONS_PATH as PREDICTIONS_PATH,
    SHARED_XGBOOST_MODEL_PATH as MODEL_PATH,
    XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.features.engineering import create_features

print("Loading shared training and test data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Engineer features separately for each partition.
train_features_df = create_features(train_df)
test_features_df = create_features(test_df)

# Use the same feature list as the existing XGBoost model.
old_artifact = joblib.load(XGBOOST_MODEL_PATH)
feature_columns = old_artifact["feature_columns"]

missing_train = [
    col for col in feature_columns
    if col not in train_features_df.columns
]
missing_test = [
    col for col in feature_columns
    if col not in test_features_df.columns
]

if missing_train or missing_test:
    raise ValueError(
        f"Missing training features: {missing_train}; "
        f"missing test features: {missing_test}"
    )

X_train = train_features_df[feature_columns]
y_train = train_df["isFraud"].astype(int)

X_test = test_features_df[feature_columns]
y_test = test_df["isFraud"].astype(int).to_numpy()

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

print("Training rows:", len(X_train))
print("Training fraud cases:", positive_count)
print("Test rows:", len(X_test))
print("Test fraud cases:", int(y_test.sum()))
print("Scale positive weight:", scale_pos_weight)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

print("\nTraining shared XGBoost model...")
model.fit(X_train, y_train)

print("Scoring shared test transactions...")
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n========== SHARED XGBOOST TEST RESULTS ==========")
print(
    "Average precision:",
    average_precision_score(y_test, probabilities),
)
print(
    "Fraud precision:",
    precision_score(y_test, predictions, zero_division=0),
)
print(
    "Fraud recall:",
    recall_score(y_test, predictions, zero_division=0),
)
print(
    "Fraud F1:",
    f1_score(y_test, predictions, zero_division=0),
)
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save a new model; do not overwrite the original artifact.
ensure_output_dirs()

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
    },
    MODEL_PATH,
)

# Row positions match shared_test.csv and the shared GNN output.
pd.DataFrame({
    "transaction_index": np.arange(len(test_df)),
    "isFraud": y_test,
    "xgboost_probability": probabilities,
}).to_csv(PREDICTIONS_PATH, index=False)

print("\nSaved model:", MODEL_PATH)
print("Saved predictions:", PREDICTIONS_PATH)
print("Shared XGBoost training and evaluation finished.")
