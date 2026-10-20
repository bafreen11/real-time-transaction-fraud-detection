import pandas as pd
import joblib

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from fraud_detection.config import (
    FEATURES_DATA_PATH,
    RANDOM_STATE,
    XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import (
    baseline_sample,
    baseline_train_test_split,
)
from fraud_detection.features.engineering import FEATURE_COLUMNS

# Step 1: Load engineered data
df = pd.read_csv(FEATURES_DATA_PATH)

print("Dataset shape:", df.shape)
print("Fraud counts:")
print(df["isFraud"].value_counts())

# Step 2: Create a stratified sample for the first experiment
df_sample = baseline_sample(df)

print("\nSample shape:", df_sample.shape)
print("Sample fraud percentage:",
      df_sample["isFraud"].mean() * 100)

# Step 3: Select numeric features
feature_columns = FEATURE_COLUMNS

X = df_sample[feature_columns]
y = df_sample["isFraud"]

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = baseline_train_test_split(X, y)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# Step 5: Calculate class imbalance weight
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight = negative_count / positive_count

print("Scale positive weight:", class_weight)

# Step 6: Train XGBoost
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Step 7: Evaluate model
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\nAverage Precision (PR metric):",
      average_precision_score(y_test, y_prob))
print("Precision:", precision_score(
    y_test, y_pred, zero_division=0
))
print("Recall:", recall_score(
    y_test, y_pred, zero_division=0
))
print("F1-score:", f1_score(
    y_test, y_pred, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Step 8: Save model and feature names
ensure_output_dirs()

joblib.dump(
    {"model": model, "feature_columns": feature_columns},
    XGBOOST_MODEL_PATH
)

print("\nModel saved to:", XGBOOST_MODEL_PATH)
