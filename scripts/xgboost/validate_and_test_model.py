import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from fraud_detection.config import (
    FEATURES_DATA_PATH,
    RANDOM_STATE,
    VALIDATED_THRESHOLD_RESULTS_PATH,
    XGBOOST_VALIDATED_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import (
    baseline_sample,
    baseline_train_test_split,
)
from fraud_detection.features.engineering import FEATURE_COLUMNS

# Step 1: Load engineered data
df = pd.read_csv(FEATURES_DATA_PATH)

feature_columns = FEATURE_COLUMNS

# Step 2: Recreate the same 10% stratified sample
df_sample = baseline_sample(df)

X = df_sample[feature_columns]
y = df_sample["isFraud"]

# Step 3: Reserve 20% for the final test set
X_train_val, X_test, y_train_val, y_test = baseline_train_test_split(X, y)

# Step 4: Split remaining data into training and validation
# Overall proportions: 70% train, 10% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.125,
    stratify=y_train_val,
    random_state=RANDOM_STATE
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# Step 5: Calculate class imbalance weight using training data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
class_weight = negative_count / positive_count

# Step 6: Train a fresh XGBoost model
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

# Step 7: Select threshold using validation data only
val_prob = model.predict_proba(X_val)[:, 1]

thresholds = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90
]

threshold_results = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, val_pred, zero_division=0
        )
    })

results_df = pd.DataFrame(threshold_results)
print("\nVALIDATION THRESHOLD RESULTS")
print(results_df.round(4).to_string(index=False))

best_row = results_df.loc[results_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\nSelected threshold:", best_threshold)
print("Validation F1:", round(float(best_row["f1"]), 4))

# Step 8: Final evaluation on the untouched test set
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Threshold:", best_threshold)
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, test_pred, zero_division=0), 4
))
print("Average Precision:", round(
    average_precision_score(y_test, test_prob), 6
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# Step 9: Save separately; do not overwrite the existing model
ensure_output_dirs()

joblib.dump({
    "model": model,
    "feature_columns": feature_columns,
    "threshold": best_threshold
}, XGBOOST_VALIDATED_MODEL_PATH)

results_df.to_csv(VALIDATED_THRESHOLD_RESULTS_PATH, index=False)

print("\nValidated model saved to:", XGBOOST_VALIDATED_MODEL_PATH)
print("Validation threshold results saved to:",
      VALIDATED_THRESHOLD_RESULTS_PATH)
