import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from fraud_detection.config import (
    BASELINE_THRESHOLDS,
    FEATURES_DATA_PATH,
    RANDOM_STATE,
    TRANSACTION_TYPE_COMPARISON_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import (
    baseline_sample,
    baseline_train_val_test_split,
)
from fraud_detection.features.engineering import (
    FEATURE_COLUMNS,
    TYPE_FEATURES,
)

# Model-selection experiment: does adding transaction type help the
# baseline XGBoost model? Both candidates are trained on the same 70%
# and compared on the same 10% validation set. The 20% test set is not
# used here; it is scored once by validate_and_test_model.py for the
# feature set that is adopted (with type, now in FEATURE_COLUMNS).

# Step 1: Load the engineered data and recreate the baseline sample
df = pd.read_csv(
    FEATURES_DATA_PATH,
    usecols=FEATURE_COLUMNS + ["type", "isFraud"],
)
df_sample = baseline_sample(df)
del df

X = df_sample[FEATURE_COLUMNS + ["type"]]
y = df_sample["isFraud"]

X_train, X_val, _, y_train, y_val, _ = baseline_train_val_test_split(X, y)

print("Training rows:", len(X_train), "| fraud:", int(y_train.sum()))
print("Validation rows:", len(X_val), "| fraud:", int(y_val.sum()))

class_weight = (y_train == 0).sum() / (y_train == 1).sum()

candidates = {
    "baseline_12_features": [
        c for c in FEATURE_COLUMNS if c not in TYPE_FEATURES
    ],
    "with_transaction_type": FEATURE_COLUMNS,
}

# Step 2: Train each candidate and score the validation set
results = []

for name, columns in candidates.items():
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
    model.fit(X_train[columns], y_train)
    val_prob = model.predict_proba(X_val[columns])[:, 1]

    # Same rule as validate_and_test_model.py: F1-best threshold
    f1_by_threshold = [
        f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
        for t in BASELINE_THRESHOLDS
    ]
    threshold = BASELINE_THRESHOLDS[int(np.argmax(f1_by_threshold))]
    val_pred = (val_prob >= threshold).astype(int)

    row = {
        "candidate": name,
        "feature_count": len(columns),
        "val_average_precision": average_precision_score(y_val, val_prob),
        "best_threshold": threshold,
        "val_precision": precision_score(y_val, val_pred, zero_division=0),
        "val_recall": recall_score(y_val, val_pred, zero_division=0),
        "val_f1": f1_score(y_val, val_pred, zero_division=0),
        "val_false_positives": int(((val_pred == 1) & (y_val == 0)).sum()),
        "val_missed_fraud": int(((val_pred == 0) & (y_val == 1)).sum()),
    }

    # Fraud patterns differ by type, so also check within each fraud type
    for fraud_type in ["TRANSFER", "CASH_OUT"]:
        mask = (X_val["type"] == fraud_type).to_numpy()
        row[f"val_average_precision_{fraud_type}"] = (
            average_precision_score(y_val[mask], val_prob[mask])
        )

    results.append(row)

results_df = pd.DataFrame(results)

print("\nVALIDATION COMPARISON (test set not used)")
print(results_df.T.to_string(header=False))

ensure_output_dirs()
results_df.to_csv(TRANSACTION_TYPE_COMPARISON_PATH, index=False)
print("\nSaved comparison:", TRANSACTION_TYPE_COMPARISON_PATH)
