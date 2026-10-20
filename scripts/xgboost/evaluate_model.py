import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from fraud_detection.config import (
    EVALUATION_RESULTS_PATH,
    FEATURES_DATA_PATH,
    XGBOOST_MODEL_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import baseline_split

# --------------------------------------------------
# 1. Load the engineered dataset
# --------------------------------------------------
print("Loading PaySim dataset...")
df = pd.read_csv(FEATURES_DATA_PATH)

print("Dataset shape:", df.shape)
print("Fraud distribution:")
print(df["isFraud"].value_counts())

# --------------------------------------------------
# 2. Load the saved XGBoost model
# --------------------------------------------------
saved = joblib.load(XGBOOST_MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

print("\nModel loaded successfully.")
print("Features used:", feature_columns)

# Check that all required features are present
missing = [col for col in feature_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

# --------------------------------------------------
# 3. Recreate the original reproducible data split
# --------------------------------------------------
# The training script selected 10% of the dataset
# and used an 80/20 train-test split
X_train, X_test, y_train, y_test = baseline_split(df, feature_columns)

print("\nTest rows:", len(X_test))
print("Actual fraud cases:", int(y_test.sum()))

# --------------------------------------------------
# 4. Predict on the test set
# --------------------------------------------------
print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.5
predictions = (probabilities >= THRESHOLD).astype(int)

# --------------------------------------------------
# 5. Calculate evaluation metrics
# --------------------------------------------------
tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
average_precision = average_precision_score(
    y_test, probabilities
)

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) > 0 else 0
)

# --------------------------------------------------
# 6. Display the results
# --------------------------------------------------
print("\n" + "=" * 55)
print("PAYSim FRAUD DETECTION - MODEL EVALUATION")
print("=" * 55)

print("\nCONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print("                 Predicted Legit   Predicted Fraud")
print(f"Actual Legit      {tn:12d}   {fp:14d}")
print(f"Actual Fraud      {fn:12d}   {tp:14d}")

print("\nEVALUATION METRICS")
print(f"Precision:           {precision:.6f}")
print(f"Recall:              {recall:.6f}")
print(f"F1-score:            {f1:.6f}")
print(f"Average Precision:   {average_precision:.6f}")
print(f"False-positive rate:  {false_positive_rate:.6f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

# --------------------------------------------------
# 7. Save results to a text file
# --------------------------------------------------
results = f"""
PAYSim FRAUD DETECTION MODEL EVALUATION

Test rows: {len(X_test)}
Actual fraud cases: {int(y_test.sum())}

Confusion Matrix:
TN = {tn}
FP = {fp}
FN = {fn}
TP = {tp}

Precision: {precision:.6f}
Recall: {recall:.6f}
F1-score: {f1:.6f}
Average Precision: {average_precision:.6f}
False-positive rate: {false_positive_rate:.6f}
"""

ensure_output_dirs()

with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write(results)

print("\nResults saved to:", EVALUATION_RESULTS_PATH)
print("Evaluation completed.")
