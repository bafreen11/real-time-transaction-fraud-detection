import joblib
import pandas as pd

from fraud_detection.config import (
    FEATURES_DATA_PATH as DATA_PATH,
    XGBOOST_MODEL_PATH as MODEL_PATH,
)
from fraud_detection.features.engineering import TRANSACTION_FIELDS

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_columns = saved["feature_columns"]

print("Model classes:", model.classes_)
print("Expected features:", feature_columns)

fraud_rows = []
legitimate_rows = []

# Read the dataset in manageable chunks
for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    # Confirm that required features are available
    missing = [
        c for c in feature_columns
        if c not in chunk.columns
    ]

    if missing:
        raise ValueError(
            f"Training features missing from dataset: {missing}"
        )

    # Collect real fraud and legitimate examples
    fraud = chunk[chunk["isFraud"] == 1]
    legitimate = chunk[chunk["isFraud"] == 0]

    if len(fraud_rows) < 5 and not fraud.empty:
        fraud_rows.append(
            fraud.head(5 - len(fraud_rows))
        )

    if len(legitimate_rows) < 5 and not legitimate.empty:
        legitimate_rows.append(
            legitimate.head(5 - len(legitimate_rows))
        )

    if len(fraud_rows) >= 1 and len(legitimate_rows) >= 1:
        if (
            sum(len(x) for x in fraud_rows) >= 5
            and sum(len(x) for x in legitimate_rows) >= 5
        ):
            break

fraud_df = pd.concat(fraud_rows, ignore_index=True)
legit_df = pd.concat(legitimate_rows, ignore_index=True)

# Build model inputs in the saved training order
fraud_X = fraud_df[feature_columns]
legit_X = legit_df[feature_columns]

# Locate the fraud class rather than assuming its column index
fraud_class_index = list(model.classes_).index(1)

fraud_probs = model.predict_proba(fraud_X)[:, fraud_class_index]
legit_probs = model.predict_proba(legit_X)[:, fraud_class_index]

print("\nACTUAL FRAUD TRANSACTIONS")
for i, probability in enumerate(fraud_probs):
    print(
        f"Fraud case {i + 1}: "
        f"probability={probability:.6f}, "
        f"predicted={'FRAUD' if probability >= 0.5 else 'LEGITIMATE'}"
    )

print("\nACTUAL LEGITIMATE TRANSACTIONS")
for i, probability in enumerate(legit_probs):
    print(
        f"Legitimate case {i + 1}: "
        f"probability={probability:.6f}, "
        f"predicted={'FRAUD' if probability >= 0.5 else 'LEGITIMATE'}"
    )

print("\nFraud cases detected:", int((fraud_probs >= 0.5).sum()), "/", len(fraud_probs))
print("Legitimate cases correctly classified:", int((legit_probs < 0.5).sum()), "/", len(legit_probs))
# Get one real fraud transaction
row = fraud_df.iloc[0]

raw_columns = TRANSACTION_FIELDS

print("\nCOPY THESE VALUES INTO FASTAPI /docs:")
payload = {
    col: (
        int(row[col]) if col == "step"
        else str(row[col]) if col == "type"
        else float(row[col])
    )
    for col in raw_columns
}

print(payload)

# Calculate the expected prediction directly from the saved model
expected_probability = float(
    model.predict_proba(
        pd.DataFrame([row[feature_columns]])[feature_columns]
    )[0, list(model.classes_).index(1)]
)

print("Expected fraud probability:", expected_probability)