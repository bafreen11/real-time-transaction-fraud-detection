import joblib
import pandas as pd

MODEL_PATH = r"E:\Transactions Fraud Detection\fraud_xgboost_model.joblib"

saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]

print("Model loaded successfully")
print("\nExpected features:")
print(feature_columns)

# Check fraud probabilities on the saved model
print("\nTesting prediction probabilities...")

X_sample = pd.DataFrame(
    model.get_booster().feature_names and
    [[0] * len(feature_columns)] or
    [[0] * len(feature_columns)],
    columns=feature_columns
)

probability = model.predict_proba(X_sample)[0, 1]

print("Fraud probability:", probability)
print("Predicted class:", model.predict(X_sample)[0])