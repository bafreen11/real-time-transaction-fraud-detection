import joblib

from fraud_detection.config import XGBOOST_VALIDATED_MODEL_PATH
from fraud_detection.features.engineering import prepare_transaction

saved = joblib.load(XGBOOST_VALIDATED_MODEL_PATH)
model = saved["model"]
features = saved["feature_columns"]
threshold = saved["threshold"]

# The exact transaction you tested in Swagger
data = {
    "step": 1,
    "amount": 181,
    "oldbalanceOrg": 181,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 181,
}

# Recreate the API's engineered features
X = prepare_transaction(data, features)

probability = float(model.predict_proba(X)[0, 1])

print("Features sent to model:")
print(X.to_string(index=False))
print("\nFraud probability:", round(probability, 6))
print("Threshold:", threshold)
print("Prediction:", "FRAUD_ALERT" if probability >= threshold else "LEGITIMATE")
