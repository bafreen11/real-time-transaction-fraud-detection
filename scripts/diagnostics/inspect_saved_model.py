import joblib

from fraud_detection.config import XGBOOST_MODEL_PATH as MODEL_PATH

artifact = joblib.load(MODEL_PATH)

print("Loaded object type:", type(artifact))
print("Dictionary keys:", list(artifact.keys()) if isinstance(artifact, dict) else "Not a dictionary")

if isinstance(artifact, dict):
    for key, value in artifact.items():
        print(f"\nKey: {key}")
        print("Value type:", type(value))
        print("Has predict_proba:", hasattr(value, "predict_proba"))
