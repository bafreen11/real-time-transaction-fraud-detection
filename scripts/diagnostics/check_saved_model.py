import joblib
import json

from fraud_detection.config import FEATURE_COLUMNS_PATH, XGBOOST_MODEL_PATH

# Load the saved model artifact
artifact = joblib.load(XGBOOST_MODEL_PATH)

print("Artifact type:", type(artifact))

if isinstance(artifact, dict):
    print("Artifact keys:", artifact.keys())

    if "model" in artifact:
        print("Model type:", type(artifact["model"]))

    if "feature_columns" in artifact:
        print("Feature columns:", artifact["feature_columns"])

# Read the feature definitions file
with open(FEATURE_COLUMNS_PATH, "r") as file:
    features = json.load(file)

print("\nContents of feature_columns.json:")
print(features)
