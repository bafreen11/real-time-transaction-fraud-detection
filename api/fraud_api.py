import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.config import XGBOOST_VALIDATED_MODEL_PATH
from fraud_detection.features.engineering import prepare_transaction


# --------------------------------------------------
# 1. Load the validated trained model
# --------------------------------------------------

MODEL_PATH = XGBOOST_VALIDATED_MODEL_PATH

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_columns = saved["feature_columns"]
THRESHOLD = float(saved["threshold"])


# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PaySim Fraud Detection API",
    description=(
        "Predicts fraud probability for a transaction "
        "using a validated XGBoost model."
    ),
    version="1.1"
)


# --------------------------------------------------
# 3. Define the incoming transaction format
# --------------------------------------------------

class Transaction(BaseModel):
    step: int = Field(ge=1)
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


# --------------------------------------------------
# 4. Create the required engineered features
# --------------------------------------------------

def prepare_features(transaction: Transaction):

    # Same feature code as training, in the exact training order
    return prepare_transaction(
        transaction.model_dump(), feature_columns
    )


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "PaySim Fraud Detection API is running",
        "model": "XGBoost",
        "threshold": THRESHOLD
    }


# --------------------------------------------------
# 6. Predict fraud for a transaction
# --------------------------------------------------


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        X = prepare_features(transaction)

        fraud_probability = float(
            model.predict_proba(X)[0, 1]
        )

        predicted_label = int(
            fraud_probability >= THRESHOLD
        )

        return {
            "fraud_probability": round(fraud_probability, 6),
            "predicted_label": predicted_label,
            "decision": (
                "FRAUD_ALERT"
                if predicted_label == 1
                else "LEGITIMATE"
            ),
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
