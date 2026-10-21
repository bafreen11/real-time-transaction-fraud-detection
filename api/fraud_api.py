import logging
import uuid

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.config import XGBOOST_VALIDATED_MODEL_PATH
from fraud_detection.features.engineering import prepare_transaction

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud_api")

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


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    # The default handler echoes the rejected input back, which fails
    # for NaN / Infinity and is not needed by the caller.
    errors = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


# --------------------------------------------------
# 3. Define the incoming transaction format
# --------------------------------------------------

class Transaction(BaseModel):
    # Invalid input (missing fields, negative or non-finite values) is
    # rejected by FastAPI with a 422 before the model is called.
    model_config = ConfigDict(allow_inf_nan=False)

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

    except Exception:
        # Full details stay in the server log; the caller only gets an
        # error id to quote, never the exception text.
        error_id = uuid.uuid4().hex[:12]
        log.exception("Prediction failed (error id %s)", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Error id: {error_id}"
        )
