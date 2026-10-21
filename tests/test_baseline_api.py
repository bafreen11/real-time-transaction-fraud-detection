"""Baseline scoring API (api/fraud_api.py, served on port 8000)."""

import pytest
from fastapi.testclient import TestClient

from api.fraud_api import THRESHOLD, app

client = TestClient(app)

# Rows 3 and 1 of paysim.csv: a fraudulent TRANSFER that empties the
# sender's account, and an ordinary PAYMENT.
FRAUD_TRANSFER = {
    "step": 1, "type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}
LEGITIMATE_PAYMENT = {
    "step": 1, "type": "PAYMENT", "amount": 9839.64, "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


def test_health_reports_threshold():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["threshold"] == THRESHOLD


def test_fraudulent_transfer_raises_alert():
    result = client.post("/predict", json=FRAUD_TRANSFER).json()
    assert result["decision"] == "FRAUD_ALERT"
    assert result["predicted_label"] == 1
    assert result["fraud_probability"] >= THRESHOLD


def test_ordinary_payment_is_legitimate():
    result = client.post("/predict", json=LEGITIMATE_PAYMENT).json()
    assert result["decision"] == "LEGITIMATE"
    assert result["predicted_label"] == 0
    assert 0 <= result["fraud_probability"] < THRESHOLD


@pytest.mark.parametrize("bad_input", [
    {**FRAUD_TRANSFER, "amount": -5},           # negative amount
    {**FRAUD_TRANSFER, "step": 0},              # steps start at 1
    {k: v for k, v in FRAUD_TRANSFER.items() if k != "amount"},  # missing
    {k: v for k, v in FRAUD_TRANSFER.items() if k != "type"},    # no type
    {**FRAUD_TRANSFER, "type": "BITCOIN"},      # not a PaySim type
])
def test_invalid_input_is_rejected(bad_input):
    assert client.post("/predict", json=bad_input).status_code == 422
