"""Error handling of the baseline API: invalid input gets a 422, and an
unexpected failure gets a generic 500 that hides the exception text."""

import logging

from fastapi.testclient import TestClient

import api.fraud_api as fraud_api

client = TestClient(fraud_api.app)

VALID = {
    "step": 1, "amount": 181.0, "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


def post_raw(body):
    # json.dumps writes NaN / Infinity, which plain JSON cannot carry.
    return client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )


def test_non_finite_amount_is_a_validation_error():
    body = '{"step": 1, "amount": NaN, "oldbalanceOrg": 181, ' \
           '"newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0}'
    assert post_raw(body).status_code == 422

    body = body.replace("NaN", "Infinity")
    assert post_raw(body).status_code == 422


def test_unexpected_error_is_logged_but_not_returned(monkeypatch, caplog):
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("internal detail: /secret/path")

    monkeypatch.setattr(fraud_api, "model", BrokenModel())

    with caplog.at_level(logging.ERROR, logger="fraud_api"):
        response = client.post("/predict", json=VALID)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "secret" not in detail
    assert detail.startswith("Prediction failed. Error id: ")

    error_id = detail.rsplit(" ", 1)[-1]
    assert "internal detail: /secret/path" in caplog.text
    assert error_id in caplog.text
