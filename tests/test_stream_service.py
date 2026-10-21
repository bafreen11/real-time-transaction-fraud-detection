"""Streaming fraud service (api/fraud_service.py, served on port 8001).

Runs with the Kafka consumer off and a temporary case database. Scoring
uses update_state=false, so the Feast online store is only read.
"""

import pytest
from fastapi.testclient import TestClient

from fraud_detection.config import FEAST_DATA_DIR

pytestmark = pytest.mark.skipif(
    not (FEAST_DATA_DIR / "registry.db").exists(),
    reason="Feast store not built (run scripts/feast/build_feature_store.py)",
)

# Rows 4 and 1 of paysim.csv.
FRAUD_CASH_OUT = {
    "step": 1, "type": "CASH_OUT", "amount": 181.0,
    "nameOrig": "C840083671", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "nameDest": "C38997010", "oldbalanceDest": 21182.0, "newbalanceDest": 0.0,
}
LEGITIMATE_PAYMENT = {
    "step": 1, "type": "PAYMENT", "amount": 9839.64,
    "nameOrig": "C1231006815", "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36, "nameDest": "M1979787155",
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import api.fraud_service as service_module

    service_module.ENABLE_KAFKA_CONSUMER = False
    service_module.SERVING_DB_PATH = tmp_path_factory.mktemp("serving") / "cases.db"
    with TestClient(service_module.app) as client:
        yield client


def score(client, transaction):
    response = client.post("/score?update_state=false", json=transaction)
    assert response.status_code == 200
    return response.json()


def test_health(client):
    health = client.get("/").json()
    assert health["kafka_consumer"] == "disabled"
    assert 0 < health["threshold"] < 1


def test_fraud_is_alerted_with_blended_score(client):
    result = score(client, FRAUD_CASH_OUT)

    assert result["decision"] == "FRAUD_ALERT"
    expected = (
        (1 - result["gnn_weight"]) * result["xgboost_probability"]
        + result["gnn_weight"] * result["gnn_probability"]
    )
    assert result["fraud_score"] == pytest.approx(expected, abs=1e-5)
    assert len(result["features"]) == 19


def test_payment_is_legitimate(client):
    assert score(client, LEGITIMATE_PAYMENT)["decision"] == "LEGITIMATE"


def test_alert_becomes_case_analyst_can_update(client):
    case_id = score(client, FRAUD_CASH_OUT)["id"]
    assert client.get(f"/cases/{case_id}").json()["case_status"] == "OPEN"

    updated = client.patch(
        f"/cases/{case_id}", json={"status": "CONFIRMED_FRAUD", "label": "FRAUD"}
    ).json()
    assert updated["case_status"] == "CONFIRMED_FRAUD"
    assert updated["analyst_label"] == "FRAUD"

    alerts = client.get("/transactions?alerts_only=true").json()
    assert case_id in [row["id"] for row in alerts]


def test_bad_requests(client):
    assert client.get("/cases/999999").status_code == 404
    assert client.patch("/cases/1", json={"status": "DONE"}).status_code == 422
    assert client.post(
        "/score", json={**FRAUD_CASH_OUT, "type": "BITCOIN"}
    ).status_code == 422
