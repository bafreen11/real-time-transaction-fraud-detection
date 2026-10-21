"""Case store: scored transactions, alert cases and analyst feedback."""

import pytest

from fraud_detection.serving.case_store import CaseStore

TRANSACTION = {
    "transaction_id": 7, "step": 600, "type": "TRANSFER", "amount": 500.0,
    "nameOrig": "C1", "nameDest": "C2",
}
FEATURES = {"amount": 500.0, "step": 600.0}


def result(decision):
    return {
        "xgboost_probability": 0.99, "gnn_probability": 0.2,
        "fraud_score": 0.99, "threshold": 0.95, "decision": decision,
        "model_version": "test",
    }


@pytest.fixture
def store(tmp_path):
    return CaseStore(tmp_path / "cases.db")


def test_alert_opens_case_and_legitimate_does_not(store):
    alert_id = store.add_scored(TRANSACTION, FEATURES, result("FRAUD_ALERT"), 1)
    legit_id = store.add_scored(TRANSACTION, FEATURES, result("LEGITIMATE"), 0)

    assert store.get(alert_id)["case_status"] == "OPEN"
    assert store.get(legit_id)["case_status"] is None

    stats = store.stats()
    assert stats["scored"] == 2
    assert stats["alerts"] == 1
    assert stats["true_alerts"] == 1


def test_case_workflow_and_feedback_label(store):
    case_id = store.add_scored(TRANSACTION, FEATURES, result("FRAUD_ALERT"))
    store.update_case(case_id, status="INVESTIGATING")
    case = store.update_case(
        case_id, status="CONFIRMED_FRAUD", label="FRAUD", note="checked"
    )

    assert case["case_status"] == "CONFIRMED_FRAUD"
    assert case["analyst_note"] == "checked"

    # Labeled cases are what retraining appends to the training data.
    features, labels = store.labeled_features()
    assert features.to_dict("records") == [FEATURES]
    assert labels.tolist() == [1]


@pytest.mark.parametrize("bad_update", [
    {"status": "DONE"}, {"label": "MAYBE"},
])
def test_unknown_status_or_label_is_rejected(store, bad_update):
    case_id = store.add_scored(TRANSACTION, FEATURES, result("FRAUD_ALERT"))
    with pytest.raises(ValueError):
        store.update_case(case_id, **bad_update)
