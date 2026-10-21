"""Streaming fraud scoring service (Feast + XGBoost/GNN blend).

Consumes transactions from Kafka, fetches recipient features from the
Feast online store, scores with the blended stream model, writes the
updated recipient state back to Feast, stores the result as a case and
publishes alerts to the fraud_alerts topic. The same scoring path is
exposed over HTTP for the dashboard and for ad-hoc requests.

Run from the project root:
    uvicorn api.fraud_service:app --port 8001
Set ENABLE_KAFKA_CONSUMER=0 to run without Kafka.
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fraud_detection.config import (
    DRIFT_LATEST_REPORT_PATH,
    DRIFT_LATEST_SUMMARY_PATH,
    GNN_SERVING_ARTIFACT_PATH,
    KAFKA_ALERTS_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_SERVICE_GROUP_ID,
    KAFKA_TRANSACTIONS_TOPIC,
    PAYSIM_START_TIME,
    SERVING_DB_PATH,
    STREAM_XGBOOST_MODEL_PATH,
)
from fraud_detection.feature_store import (
    get_online_state,
    get_store,
    write_online_state,
)
from fraud_detection.features.behavioral import (
    is_tracked_recipient,
    update_recipient_state,
)
from fraud_detection.features.stream import (
    STREAM_TRANSACTION_FIELDS,
    build_stream_features,
)
from fraud_detection.serving.case_store import (
    CASE_STATUSES,
    FEEDBACK_LABELS,
    CaseStore,
)
from fraud_detection.serving.scorer import StreamScorer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud_service")

ENABLE_KAFKA_CONSUMER = os.environ.get("ENABLE_KAFKA_CONSUMER", "1") == "1"
# Apply the Feast definitions on start-up (a fresh registry in a container).
FEAST_APPLY_ON_START = os.environ.get("FEAST_APPLY_ON_START", "0") == "1"


# --------------------------------------------------
# 1. Shared state
# --------------------------------------------------

class Service:
    def __init__(self):
        self.scorer = StreamScorer(
            STREAM_XGBOOST_MODEL_PATH, GNN_SERVING_ARTIFACT_PATH
        )
        self.store = get_store(apply_definitions=FEAST_APPLY_ON_START)
        self.cases = CaseStore(SERVING_DB_PATH)
        # Read-score-update of recipient state must not interleave.
        self.lock = threading.Lock()
        self.alert_producer = None
        self.consumer_status = "disabled"
        self.stop_event = threading.Event()

    def score_transaction(self, transaction, update_state=True):
        """Score one transaction dict and record it. Returns the result."""
        txn = pd.DataFrame([transaction])[STREAM_TRANSACTION_FIELDS]
        recipient = str(transaction["nameDest"])
        tracked = bool(is_tracked_recipient([recipient])[0])

        with self.lock:
            if tracked:
                state = get_online_state(self.store, [recipient])
            else:
                state = pd.DataFrame([{}])
                state = state.reindex(
                    columns=["txn_count", "amount_sum", "last_step",
                             "day_txn_count"]
                )

            features = build_stream_features(txn, state)
            result = self.scorer.score(txn, features)[0]

            if tracked and update_state:
                previous = {
                    k: (None if pd.isna(v) else v)
                    for k, v in state.iloc[0].to_dict().items()
                }
                new_state = update_recipient_state(
                    previous, transaction["step"], transaction["amount"]
                )
                write_online_state(self.store, [{
                    "account_id": recipient,
                    "event_timestamp": pd.Timestamp(PAYSIM_START_TIME)
                    + pd.Timedelta(hours=int(transaction["step"]) - 1),
                    **new_state,
                }])

        feature_record = {
            k: float(v) for k, v in features.iloc[0].to_dict().items()
        }
        record_id = self.cases.add_scored(
            transaction,
            feature_record,
            result,
            actual_is_fraud=transaction.get("actual_is_fraud"),
        )

        result = {"id": record_id, **result, "features": feature_record}

        if result["decision"] == "FRAUD_ALERT" and self.alert_producer:
            self.alert_producer.send(KAFKA_ALERTS_TOPIC, value={
                "case_id": record_id,
                "transaction_id": transaction.get("transaction_id"),
                "fraud_score": result["fraud_score"],
                "amount": transaction["amount"],
                "type": transaction["type"],
                "nameOrig": transaction["nameOrig"],
                "nameDest": transaction["nameDest"],
                "step": transaction["step"],
            })

        return result


service: Optional[Service] = None


# --------------------------------------------------
# 2. Kafka consumer thread
# --------------------------------------------------

def consume_transactions():
    from kafka import KafkaConsumer, KafkaProducer

    try:
        consumer = KafkaConsumer(
            KAFKA_TRANSACTIONS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=KAFKA_SERVICE_GROUP_ID,
            value_deserializer=lambda data: json.loads(data.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        service.alert_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda data: json.dumps(data).encode("utf-8"),
        )
    except Exception as exc:
        service.consumer_status = f"error: {exc}"
        log.error("Kafka consumer could not start: %s", exc)
        return

    service.consumer_status = "running"
    log.info("Consuming %s from %s",
             KAFKA_TRANSACTIONS_TOPIC, KAFKA_BOOTSTRAP_SERVERS)

    while not service.stop_event.is_set():
        for message in consumer:
            event = message.value
            transaction = event.get("transaction") if isinstance(event, dict) else None

            if not transaction or any(
                f not in transaction for f in STREAM_TRANSACTION_FIELDS
            ):
                log.warning("Skipping message without transaction fields")
                continue

            transaction = {
                **transaction,
                "transaction_id": event.get("transaction_id"),
                "actual_is_fraud": event.get("actual_is_fraud"),
            }
            try:
                result = service.score_transaction(transaction)
            except Exception:
                log.exception("Scoring failed for %s", event.get("transaction_id"))
                continue

            if result["decision"] == "FRAUD_ALERT":
                log.info("FRAUD_ALERT case %s | txn %s | score %.4f",
                         result["id"], transaction["transaction_id"],
                         result["fraud_score"])

            if service.stop_event.is_set():
                break

    consumer.close()
    service.alert_producer.close()
    service.consumer_status = "stopped"


# --------------------------------------------------
# 3. Application
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    global service
    service = Service()

    thread = None
    if ENABLE_KAFKA_CONSUMER:
        service.consumer_status = "starting"
        thread = threading.Thread(target=consume_transactions, daemon=True)
        thread.start()

    yield

    service.stop_event.set()
    if thread:
        thread.join(timeout=5)


app = FastAPI(
    title="PaySim Streaming Fraud Service",
    description=(
        "Scores streamed transactions with Feast features and a blended "
        "XGBoost + GNN model, and manages alert cases."
    ),
    version="2.0",
    lifespan=lifespan,
)


class StreamTransaction(BaseModel):
    step: int = Field(ge=1)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    nameDest: str
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    transaction_id: Optional[int] = None
    actual_is_fraud: Optional[int] = None


class CaseUpdate(BaseModel):
    status: Optional[str] = None
    label: Optional[str] = None
    note: Optional[str] = None


@app.get("/")
def health():
    return {
        "message": "PaySim Streaming Fraud Service is running",
        "model_version": service.scorer.version,
        "threshold": service.scorer.threshold,
        "gnn_weight": service.scorer.gnn_weight,
        "kafka_consumer": service.consumer_status,
    }


@app.post("/score")
def score(transaction: StreamTransaction, update_state: bool = True):
    """Score a transaction. update_state=false leaves Feast unchanged."""
    try:
        return service.score_transaction(
            transaction.model_dump(), update_state=update_state
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}")


@app.get("/stats")
def stats():
    return service.cases.stats()


@app.get("/transactions")
def transactions(limit: int = 100, alerts_only: bool = False,
                 status: Optional[str] = None):
    return service.cases.list_transactions(limit, alerts_only, status)


@app.get("/cases/{case_id}")
def get_case(case_id: int):
    case = service.cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.patch("/cases/{case_id}")
def update_case(case_id: int, update: CaseUpdate):
    if not service.cases.get(case_id):
        raise HTTPException(status_code=404, detail="Case not found")
    try:
        return service.cases.update_case(
            case_id, update.status, update.label, update.note
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/options")
def options():
    return {
        "case_statuses": CASE_STATUSES,
        "feedback_labels": list(FEEDBACK_LABELS),
    }


@app.get("/drift/latest")
def drift_latest():
    if not DRIFT_LATEST_SUMMARY_PATH.exists():
        return {"status": "no drift check has run yet"}
    return json.loads(DRIFT_LATEST_SUMMARY_PATH.read_text())


@app.get("/drift/report")
def drift_report():
    if not DRIFT_LATEST_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="No drift report yet")
    return FileResponse(DRIFT_LATEST_REPORT_PATH, media_type="text/html")


@app.post("/model/reload")
def reload_model():
    """Load the current model file (after a retrain promoted a new one)."""
    with service.lock:
        service.scorer.reload()
    return {"model_version": service.scorer.version}
