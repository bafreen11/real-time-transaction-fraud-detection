# API documentation

Two FastAPI services. Both publish interactive docs at `/docs` (Swagger UI) and a schema at `/openapi.json`.

| Service | File | Start command | Model |
|---|---|---|---|
| Baseline API | `api/fraud_api.py` | `uvicorn api.fraud_api:app --port 8000` | `models/fraud_xgboost_validated_model.joblib` (14 features, threshold 0.3) |
| Streaming fraud service | `api/fraud_service.py` | `uvicorn api.fraud_service:app --port 8001` | `models/stream_xgboost_model.joblib` + GNN artifact (19 features) |

Run both from the project root, after `pip install -e .`.

---

## 1. Baseline API (port 8000)

### `GET /`: health

```json
{"message": "PaySim Fraud Detection API is running", "model": "XGBoost", "threshold": 0.3}
```

### `POST /predict`: score one transaction

Request body (all fields required):

| Field | Type | Rule |
|---|---|---|
| `step` | int | ≥ 1 |
| `type` | string | one of `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN` |
| `amount` | float | ≥ 0 |
| `oldbalanceOrg` | float | ≥ 0 |
| `newbalanceOrig` | float | ≥ 0 |
| `oldbalanceDest` | float | ≥ 0 |
| `newbalanceDest` | float | ≥ 0 |

```json
{"step": 1, "type": "TRANSFER", "amount": 181, "oldbalanceOrg": 181,
 "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0}
```

Response:

```json
{"fraud_probability": 0.99972, "predicted_label": 1,
 "decision": "FRAUD_ALERT", "threshold": 0.3}
```

| Status | When |
|---|---|
| 200 | scored |
| 422 | missing field, wrong type, unknown transaction `type`, negative, NaN or infinite value, or `step` < 1. The body lists each error's `loc`, `msg` and `type`, without echoing the input |
| 500 | unexpected server error. The body is only `{"detail": "Prediction failed. Error id: <id>"}`; the traceback is in the server log under the same id |

PowerShell example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" `
  -Body '{"step":1,"type":"TRANSFER","amount":181,"oldbalanceOrg":181,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
```

---

## 2. Streaming fraud service (port 8001)

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `ENABLE_KAFKA_CONSUMER` | `1` | `0` = HTTP only, no Kafka |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka address (`kafka:19092` inside Docker) |
| `SERVING_DB_PATH` | `data/serving/fraud_cases.db` | case database |
| `FEAST_APPLY_ON_START` | `0` | `1` = register Feast definitions at start-up (fresh container) |

### `GET /`: health

```json
{"message": "PaySim Streaming Fraud Service is running",
 "model_version": "20261020154142", "threshold": 0.95, "gnn_weight": 0.0,
 "kafka_consumer": "running"}
```

`kafka_consumer` is one of `disabled`, `starting`, `running`, `stopped`, `error: ...`.

### `POST /score?update_state=true`: score one transaction

| Field | Type | Rule |
|---|---|---|
| `step` | int | ≥ 1 |
| `type` | string | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN` |
| `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | float | ≥ 0 |
| `nameOrig`, `nameDest` | string | account IDs |
| `transaction_id` | int | optional |
| `actual_is_fraud` | int | optional (known only in replays) |

`update_state=false` scores without writing the recipient's new state to Feast (useful for "what if" requests). The transaction is still saved in the case store.

```json
{"step": 1, "type": "CASH_OUT", "amount": 181, "nameOrig": "C840083671",
 "oldbalanceOrg": 181, "newbalanceOrig": 0, "nameDest": "C38997010",
 "oldbalanceDest": 21182, "newbalanceDest": 0}
```

Response:

```json
{"id": 2, "xgboost_probability": 0.999427, "gnn_probability": 0.234229,
 "fraud_score": 0.999427, "gnn_weight": 0.0, "threshold": 0.95,
 "predicted_label": 1, "decision": "FRAUD_ALERT",
 "model_version": "20261020154142",
 "features": {"step": 1.0, "amount": 181.0, "...": "all 19 features"}}
```

`fraud_score = (1 − gnn_weight) × xgboost_probability + gnn_weight × gnn_probability`. An alert opens a case with status `OPEN` and publishes to the `fraud_alerts` topic.

Errors work as in the baseline API: 422 for invalid input (including NaN or infinite numbers), listing `loc`, `msg` and `type` without echoing the input; 500 returns only `{"detail": "Scoring failed. Error id: <id>"}`, with the traceback in the server log.

### `GET /stats`

Example shape (values depend on the replay and on analyst activity):

```json
{"scored": 2000, "alerts": 111, "open_cases": 109, "labeled_fraud": 2,
 "labeled_legitimate": 0, "true_alerts": 110, "actual_fraud": 110, "latest_step": 601}
```

### `GET /transactions?limit=100&alerts_only=false&status=OPEN`

The newest scored transactions first. Each record holds the case-store columns plus `transaction` and `features` objects.

### `GET /cases/{id}` and `PATCH /cases/{id}`

```json
{"status": "CONFIRMED_FRAUD", "label": "FRAUD", "note": "Account emptied, then cashed out"}
```

All three fields are optional. Unknown status or label → 422; unknown id → 404.

| Status flow | `OPEN → INVESTIGATING → CONFIRMED_FRAUD / FALSE_POSITIVE → CLOSED` |
|---|---|
| Labels | `FRAUD`, `LEGITIMATE` (appended to training on the next retrain) |

### `GET /options`

The allowed statuses and labels (used by the dashboard).

### `GET /drift/latest` and `GET /drift/report`

The latest drift summary (JSON) and the full Evidently report (HTML, 404 until the first check).

### `POST /model/reload`

Loads `models/stream_xgboost_model.joblib` again without a restart (called by the drift monitor after a promotion).

```json
{"model_version": "20261020154142"}
```

---

## 3. Kafka messages

### Topic `transactions` (written by `streaming/kafka_replay_producer.py`)

```json
{
  "transaction_id": 4182938,
  "transaction": {"step": 600, "type": "TRANSFER", "amount": 1520.0,
                   "nameOrig": "C...", "oldbalanceOrg": 1520.0, "newbalanceOrig": 0.0,
                   "nameDest": "C...", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
  "features": {"step": 600, "amount": 1520.0, "...": "6 raw fields + type, for the baseline consumer"},
  "transaction_type": "TRANSFER", "nameOrig": "C...", "nameDest": "C...",
  "actual_is_fraud": 1
}
```

The fraud service reads `transaction`; `streaming/kafka_fraud_consumer.py` (baseline path) reads `features`.

Replay settings: `REPLAY_START_STEP` (600), `REPLAY_MAX_TRANSACTIONS` (2000, 0 = all), `REPLAY_DELAY_SECONDS` (0.05), `REPLAY_FRAUD_ONLY_TYPES` (0).

### Topic `fraud_alerts` (written by the fraud service)

```json
{"case_id": 57, "transaction_id": 4182938, "fraud_score": 0.9998,
 "amount": 1520.0, "type": "TRANSFER", "nameOrig": "C...", "nameDest": "C...", "step": 600}
```
