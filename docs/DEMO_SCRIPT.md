# Demo recording script (about 5 minutes)

Record the screen with **Xbox Game Bar** (`Win + Alt + R` starts and stops recording, saved to `Videos\Captures`) or OBS Studio. Use a 1080p screen, close notifications, and make the terminal font large (`Ctrl + +`).

## Before recording (about 10 minutes, not recorded)

```powershell
cd "E:\Transactions Fraud Detection"
.venv\Scripts\Activate.ps1
python -m pytest -q                         # expect: 32 passed
docker compose up -d --wait kafka           # Kafka (if Docker Desktop is slow, see Plan B)
```

**Reset the replay** so the counts start from zero and recipient history isn't counted twice. In VS Code, run *Terminal → Run Task → setup: reset replay*, or:

```powershell
Remove-Item -Force -ErrorAction SilentlyContinue data/serving/fraud_cases.db*
docker exec fraud-kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --group paysim-fraud-service --topic transactions --reset-offsets --to-latest --execute
python scripts/feast/build_feature_store.py   # rebuilds the online store up to step 600 (several minutes)
```

Open four terminals and a browser with these tabs ready (they load once the servers start):
`http://127.0.0.1:8000/docs`, `http://127.0.0.1:8001/docs`, `http://localhost:8501`, `http://127.0.0.1:5000`.

## Recording

| Time | Show | Say |
|---|---|---|
| 0:00 | README on GitHub or in VS Code, the architecture diagram in `docs/ARCHITECTURE.md` | "Fraud detection on 6.3 million PaySim transactions. Fraud is 0.13%. Three tracks: an XGBoost baseline, a graph neural network, and a live streaming platform." |
| 0:30 | Terminal 1: `uvicorn api.fraud_api:app --port 8000`, then `/docs` → `POST /predict` → *Try it out* with the fraud example below | "This is the baseline API. A transfer that empties an account gets 0.9997 fraud probability, above the 0.3 threshold, so it's a FRAUD_ALERT." Then change only `type` to `PAYMENT`: "The same amounts as a payment score 0.009. Fraud in PaySim only happens in transfers and cash-outs, and the model uses the transaction type." Then the normal payment: "A normal payment: 0.00004, legitimate." |
| 1:15 | Terminal 2: `uvicorn api.fraud_service:app --port 8001`; open `http://127.0.0.1:8001/` | "The streaming service loads the stream model, 19 features including recipient history from the Feast feature store, and connects to Kafka. The consumer says running." |
| 1:45 | Terminal 3: `streamlit run dashboard/app.py`; dashboard **Alerts** tab (empty) | "This is the analyst dashboard. It only talks to the service API." |
| 2:00 | Terminal 4: `python streaming/kafka_replay_producer.py`; switch to the dashboard | "Now I replay real transactions from hour 600 through Kafka. Watch the counters and the alert toasts." Wait until roughly 500 to 1,000 are scored. |
| 2:45 | Dashboard **Alerts** tab: alert precision, latest alerts | "Every alert so far is real fraud. Alert precision is shown live, because in a replay we know the true labels." |
| 3:15 | **Case review** tab: open a case, show the features, set *CONFIRMED_FRAUD* + label *FRAUD* + a note → Save | "The analyst sees exactly what the model saw, for example how many times this recipient received money before. The label goes into the next retraining." |
| 3:45 | **Drift monitor** tab, then *Open the full Evidently report* | "Drift monitoring compares live features with the training data. When half the features drift, it retrains automatically. That happened yesterday: 68% drift, a new model was trained with analyst labels and promoted because it was at least as good." |
| 4:20 | MLflow `http://127.0.0.1:5000` → experiment *PaySim_Stream_Model* → the two runs, `trigger` tag | "Every training run is tracked: parameters, metrics, and what triggered it." |
| 4:45 | Terminal: `python -m pytest -q` (32 passed) | "32 automated tests cover the features, both APIs including error handling, the case workflow and model quality." |
| 5:00 | stop | |

Fraud example (paysim.csv row 3):

```json
{"step": 1, "type": "TRANSFER", "amount": 181, "oldbalanceOrg": 181, "newbalanceOrig": 0, "oldbalanceDest": 0, "newbalanceDest": 0}
```

Normal payment (row 1):

```json
{"step": 1, "type": "PAYMENT", "amount": 9839.64, "oldbalanceOrg": 170136, "newbalanceOrig": 160296.36, "oldbalanceDest": 0, "newbalanceDest": 0}
```

Shortcut: in VS Code, *Run and Debug* → **Run streaming platform (Feast + dashboard)** → F5 starts Kafka, the service, the dashboard, MLflow, the drift monitor and the replay producer together. To narrate each part, start them one by one as above.

## Plan B: Docker or Kafka won't start

The service works without Kafka. Score transactions by hand instead of replaying:

```powershell
$env:ENABLE_KAFKA_CONSUMER="0"; uvicorn api.fraud_service:app --port 8001
streamlit run dashboard/app.py
```

In `http://127.0.0.1:8001/docs` → `POST /score`, set `update_state` to `false` (so the Feast store stays clean for a later replay), then send this fraud CASH_OUT (row 4), then the payment. The alert appears in the dashboard, and you can review it as a case:

```json
{"step": 1, "type": "CASH_OUT", "amount": 181, "nameOrig": "C840083671", "oldbalanceOrg": 181,
 "newbalanceOrig": 0, "nameDest": "C38997010", "oldbalanceDest": 21182, "newbalanceDest": 0}
```

## After recording

Stop everything with `Ctrl + C` in each terminal and `docker compose down`. Before another replay, reset again (the first section above).
