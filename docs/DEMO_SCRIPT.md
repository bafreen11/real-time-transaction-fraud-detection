# Demo recording scripts

Two versions of the same project.

- **Option A — deployed demo and results (about 7 minutes).** Runs entirely in a browser against
  the live deployment. No Docker, no local servers, very little that can break while recording.
  This is the one to record for a portfolio, a submission or a CV link.
- **Option B — full streaming platform (about 5 minutes).** Kafka replay, the analyst dashboard,
  drift-triggered retraining and MLflow, all running locally. More impressive, but it needs Docker
  and about 30 minutes of preparation.

Record with **Xbox Game Bar** (`Win + Alt + R` starts and stops, saved to `Videos\Captures`) or OBS
Studio. Use a 1080p screen, close notifications, and hide the bookmarks bar (`Ctrl + Shift + B`).

---

# Option A - deployed demo and results (about 7 minutes)

## Before recording (about 5 minutes, not recorded)

1. **Wake the API.** It sleeps after 15 idle minutes on the free plan and takes about a minute to
   come back. Open <https://fraud-api-bafreen11.onrender.com/> and wait for the JSON.
2. **Open these tabs, in this order:**
   - <https://github.com/bafreen11/real-time-transaction-fraud-detection>
   - <https://fraud-detection-bafreen11.netlify.app>
   - <https://fraud-api-bafreen11.onrender.com/docs>
   - `outputs/figures/confusion_matrix.png` and `outputs/figures/pr_curve.png` in VS Code
3. Zoom the browser to 125% (`Ctrl + +`).

## Recording

The heart of this demo is the **ablation in the middle**: change one field at a time and show the
score collapse. It proves the model discriminates rather than flagging every transfer.

| Time | Show | Say |
|---|---|---|
| 0:00 | GitHub README, top of the page | "Fraud detection on PaySim, a synthetic mobile-money dataset. Only 0.13% of transactions are fraud, so accuracy is the wrong metric - flagging nothing scores 99.87%. I report average precision instead." |
| 0:40 | README architecture section | "An XGBoost classifier scores each transaction through a FastAPI service. There is also a graph neural network track and a Kafka streaming platform with a feature store, which I will come back to." |
| 1:20 | Demo page, press **Load a fraud example**, then Check for fraud | "A transfer that moves the entire balance out of an account and leaves it at zero. The model gives 99.97% - a fraud alert. The threshold is 0.3, chosen on validation data for the best F1." |
| 2:10 | Change **only** the sender balance: before 10000, after 9819. Check again | "Same transfer, same amount, but now the sender keeps most of their money. The probability drops to 0.00005 - about twenty thousand times lower. So it is not flagging transfers; it is flagging accounts being emptied." |
| 3:00 | Undo that, then change **only** Type to **PAYMENT**. Check again | "Back to the drained account, but as a payment instead of a transfer: 0.009, legitimate. In PaySim, fraud only happens through transfers and cash-outs, and the model learned that." |
| 3:40 | Press **Load a normal payment**, Check | "An ordinary payment: 0.00004. Four orders of magnitude from the fraud case." |
| 4:10 | API tab, `/docs`, `POST /predict` → Try it out, send a negative amount | "The same API, documented. Invalid input is rejected with a 422 that names the field, before the model ever runs." |
| 4:45 | README results table | "On a held-out test set of 127,253 transactions containing 164 frauds: average precision 0.9916, recall 99.4% at 93.7% precision. Eleven false alarms and one missed fraud." |
| 5:30 | `outputs/figures/confusion_matrix.png` and `pr_curve.png` | "The confusion matrix and the precision-recall curve for exactly that model and that test set." |
| 6:00 | README, the rigor paragraph above the table | "Every threshold and blend weight was chosen on a validation set, and the test set was scored once at the end. That matters on imbalanced data, where tuning on the test set inflates everything." |
| 6:30 | README, the GNN row of the results table | "I also built a graph neural network over the account-to-account graph. It did not beat XGBoost alone, so the selected blend weight is zero. I kept the result in the README rather than dropping it." |
| 7:00 | `docs/ARCHITECTURE.md` | "The full platform - Kafka, a Feast feature store, drift monitoring and automatic retraining - runs locally and is documented here. Links to the live demo are at the top of the README. Thanks for watching." |

## If the API is asleep on camera

The page shows "Waking up the free server" after three seconds. Say: *"This is on a free tier that
sleeps when idle."* Keep talking through the architecture while it loads.

---

# Option B - full streaming platform (about 5 minutes)

Needs Docker. Make the terminal font large (`Ctrl + +`).

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
