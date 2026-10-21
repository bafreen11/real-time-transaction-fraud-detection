# Step-by-step walkthrough

This guide takes you through the whole project from an empty terminal to a live fraud-scoring system. For every step it says **what happens**, **which files are involved**, **what goes in and what comes out**, and **how long it roughly takes**. Run every command from the project root (`E:\Transactions Fraud Detection`) in PowerShell.

> **The idea in one sentence.** We learn what fraud looks like from 6.3 million simulated mobile-money transactions, train models that give each transaction a fraud probability, and put those models behind an API that scores transactions as they stream in through Kafka, while an analyst reviews the alerts on a dashboard.

---

## 0. The big picture

The project has **three tracks**. Each one builds on the previous one:

| Track | Question it answers | Main output |
|---|---|---|
| **1. Baseline XGBoost** | Can a model tell fraud apart from the 6 balance numbers of a transaction? | `models/fraud_xgboost_validated_model.joblib`, served by the **baseline API** on port 8000 |
| **2. Shared split (XGBoost + GNN)** | Does knowing *who sends money to whom* (a graph of accounts) help? | A GraphSAGE GNN and a comparison of blended scores |
| **3. Streaming platform** | Can we score transactions live, with account history, and keep the model healthy? | The **fraud service** on port 8001, the analyst dashboard, and drift monitoring with automatic retraining |

```
paysim.csv (6.3M rows)
   │
   ├─ Track 1 ─► feature engineering ─► XGBoost ─► threshold on validation ─► Baseline API :8000
   │
   ├─ Track 2 ─► shared 80/20 split ─► XGBoost + account graph ─► GraphSAGE GNN ─► blend comparison
   │
   └─ Track 3 ─► Feast feature store ─► stream model ─► Kafka ─► Fraud service :8001 ─► Dashboard :8501
                                                                       ▲                     │
                                                drift monitor ─► retrain ◄── analyst labels ─┘
```

**You do not have to re-run everything to understand or demo it.** Every model and data file already exists on this machine. Section 6 lists which steps are safe to re-run and which ones overwrite things.

---

## 1. Setup (once)

### 1.1 The tools

| Tool | Why we need it |
|---|---|
| Python 3.11 or 3.12 (this machine: 3.12.1) | everything is Python |
| `.venv/` virtual environment | keeps this project's library versions separate |
| Docker Desktop | runs Kafka (the message stream) |
| The dataset `E:\paysim.csv` (471 MB) | from Kaggle: *PaySim1* |

### 1.2 Commands

```powershell
.venv\Scripts\Activate.ps1                 # turn on the project's Python
pip install -r requirements-dev.txt        # all libraries (XGBoost, torch, Feast, ...)
pip install -e .                           # makes `import fraud_detection` work everywhere
python -m pytest                           # 32 tests, about 15 seconds: proves the setup works
```

**What `pip install -e .` does:** it registers `src/fraud_detection/` as an installed package, so every script can write `from fraud_detection.config import ...`. The `-e` means "editable": if you change the code, the change is used at once.

### 1.3 How the code is organized

| Folder | What is in it | Rule |
|---|---|---|
| `src/fraud_detection/` | the **reusable library**: settings, feature code, models, serving helpers | every other part imports from here |
| `scripts/` | **pipeline steps**, run one by one, top to bottom | never imported |
| `api/` | the two web services (FastAPI) | |
| `streaming/` | Kafka producers and consumer | |
| `dashboard/` | the analyst web page (Streamlit) | |
| `tests/` | automated tests (pytest) | |
| `models/` | trained model files | |
| `data/processed/`, `outputs/` | generated data, predictions, reports, charts | not in git (too big) |

The single most important file is **`src/fraud_detection/config.py`**. Every path and every setting (random seed 42, sample sizes, thresholds to try, Kafka address, etc.) is defined there once. No script hard-codes a path.

---

## 2. Track 1: data and the baseline XGBoost model

### Step 1. Check the data: `scripts/data/run_data_checks.py`

```powershell
python scripts/data/run_data_checks.py
```

- **Uses:** `data/loading.py` (`load_data`, `inspect_data`, `validate_required_columns`, `check_data_quality`)
- **Reads:** `E:\paysim.csv`
- **Writes:** nothing; it only prints
- **What you see:** 6,362,620 rows and 11 columns, no missing values, no duplicate rows

**The 11 columns:**

| Column | Meaning |
|---|---|
| `step` | the hour of the simulation (1 to 743, about 31 days) |
| `type` | PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT |
| `amount` | the amount of money moved |
| `nameOrig` / `nameDest` | sender and recipient account IDs (`C...` = customer, `M...` = merchant) |
| `oldbalanceOrg` / `newbalanceOrig` | the sender's balance before and after |
| `oldbalanceDest` / `newbalanceDest` | the recipient's balance before and after |
| `isFraud` | **the label**: 1 = fraud (8,213 rows, 0.13%) |
| `isFlaggedFraud` | the simulator's own simple rule (not used as a feature) |

### Step 2. Explore the data (EDA): `scripts/data/run_eda.py` and `feature_analysis.py`

```powershell
python scripts/data/run_eda.py
python scripts/data/feature_analysis.py
```

- **Uses:** `src/fraud_detection/eda.py` (10 analysis functions)
- **Writes:** `outputs/reports/fraud_by_type.csv`, `feature_comparison.csv`, `amount_analysis.csv`

**Key findings that shaped the whole project:**

1. **Fraud happens only in TRANSFER (0.77% fraud) and CASH_OUT (0.18%)**. PAYMENT, CASH_IN and DEBIT never contain fraud.
2. **The data is extremely imbalanced:** about 1 fraud per 775 transactions. So a model that always says "legitimate" is 99.87% accurate and completely useless. That is why we measure **precision, recall, F1 and average precision**, never accuracy.
3. **Fraud empties accounts.** A typical fraud is a TRANSFER of the whole balance followed by a CASH_OUT, and the balances often "don't add up".
4. **Senders almost never repeat** (99.7% of senders appear once), but recipients do. This matters in Track 3.

### Step 3. Create features: `scripts/data/run_feature_engineering.py`

```powershell
python scripts/data/run_feature_engineering.py      # about 1-2 minutes, needs ~4 GB RAM
```

- **Uses:** `features/engineering.py` → `create_features()`
- **Reads:** `E:\paysim.csv`
- **Writes:** `E:\paysim_features.csv` (859 MB)

A **feature** is a number the model learns from. The six raw numbers and the transaction type become 14 features:

| Feature | Formula | Why it helps |
|---|---|---|
| 6 raw fields | `step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest` | the facts |
| `origin_balance_error` | `oldbalanceOrg − amount − newbalanceOrig` | should be 0; fraud often breaks this |
| `destination_balance_error` | `oldbalanceDest + amount − newbalanceDest` | same for the recipient |
| `amount_to_origin_balance` | `amount / (oldbalanceOrg + 1)` | ≈ 1 means "sends everything" |
| `amount_to_destination_balance` | `amount / (oldbalanceDest + 1)` | large amounts into empty accounts |
| `step_day`, `step_hour` | `(step−1) // 24`, `(step−1) % 24` | time of day |
| `type_TRANSFER`, `type_CASH_OUT` | 1 if the transaction has that type | PaySim fraud only happens in these two types. Added after `scripts/xgboost/compare_transaction_type.py` showed fewer false alarms on the validation set (3 instead of 12, no fraud missed) |

**Important design rule:** `create_features()` is the **only** place these formulas exist. Training uses it, and the API uses it through `prepare_transaction()`. So the model always gets the same numbers in production as in training (this mismatch, "training/serving skew", is a classic bug in ML systems). The tests in `tests/test_features.py` check this.

### Step 4. Train the first model: `scripts/xgboost/train_model.py`

```powershell
python scripts/xgboost/train_model.py               # about 1 minute
```

- **Uses:** `data/splits.py` → `baseline_sample()` (a 10% stratified sample) and `baseline_train_test_split()` (80% train / 20% test)
- **Reads:** `E:\paysim_features.csv`
- **Writes:** `models/fraud_xgboost_model.joblib` (`{"model", "feature_columns"}`)

**What happens:**
1. Take a 10% sample (636,262 rows). *Stratified* means it keeps the same fraud percentage as the full data.
2. Split it 80/20. The 20% test part is never used for training.
3. **Class weighting:** `scale_pos_weight = legitimate count / fraud count` (about 774). Each fraud row counts ~774 times in training, so the model can't ignore the rare class.
4. Train **XGBoost**: 100 decision trees, each one correcting the mistakes of the previous ones (gradient boosting).
5. Print the metrics at threshold 0.5.

**How to read the metrics:**
- **Precision:** of the transactions we flagged, how many were really fraud? (high = few false alarms)
- **Recall:** of all real frauds, how many did we catch? (high = few missed frauds)
- **F1:** a single score that balances the two
- **Average precision (PR-AUC):** quality of the ranking over *all* thresholds; the best single metric for imbalanced data

### Step 5. Pick the threshold properly: `scripts/xgboost/validate_and_test_model.py`

```powershell
python scripts/xgboost/validate_and_test_model.py
```

- **Writes:** `models/fraud_xgboost_validated_model.joblib` (adds `"threshold"`) and `outputs/reports/validated_threshold_results.csv`
- **This is the model the baseline API serves.**

The model outputs a probability, and the **threshold** turns it into a yes/no decision. If you choose the threshold by looking at the test set, the test result looks better than it really is (that's cheating, or "leakage"). So this script splits **70% train / 10% validation / 20% test**:

1. Train on the 70%.
2. Try thresholds 0.1 to 0.9 on the **validation** 10%, and keep the one with the best F1 → **0.3**.
3. Report the final numbers on the untouched **test** 20%.

**Result (test set, 127,253 transactions, 164 frauds):** precision 0.937, recall 0.994, F1 0.964, average precision 0.992. Of the 164 frauds it catches 163, with 11 false alarms among 127,089 legitimate transactions.

### Step 6. Evaluation, threshold table, chart, MLflow

```powershell
python scripts/xgboost/evaluate_model.py            # → outputs/reports/evaluation_results.txt
python scripts/xgboost/threshold_analysis.py        # → outputs/reports/threshold_results.csv
python scripts/xgboost/plot_evaluation_figures.py   # served model → confusion_matrix.png (counts), pr_curve.png
python scripts/xgboost/compare_transaction_type.py  # validation-only check: 12 features vs 12 + transaction type
python scripts/xgboost/track_experiment.py          # → logs a run to MLflow (mlflow.db)
```

These all re-create **the same split** through `baseline_split()` (same seed, 42), so their numbers are comparable.

**MLflow** is an experiment notebook: every run records its parameters, metrics and model file. To browse it:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000      # open http://127.0.0.1:5000
```

### Step 7. Start the baseline API ★

```powershell
uvicorn api.fraud_api:app --port 8000
```

- **File:** `api/fraud_api.py`
- **What it does at start-up:** loads `fraud_xgboost_validated_model.joblib` (the model, the 14 feature names and threshold 0.3)
- **Open:** http://127.0.0.1:8000/docs (an interactive page where you can try the API)

Test it from a second PowerShell window:

```powershell
# a real fraud from paysim.csv (row 3): a TRANSFER that empties the account
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" `
  -Body '{"step":1,"type":"TRANSFER","amount":181,"oldbalanceOrg":181,"newbalanceOrig":0,"oldbalanceDest":0,"newbalanceDest":0}'
# → fraud_probability 0.9997, decision FRAUD_ALERT (the same amounts as a PAYMENT score 0.009)

# a normal payment (row 1)
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" `
  -Body '{"step":1,"type":"PAYMENT","amount":9839.64,"oldbalanceOrg":170136,"newbalanceOrig":160296.36,"oldbalanceDest":0,"newbalanceDest":0}'
# → fraud_probability 0.00004, decision LEGITIMATE
```

**What happens inside one request:** JSON → Pydantic checks the fields (negative, NaN or infinite numbers and unknown types are rejected with error 422) → `prepare_transaction()` builds the 14 features → `model.predict_proba()` → compare with 0.3 → JSON answer. An unexpected error returns only a generic message with an error id; the details go to the server log.

To run it in Docker instead: `docker build -t fraud-api .` then `docker run -p 8000:8000 fraud-api`.

### Step 8. First streaming test: Kafka → baseline API

**Kafka** is a message queue: producers put messages into a *topic*, and consumers read them in order. It's how real banks move transactions between systems.

```powershell
docker compose up -d --wait kafka                    # Kafka at localhost:9092
uvicorn api.fraud_api:app --port 8000                # window 1 (if not already running)
python streaming/kafka_fraud_consumer.py             # window 2: reads topic "transactions", calls the API
python streaming/kafka_paysim_producer.py            # window 3: sends 10 transactions
```

- `kafka_paysim_producer.py` reads the first 10 rows of `paysim_features.csv` and publishes their raw fields and type to the topic `transactions`.
- `kafka_fraud_consumer.py` reads each message, sends the 6 raw fields and `type` to `POST /predict`, and prints the decision next to the true label.
- `streaming/test_kafka_connection.py` is a quick manual check that Kafka is reachable.

---

## 3. Track 2: the shared split, the graph and the GNN

**Why a GNN?** XGBoost looks at one transaction at a time. Fraudsters, though, create *patterns between accounts* (money moves through mule accounts). A **Graph Neural Network** sees accounts as **nodes** and transactions as **edges**, and learns a summary vector (an *embedding*) for each account from its neighbours.

To compare and blend the two models fairly, both must be trained and tested on **exactly the same rows**. That's the shared split.

### Step 9. Create the shared split: `scripts/data/create_shared_split.py`

```powershell
python scripts/data/create_shared_split.py           # about 1 minute
```

- **Uses:** `data/splits.py` → `shared_split()`
- **Writes:** `data/processed/shared_train.csv` and `shared_test.csv` (95 MB; never open it whole)
- **Test:** a stratified 20% of the **full** dataset (about 1.27M rows), kept whole
- **Train:** from the other 80%, **all** fraud rows plus 100,000 random legitimate rows (106,570 rows). Undersampling keeps GNN training fast.

### Step 10. Build the account graph: `scripts/shared/build_shared_train_graph.py`

```powershell
python scripts/shared/build_shared_train_graph.py
```

- **Writes:** `data/processed/shared_train_graph.pt` and `shared_train_graph_transactions.csv`
- **Nodes:** every account seen in **training** transactions. Each node gets 2 features: log(times it sent money), log(times it received money).
- **Edges:** one per training transaction (sender → recipient), labelled fraud or not.

**Leakage prevention:** the graph is built **only from training rows**. A test account the model never saw gets a zero embedding, just as a brand-new customer would in real life.

### Step 11. Train XGBoost on the shared split: `scripts/shared/train_shared_xgboost.py`

```powershell
python scripts/shared/train_shared_xgboost.py
```

- Same 14 features (it takes the list from `models/fraud_xgboost_model.joblib`, so step 4 must have run).
- Trains on 85% of `shared_train.csv`. The other 15% (`splits.shared_validation_split()`, the same rows for the GNN) is the **validation** set used in step 13.
- **Writes:** `models/shared_fraud_xgboost_model.joblib`, `outputs/predictions/shared_xgb_validation_predictions.csv` and `shared_xgb_test_predictions.csv`

### Step 12. Train the GNN: `scripts/shared/train_shared_gnn.py`

```powershell
python scripts/shared/train_shared_gnn.py           # a few minutes on CPU
```

- **Model:** `src/fraud_detection/models/gnn.py` → `FraudGNN`
  - `conv1`, `conv2`: two **GraphSAGE** layers. Each account's vector is updated by averaging its neighbours' vectors, twice, so it "sees" two hops away.
  - `classifier`: a small neural network that scores a transaction from [sender embedding, recipient embedding, 6 scaled transaction numbers].
- **Training:** 15 epochs, Adam optimizer, a weighted loss for the rare fraud class.
- **No edge shortcut:** a transaction is never in the graph used to compute its own accounts' embeddings. Each epoch splits the training transactions 70/30 into "graph" edges and "scored" edges, and accounts with no graph edge get zero embeddings, exactly like unseen accounts at test time. The old method (every transaction in its own graph) is trained too, and the one with the better validation average precision is kept → the new "disjoint" method (0.316 vs 0.310).
- **Writes:** `models/shared_fraud_gnn_model.pt` (weights + account-ID map + scaler), `outputs/reports/shared_gnn_strategy_comparison.csv` and the validation and test predictions

### Step 13. Compare and blend: `scripts/shared/blend_shared_models.py`

```powershell
python scripts/shared/blend_shared_models.py
```

- Checks that both models' prediction files describe the same rows in the same order (`transaction_index`).
- Tries `score = w × XGBoost + (1 − w) × GNN` for 6 weights × 19 thresholds **on the validation set only**, and keeps the best F1. Validation rows are weighted back to the real fraud rate (the training file is ~6% fraud, real data 0.13%).
- Only then scores the test set, once, with the chosen weight and threshold.
- **Writes:** `outputs/reports/shared_blend_validation_selection.csv`, `shared_blend_test_results.csv`, a PR curve and a confusion matrix in `outputs/figures/`

**Result:** validation picks XGBoost weight 1.0 (GNN 0) and threshold 0.95. On the test set (1.27M transactions, 1,643 frauds): average precision 0.9986, precision 0.973, recall 0.998, with 46 false alarms and 4 missed frauds. GNN alone: average precision 0.32. **What this tells us:** in PaySim most accounts appear only once, so the graph has very little structure to learn from. The GNN adds little here. It would matter more on real bank data with repeat customers. This is an honest and useful finding to present.

---

## 4. Track 3: the streaming platform

This is the "production" part. A transaction arrives → the system looks up the recipient's history → scores it → saves it → raises an alert → an analyst reviews it → the labels improve the next model.

### Step 14. Build the feature store: `scripts/feast/build_feature_store.py`

```powershell
python scripts/feast/build_feature_store.py         # several minutes
```

**Problem it solves:** the model wants *history* ("how many transactions did this recipient receive before, and how big were they on average?"). In training we must compute this **as it was at that moment**, never using future rows. In production we must compute **the same numbers** live. A **feature store** (Feast) does both.

- **Uses:** `features/behavioral.py`, `feature_store.py`, `feature_repo/feature_store.yaml`
- **Per recipient account (`C...` only)** it keeps a running **state**: `txn_count`, `amount_sum`, `last_step`, `day_txn_count`
- **Writes:**
  - `feature_repo/data/recipient_stats.parquet`: the state after **every** transaction (the **offline store**, used for training)
  - `feature_repo/data/online_store.db`: the **latest** state per account (the **online store**, SQLite, used by the live service), filled with all history **before step 600** (`REPLAY_START_STEP`)
- **Self-check:** it replays 200,000 rows one by one with the live update function `update_recipient_state()` and checks it gets exactly the same state as the offline version. If not, it stops.

From the state, `recipient_features()` makes **5 behavioral features**: previous transaction count, same-day count (velocity), hours since the last transaction, previous average amount, and this amount relative to that average.

### Step 15. Export the GNN for serving: `scripts/feast/export_gnn_for_serving.py`

```powershell
python scripts/feast/export_gnn_for_serving.py
```

- PyTorch is huge (over 1 GB). We don't want it inside the service container.
- So the script runs the GNN **once**, saves every account's embedding plus the classifier weights as numpy arrays → `models/gnn_serving_artifact.joblib` (29 MB).
- `models/gnn_serving.py` → `GNNServingScorer` recomputes the classifier with numpy only.
- **Self-check:** it compares 20,000 scores with the original PyTorch predictions (max difference must be < 0.0001).

### Step 16. Train the stream model: `scripts/feast/train_stream_model.py`

```powershell
python scripts/feast/train_stream_model.py          # 10+ minutes
```

**19 features** = the 14 baseline features (including `type_TRANSFER`, `type_CASH_OUT`) + 5 behavioral (from Feast, point-in-time).

1. Re-create the shared split and check it matches `shared_train.csv`.
2. Take 25% of the test side and split it into **validation** (half) and **final test** (half).
3. Ask Feast for each row's recipient state **1 ms before** the transaction (no peeking at the future).
4. Add **analyst feedback labels** from the case database (Step 19).
5. Train XGBoost (200 trees, class-weighted).
6. On validation, try GNN weights {0, 0.1, 0.25, 0.5} × thresholds {0.05 … 0.95} and pick the best F1.
7. Evaluate on the final test half.
8. **Champion/challenger:** compare with the model currently in production. The new one replaces it only if its average precision is not worse (tolerance 0.002). Old models are kept in `models/archive/`.
9. Save the **drift reference** (`data/processed/drift_reference_features.parquet`) and log everything to MLflow (experiment `PaySim_Stream_Model`, registry name `paysim-stream-xgboost`).

**Current production model (version 20261020154142):** GNN weight 0, threshold 0.95. Test half (159,066 transactions, 206 frauds): precision 0.962, recall 0.995, F1 0.979, average precision 0.999.

### Step 17. Start the streaming platform ★

You need **4 windows** (or VS Code: *Run and Debug* → **Run streaming platform (Feast + dashboard)** → F5 starts everything at once).

```powershell
# window 1: Kafka
docker compose up -d --wait kafka

# window 2: the fraud service (port 8001)
uvicorn api.fraud_service:app --port 8001
#   no Kafka?  $env:ENABLE_KAFKA_CONSUMER="0"; uvicorn api.fraud_service:app --port 8001
#   → http://127.0.0.1:8001/docs

# window 3: the analyst dashboard (port 8501)
streamlit run dashboard/app.py
#   → http://localhost:8501

# window 4: stream 2,000 real transactions starting at step 600
python streaming/kafka_replay_producer.py
#   more or faster:  $env:REPLAY_MAX_TRANSACTIONS="5000"; $env:REPLAY_DELAY_SECONDS="0.01"
```

**What happens to one transaction (`api/fraud_service.py`, `Service.score_transaction`):**

```
Kafka "transactions" message
  1. the consumer thread receives it
  2. if the recipient is a customer (C...): read its state from the Feast ONLINE store
  3. build_stream_features()  → the same 19 features as in training
  4. StreamScorer.score():  XGBoost probability, GNN probability, blended score, compare with threshold
  5. update_recipient_state() → write the new state back to Feast (the next transaction sees it)
  6. CaseStore.add_scored()  → saved in data/serving/fraud_cases.db with the exact features used
  7. if FRAUD_ALERT: a case is opened (status OPEN) and an alert goes to Kafka topic "fraud_alerts"
```

Steps 2 to 5 run under a **lock**, so two transactions to the same account can't read stale state.

**Last live result:** 2,000 transactions replayed from step 600 → 111 alerts, 110 of them real fraud, and all 110 frauds were caught.

> ⚠️ **Before replaying a second time**, stop the service and run the VS Code task **setup: reset replay**. Otherwise every recipient's history is counted twice. The task deletes `data/serving/fraud_cases.db`, moves the Kafka consumer group to the end of the topic, and rebuilds the online store (Step 14).

### Step 18. The dashboard: `dashboard/app.py`

It only talks to the service API (it never touches the database directly). It has four tabs:

| Tab | What you see |
|---|---|
| **Alerts** | live counters (scored, alerts, open cases, alert precision) refreshed every 5 s, a toast for new alerts, the latest 50 alerts |
| **Case review** | pick a case → the transaction, the scores, **the exact features the model saw** → set the status and a feedback label → Save |
| **All transactions** | everything that was scored |
| **Drift monitor** | the latest drift result per feature + a link to the full Evidently report |

**Case workflow:** `OPEN → INVESTIGATING → CONFIRMED_FRAUD or FALSE_POSITIVE → CLOSED`. Labels are `FRAUD` or `LEGITIMATE`.

### Step 19. Drift monitoring and automatic retraining: `scripts/monitoring/check_drift.py`

```powershell
python scripts/monitoring/check_drift.py                                   # run once
$env:DRIFT_CHECK_INTERVAL_SECONDS="300"; python scripts/monitoring/check_drift.py   # every 5 min
$env:RETRAIN_ON_DRIFT="0"; python scripts/monitoring/check_drift.py        # report only
```

**Drift** means the live data no longer looks like the training data, so the model may get worse.

1. Take the last 2,000 scored feature vectors from the case DB (at least 500 are needed).
2. Compare each feature with the training reference using **Evidently** (statistical distance tests).
3. Save an HTML report (`outputs/reports/drift/`) and log to MLflow (`PaySim_Drift_Monitoring`).
4. If **≥ 50%** of features drifted and no retrain ran in the last 12 hours: run `train_stream_model.py` (Step 16, which includes the analyst labels), then call `POST /model/reload` so the service switches to the new model **without restarting**.

**This really happened on 20 Oct 2026:** 13 of 19 features drifted (68%) → retrain with 2 analyst labels → the new model had average precision 0.9993 against the old model's 0.9989 → promoted and reloaded. (Most of the drift comes from the time features `step`, `step_day`, `step_hour`: the replay is later in time than the training reference, so these drift by design. That's a known limitation worth mentioning.)

### Step 20. Containers and Kubernetes (optional)

```powershell
docker compose --profile platform up -d --build --wait    # Kafka + service + dashboard in containers
```

`k8s/` has Minikube manifests (Kafka, service, dashboard, a drift CronJob, a replay Job). The full commands are in the README. On this machine, Docker Desktop has limited disk and memory, so prefer running the Python processes directly for the demo.

---

## 5. Tests: `tests/`

```powershell
python -m pytest -v
```

| File | What it proves |
|---|---|
| `test_features.py` | feature formulas are right; saved models use the exact feature list; **the live state update equals the offline training state** |
| `test_baseline_api.py` | API on :8000 flags real fraud rows, passes normal payments, rejects bad input (422) |
| `test_stream_service.py` | service on :8001 scores with the blend, opens a case for alerts, the case workflow works (Kafka off, temporary DB, Feast read-only) |
| `test_case_store.py` | case statuses, feedback labels, statistics |
| `test_model_quality.py` | on 200,000 held-out rows: average precision ≥ 0.95 and recall ≥ 0.95; the numpy GNN equals the PyTorch GNN |

---

## 6. What is safe to re-run?

| Step | Re-run safe? | Notes |
|---|---|---|
| 1, 2 (checks, EDA) | ✅ | only print / write reports |
| 3 (features) | ✅ | rewrites `paysim_features.csv`; same result |
| 4, 5, 6 (baseline XGBoost) | ✅ | fixed seed → same model; restart the API afterwards |
| 9 to 13 (shared track) | ⚠️ | same split; the GNN may change slightly → then also re-run 15 |
| 14 (feature store) | ✅ | also the way to **reset** the online store before a replay |
| 15 (GNN export) | ✅ | fails loudly if it doesn't match the trained GNN |
| 16 (stream model) | ⚠️ | may **replace the production model** (champion/challenger); then `POST /model/reload` |
| 17 (replay) | ⚠️ | reset first (see the warning in Step 17) |

**Heavy steps** (3, 9, 14, 16) read the full 471 MB / 834 MB CSV files and use several GB of RAM. Close other programs first.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **Stratified split** | a split that keeps the same fraud percentage in each part |
| **Class imbalance / `scale_pos_weight`** | fraud is rare; weighting makes each fraud row count more |
| **Threshold** | the probability above which we raise an alert |
| **Precision / recall / F1** | false-alarm rate vs missed-fraud rate vs their balance |
| **Average precision (PR-AUC)** | ranking quality over all thresholds; the main metric here |
| **Data leakage** | using information in training that wouldn't exist at prediction time |
| **Point-in-time join** | fetching features as they were just before each event |
| **Embedding** | a learned vector that summarizes an account |
| **GraphSAGE** | a GNN layer that builds a node's vector from its neighbours |
| **Feature store (Feast)** | one place that serves the same features to training (offline) and production (online) |
| **Kafka topic** | a named, ordered stream of messages |
| **Drift** | live data stops looking like training data |
| **Champion / challenger** | a new model replaces the old one only if it is at least as good |
| **MLflow** | records every experiment's settings, metrics and model files |
