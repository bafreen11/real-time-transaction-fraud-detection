# Project report: real-time transaction fraud detection

**Author:** [Your name] · **Internship:** [team / company] · **Date:** 21 October 2026
**Repository:** https://github.com/bafreen11/real-time-transaction-fraud-detection

## 1. Summary

This project detects fraudulent mobile-money transactions in the PaySim dataset (6.36 million transactions, 0.13% fraud) and runs the detector as a live system.

- A **baseline XGBoost model** on 14 features, served by a REST API, catches 163 of 164 test frauds with 11 false alarms among 127,089 legitimate transactions (precision 0.937, recall 0.994, F1 0.965).
- A **graph neural network** (GraphSAGE) over the account graph was compared with XGBoost on one shared split. It does not help on this dataset (average precision 0.32 against 0.999), because PaySim accounts rarely repeat.
- A **streaming platform** scores transactions from Kafka with account history from a Feast feature store. Alerts go to an analyst dashboard. A drift monitor retrains the model and promotes it only if it is at least as good. The production model reaches precision 0.962 and recall 0.995 on held-out data. In a live replay of 2,000 transactions it caught all 110 frauds, and 110 of its 111 alerts were real.
- **32 automated tests** cover the features, both APIs, the analyst workflow and minimum model quality.

## 2. Problem

Mobile-money fraud is rare and fast. A fraudster takes over an account, transfers the whole balance to a mule account and cashes it out. A detector must:

1. find a very rare class: about 1 transaction in 775;
2. decide while the transaction happens, using what is known about the accounts involved at that moment;
3. keep a human analyst in the loop, with low false alarms;
4. stay reliable when transaction patterns change.

Because of point 1, accuracy is meaningless: a model that never flags fraud is 99.87% accurate. This report uses **precision** (share of alerts that are real fraud), **recall** (share of frauds caught), **F1**, and **average precision** (ranking quality over all thresholds).

## 3. Data

PaySim (Kaggle) simulates 31 days of an African mobile-money service: 6,362,620 transactions with type, amount, sender and recipient accounts, their balances before and after, and a fraud label.

Exploration showed four things that drove the design:

| Finding | Consequence |
|---|---|
| Fraud occurs only in TRANSFER (0.77% fraud) and CASH_OUT (0.18%) | transaction-type features |
| Fraud empties the sender's account, and balances often don't add up | balance-error and ratio features |
| 99.7% of senders appear only once | account history is kept per **recipient** |
| 8,213 fraud rows in 6.36M | class weighting; precision-recall metrics |

## 4. Approach

### 4.1 Baseline XGBoost

14 features: the 6 raw amounts and balances, 4 balance checks (for example `oldbalanceOrg − amount − newbalanceOrig`), day and hour, and TRANSFER / CASH_OUT flags. The type flags were added only after a validation comparison: false alarms fell from 12 to 3 with no fraud missed. XGBoost is trained with `scale_pos_weight` (legitimate ÷ fraud ≈ 774) on a 10% stratified sample split 70/10/20. The alert threshold (0.3) is the best F1 on the validation 10%, and the test 20% is scored once.

### 4.2 Graph neural network

Accounts are nodes and transactions are edges. A two-layer GraphSAGE encoder learns account embeddings, and a small classifier scores each transaction from the sender embedding, the recipient embedding and the scaled amounts. To prevent leakage:

- the graph contains training transactions only, and unseen accounts get zero embeddings;
- during training, a transaction is never part of the message passing used to score it.

XGBoost and the GNN are trained on the same split (all training fraud + 100,000 legitimate rows; 1.27M test rows), so their scores can be blended. The blend weight and threshold are chosen on validation, with legitimate rows re-weighted to the real fraud rate.

### 4.3 Streaming platform

```
replay producer → Kafka "transactions" → fraud service (FastAPI)
      1. recipient history from the Feast online store
      2. 19 features (14 + 5 history), the same code as in training
      3. XGBoost + GNN blended score
      4. updated history written back to Feast
      5. transaction, features and score saved; an alert opens a case → analyst dashboard
      6. alert published to Kafka "fraud_alerts"
drift monitor (Evidently) → retrain with analyst labels → promote if not worse → hot reload
```

- **Feature store.** Feast holds each recipient's running state (transaction count, amount sum, last step, same-day count). Training reads it point-in-time (1 ms before each transaction) from the offline store; the service reads and updates the online store. A build-time check confirms that the streaming update and the offline computation agree.
- **Serving the GNN without PyTorch.** Account embeddings and classifier weights are exported to numpy (29 MB). The export is checked against the PyTorch predictions (difference < 0.0001).
- **Analyst workflow.** Cases move `OPEN → INVESTIGATING → CONFIRMED_FRAUD / FALSE_POSITIVE → CLOSED`. Labels are added to the training data on the next retrain.
- **Model lifecycle.** Every training run and drift check is tracked in MLflow. A retrained model replaces production only if its test average precision is not lower (tolerance 0.002), and the previous model is archived.
- **Deployment.** Docker images for the API, service and dashboard; Docker Compose for local runs; Kubernetes (Minikube) manifests including a drift CronJob.

## 5. Results

All thresholds and weights were chosen on validation data. Each test set was scored once. The tracks use different test sets, so compare within a track.

| Model | Test set | Threshold | Avg. precision | Precision | Recall | F1 | False alarms / missed |
|---|---|---|---|---|---|---|---|
| Baseline API (14 features) | 127,253 rows, 164 fraud | 0.30 | 0.9916 | 0.9368 | 0.9939 | 0.9645 | 11 / 1 |
| Shared track, XGBoost (selected blend: GNN weight 0) | 1,272,524 rows, 1,643 fraud | 0.95 | 0.9986 | 0.9727 | 0.9976 | 0.9850 | 46 / 4 |
| Shared track, GNN alone | same | 0.65 | 0.3166 | 0.5277 | 0.2666 | 0.3542 | 392 / 1,205 |
| **Stream model (production)** | 159,066 rows, 206 fraud | 0.95 | **0.9993** | **0.9624** | **0.9951** | **0.9785** | 8 / 1 |

**Live replay.** 2,000 transactions from hour 600 were streamed through Kafka: 111 alerts, 110 real fraud, all 110 frauds caught.

**Drift and retraining (20 October 2026).** 13 of 19 features drifted (68%, above the 50% trigger). The model retrained automatically with 2 analyst labels. The new model's average precision was 0.9993 against 0.9989 for the old one, so it was promoted and hot-reloaded without a restart.

**Speed** (in-process, one CPU): baseline API median 24 ms per request; streaming service median 91 ms, mostly the feature-store read/write and the case write.

**Tests.** 32 automated tests pass. Charts: `outputs/figures/` (confusion matrices and precision-recall curves). Full details: [TEST_RESULTS.md](TEST_RESULTS.md).

## 6. Challenges and solutions

| Challenge | Solution |
|---|---|
| Extreme class imbalance | class weighting, precision-recall metrics, threshold chosen on validation |
| Training/serving skew | one feature function used by training and both APIs, covered by tests |
| Leakage in the graph and in account history | training-only graph, no message passing through scored edges, point-in-time lookups |
| No sender history in PaySim | history tracked per recipient account |
| PyTorch too large for the service image | GNN exported to numpy and verified |
| Dependency conflicts (Feast needs numpy < 2; a pyarrow version crashed PyTorch on Windows) | pinned, verified versions |
| Replays double-counting account history | a reset task that rebuilds the online store |

## 7. Limitations

- PaySim is synthetic. Accounts rarely repeat, and there are no device, location or card fields, so the graph model and history features have less to work with than on real data.
- The time features drift by design during a replay, which inflated the drift share. They should be excluded from the drift decision.
- The online store is SQLite: fine for a demo, not for production throughput.
- The drift trigger looks at feature distributions only, not at live precision.

## 8. Next steps

1. Repeat the study on real-world-like data with devices and repeat customers (for example IEEE-CIS).
2. Show SHAP explanations for each alert on the dashboard.
3. Improve the drift trigger: leave out time features; add live precision from analyst labels.
4. Scale the service: Redis online store, more workers and Kafka partitions.
5. Add CI/CD: run the tests on every push and build images automatically.
6. Deploy to a managed Kubernetes cluster.

## 9. How to run

Setup, every pipeline step and the start commands: [WALKTHROUGH.md](WALKTHROUGH.md). Architecture and data: [ARCHITECTURE.md](ARCHITECTURE.md). Endpoints: [API.md](API.md). Demo: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

```powershell
pip install -r requirements-dev.txt; pip install -e .
python -m pytest                                   # 32 tests
uvicorn api.fraud_api:app --port 8000              # baseline API
uvicorn api.fraud_service:app --port 8001          # streaming service
streamlit run dashboard/app.py                     # analyst dashboard
```
