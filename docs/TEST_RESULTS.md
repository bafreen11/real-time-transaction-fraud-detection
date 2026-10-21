# Test results and model performance

All numbers come from saved reports, MLflow runs (`mlflow.db`) or a re-run on 21 Oct 2026. Fraud is 0.13% of transactions, so accuracy is not reported: a model that never raises an alert would score 99.87%.

## 1. Automated tests

```powershell
python -m pytest -v
```

**Result: 28 passed, 0 failed (about 20 s).**

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_features.py` | 11 | feature formulas; saved models match the feature list; live recipient-state update equals the offline training state; 19-feature stream vector |
| `tests/test_baseline_api.py` | 6 | `GET /`, fraud row → `FRAUD_ALERT`, normal payment → `LEGITIMATE`, invalid input → 422 |
| `tests/test_stream_service.py` | 5 | health, blended score formula, alert opens a case, case update via API, 404 / 422 handling |
| `tests/test_case_store.py` | 4 | alert vs non-alert cases, status workflow, feedback labels for retraining, invalid status/label |
| `tests/test_model_quality.py` | 2 | shared XGBoost on 200,000 held-out rows: AP ≥ 0.95, recall ≥ 0.95; numpy GNN = PyTorch GNN (diff < 1e-4) |

The service tests run with Kafka off, a temporary case database and read-only Feast access, so they never change the real data.

Built-in pipeline checks (they stop the script if they fail):

| Check | Where |
|---|---|
| offline recipient snapshots = streaming updates on 200,000 rows | `scripts/feast/build_feature_store.py` |
| exported numpy GNN reproduces PyTorch predictions on 20,000 rows (max diff < 1e-4) | `scripts/feast/export_gnn_for_serving.py` |
| recreated split matches `shared_train.csv` | `scripts/feast/train_stream_model.py` |
| XGBoost and GNN prediction files are row-aligned (validation and test) | `scripts/shared/blend_shared_models.py` |
| served features = training features for 500 streamed transactions (max diff 0.0) | manual check during development |

## 2. Baseline XGBoost (served by the API on port 8000)

10% stratified sample of `E:\paysim_features.csv`; test set of 127,253 transactions with 164 frauds (the same rows for both models). 14 features: the 12 numeric features plus `type_TRANSFER` and `type_CASH_OUT`.

| Model | Split | Threshold | Avg. precision | Precision | Recall | F1 | False-positive rate | Script |
|---|---|---|---|---|---|---|---|---|
| `fraud_xgboost_model` | 80/20 | 0.5 (fixed) | 0.9919 | 0.9588 | 0.9939 | 0.9760 | 0.000055 | `evaluate_model.py` |
| **`fraud_xgboost_validated_model` (served)** | 70/10/20 | **0.3** (best F1 on validation) | **0.9916** | **0.9368** | **0.9939** | **0.9645** | **0.000087** | `plot_evaluation_figures.py` |

Confusion matrix of the served model (test set, threshold 0.3):

| | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actual legitimate** | 127,078 | 11 |
| **Actual fraud** | 1 | 163 |

ROC-AUC: 0.99995. Figures: `outputs/figures/confusion_matrix.png`, `outputs/figures/pr_curve.png`. Saved metrics: `outputs/reports/served_model_test_results.csv`. Threshold sweep for the first model (exploratory, on the test set, not used to choose anything): `outputs/reports/threshold_results.csv`.

**Transaction type experiment** (`scripts/xgboost/compare_transaction_type.py`, validation set only: 63,627 rows, 82 frauds; `outputs/reports/transaction_type_comparison.csv`):

| Features | Val. avg. precision | Best threshold | Val. precision | Val. recall | Val. F1 | False alarms |
|---|---|---|---|---|---|---|
| 12 numeric | 0.9981 | 0.4 | 0.8723 | 1.0000 | 0.9318 | 12 |
| **12 + type flags (adopted)** | **0.9996** | **0.3** | **0.9647** | **1.0000** | **0.9820** | **3** |

Before the type flags, the served model scored AP 0.9856, precision 0.8703, recall 0.9817, F1 0.9226 at threshold 0.4 on the same test set.

## 3. Shared split: XGBoost vs GNN vs blend

**Selection (validation only).** 15% of `shared_train.csv` (15,986 rows, 986 frauds) is held out; both models train on the other 85%. `shared_train.csv` is ~6% fraud (all fraud + 100,000 legitimate rows), so validation metrics weight each legitimate row by ~50.8 to restore the natural 0.13% fraud rate. Without this, the chosen threshold would raise far too many alerts on real traffic.

GNN training strategy (`outputs/reports/shared_gnn_strategy_comparison.csv`, threshold 0.5):

| Strategy | Val. avg. precision (weighted) |
|---|---|
| **disjoint: a scored transaction is never in its own message-passing graph (kept)** | **0.3156** |
| transductive: the original method, every training edge is also a message edge | 0.3098 |

Best threshold per blend weight on validation (`outputs/reports/shared_blend_validation_selection.csv`; ties go to the higher threshold):

| XGBoost / GNN weight | Threshold | Val. avg. precision | Val. precision | Val. recall | Val. F1 |
|---|---|---|---|---|---|
| **1.00 / 0.00 (selected)** | **0.95** | **0.9944** | **1.0000** | **0.9939** | **0.9969** |
| 0.90 / 0.10 | 0.90 | 0.9939 | 1.0000 | 0.9939 | 0.9969 |
| 0.75 / 0.25 | 0.80 | 0.9939 | 1.0000 | 0.9939 | 0.9969 |
| 0.50 / 0.50 | 0.65 | 0.9939 | 1.0000 | 0.9939 | 0.9969 |
| 0.25 / 0.75 | 0.55 | 0.9939 | 1.0000 | 0.9888 | 0.9944 |
| 0.00 / 1.00 | 0.65 | 0.3156 | 0.5169 | 0.2759 | 0.3597 |

**Test (scored once, after the choices were fixed).** Full 20% test set: 1,272,524 transactions, 1,643 frauds (`outputs/reports/shared_blend_test_results.csv`):

| Model (threshold from validation) | Avg. precision | Precision | Recall | F1 | False alarms | Missed fraud |
|---|---|---|---|---|---|---|
| **Selected: XGBoost only (0.95)** | **0.9986** | **0.9727** | **0.9976** | **0.9850** | **46** | **4** |
| GNN only (0.65), for reference | 0.3166 | 0.5277 | 0.2666 | 0.3542 | 392 | 1,205 |

Figures: `outputs/figures/shared_blend_pr_curve.png`, `outputs/figures/shared_blend_confusion_matrix.png`.

**Takeaway:** the GNN alone is weak (AP 0.32) because PaySim accounts rarely repeat, so the graph has little structure. On validation, adding it to XGBoost never beats XGBoost alone, so the selected model gives the GNN weight 0. Removing the GNN's edge shortcut did not change its test AP noticeably (0.3166, against 0.3206 for the earlier GNN).

Earlier versions of this table compared blend weights on the test set at threshold 0.5. That was exploratory and has been replaced by the validation selection above.

## 4. Stream model (served by the fraud service on port 8001)

Test half of a 25% stratified sample of `shared_test.csv`: 159,066 transactions, 206 frauds. Blend weight and threshold chosen on the other (validation) half.

| Run | Trigger | GNN weight / threshold | Avg. precision | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| `20261020152028` | manual | 0.1 / 0.90 | 0.9989 | 0.9624 | 0.9951 | 0.9785 |
| **`20261020154142` (in production)** | **drift (share 0.68), 2 analyst labels** | **0.0 / 0.95** | **0.9993** | **0.9624** | **0.9951** | **0.9785** |

Comparison inside the first run (same test rows):

| Model | Avg. precision | Precision | Recall | F1 |
|---|---|---|---|---|
| Blend (0.9 / 0.1, threshold 0.9) | 0.9989 | 0.9624 | 0.9951 | 0.9785 |
| XGBoost only (threshold 0.9) | 0.9993 | 0.9361 | 0.9951 | 0.9647 |
| GNN only (threshold 0.5) | 0.3103 | 0.0047 | 0.8883 | 0.0094 |

The GNN serving artifact was re-exported from the new GNN (see section 3). The production model gives the GNN weight 0.0, so its scores and the numbers above are unchanged; the "GNN only" row is the earlier GNN.

With the 5 recipient-history features from Feast, precision is 0.96 on this test sample. The test sets and training data differ, so this comparison is indicative, not exact.

## 5. Live streaming replay

2,000 transactions replayed through Kafka from step 600 into the fraud service:

| Scored | Alerts | True alerts | Actual frauds | Frauds caught |
|---|---|---|---|---|
| 2,000 | 111 | 110 (99.1% alert precision) | 110 | 110 (100%) |

## 6. Drift monitoring and retraining (20 Oct 2026)

| Step | Result |
|---|---|
| Drift check on the last 2,000 scored transactions vs 20,000 reference rows | 13 of 19 features drifted → share 0.68 ≥ 0.50 |
| Automatic retrain | 106,572 training rows (incl. 2 analyst labels) |
| Champion/challenger | new AP 0.9993 ≥ current 0.9989 − 0.002 → **promoted** |
| Service | reloaded via `POST /model/reload`, no restart |

The time features (`step`, `step_day`, `step_hour`) drift by design because the replay is later in time than the training reference. A production version should exclude them from the drift decision.

## 7. Performance

Latency was measured in-process (FastAPI TestClient, 300 sequential requests after 20 warm-up requests, one laptop CPU, no network):

| Endpoint | Work per request | Median | p95 | Throughput (one worker) |
|---|---|---|---|---|
| Baseline `POST /predict` | 12 features, one XGBoost call | 24 ms | 37 ms | ~37 req/s |
| Service `POST /score` | Feast online read, 19 features, XGBoost + numpy GNN, SQLite case write | 91 ms | 193 ms | ~10 req/s |

| Other | Value |
|---|---|
| Test suite | 28 tests in about 20 s |
| GNN serving artifact | 29 MB numpy, no PyTorch at serving time |

Most of the service's time goes to the Feast SQLite lookup and the case-store write. Scaling options: Redis as the online store, more uvicorn workers with Kafka partitions, and batching writes.
