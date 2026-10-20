# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fraud detection on the PaySim synthetic mobile-money dataset. The project covers EDA, feature engineering, XGBoost training and threshold selection, a GraphSAGE GNN over the account→account transaction graph, XGBoost+GNN score blending, MLflow tracking, a FastAPI scoring service, and a Kafka producer/consumer that streams transactions through that API. See README.md for results and the layout.

## Layout and conventions

- `src/fraud_detection/` is the only importable code. It holds `config.py` (every path and experiment constant), `data/loading.py`, `data/splits.py`, `features/engineering.py`, `models/gnn.py` and `eda.py`.
- `scripts/{data,xgboost,shared,diagnostics}/`, `api/` and `streaming/` contain standalone scripts that run top to bottom (no `main()`). Never import them.
- **Never hard-code paths.** Add a constant to `fraud_detection.config` and import it. Output-writing scripts call `config.ensure_output_dirs()` first.
- The raw data lives outside the repo: `E:\paysim.csv` and `E:\paysim_features.csv`, overridable with the `PAYSIM_RAW_PATH` and `PAYSIM_FEATURES_PATH` environment variables.
- `archive/` holds superseded code (legacy graph prototype, backups, broken notebook leftovers) kept untouched. `archive/MOVE_LOG.txt` maps every file's old root location to its new one. Don't update or import from `archive/`.
- Large files: `data/processed/shared_test.csv` (~100 MB) and the files in `outputs/predictions/` (25–65 MB). Use `nrows` or chunking. Never read them whole.

## Commands

Run everything from the project root. Imports need the package installed (`pip install -e .`) or `PYTHONPATH=src`. The API also needs the project root on the path, which uvicorn provides when run from the root.

```powershell
pip install -r requirements-dev.txt; pip install -e .
python scripts/xgboost/train_model.py               # any script: python <path>
uvicorn api.fraud_api:app --port 8000               # API
docker build -t fraud-api .                         # image uses PYTHONPATH=/app/src, copies only api/, src/ and the served model
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

There are no automated tests or linters. `streaming/test_kafka_connection.py` is a manual smoke script. `requirements.txt` covers only the API image; `requirements-dev.txt` adds torch, torch_geometric, mlflow, kafka-python, requests and matplotlib.

## Pipelines

There are two separate modeling tracks. Do not mix their artifacts.

**1. Baseline XGBoost track** (`scripts/xgboost/`; 10% stratified sample of `paysim_features.csv`, then an 80/20 split)
`scripts/data/run_feature_engineering.py` → `train_model.py` (writes `models/fraud_xgboost_model.joblib`, threshold 0.5) → `validate_and_test_model.py` (70/10/20 split; chooses the F1-best threshold on validation; writes `models/fraud_xgboost_validated_model.joblib`, **the model the API serves**). Evaluation, threshold analysis, the confusion-matrix plot and MLflow tracking all re-create the split through `fraud_detection.data.splits.baseline_split`. Change sampling only there and in `config`.

**2. Shared track** (one fixed split used by both XGBoost and the GNN, so their scores can be blended)
`scripts/data/create_shared_split.py` (stratified 80/20 split of the full raw data; the training side keeps all fraud plus 100k sampled legitimate rows) → `scripts/shared/build_shared_train_graph.py` → `train_shared_xgboost.py` and `train_shared_gnn.py` → `blend_shared_models.py`.
- Prediction CSVs are aligned on `transaction_index` (the row position in `shared_test.csv`), and the blend script checks this. Keep row order unchanged.
- The GNN builds account IDs and node features (log sender/recipient counts) from **training transactions only**. Test accounts never seen in training map to `-1` and get zero embeddings. This is deliberate leakage prevention.
- `train_shared_xgboost.py` takes its feature list from `models/fraud_xgboost_model.joblib`, so that file has to exist.
- Blend weights are exploratory and evaluated on the test set. The saved 0.75/0.25 blend is a candidate, not a selected model.

## Model / feature contract

- Joblib artifacts are dicts: `{"model", "feature_columns"}`, plus `"threshold"` in the validated model.
- `features/engineering.py` defines `RAW_TRANSACTION_FIELDS` (the 6 API inputs), `ENGINEERED_FEATURES` and `FEATURE_COLUMNS` (12, in training order; mirrored in `models/feature_columns.json`). `create_features` is the single feature implementation. The API and the debug script go through `prepare_transaction`, so training and serving can't drift apart.
- GNN `.pt` checkpoints hold `model_state_dict`, `account_to_id`, the fitted `StandardScaler`, `edge_feature_columns` and `hidden_channels`. Keep the `FraudGNN` layer names (`conv1`, `conv2`, `classifier`) unchanged, or existing checkpoints won't load.

## MLflow

Runs are stored in `mlflow.db` (SQLite). Their artifacts are in `mlruns/` under **absolute** `file:///E:/Transactions Fraud Detection/mlruns/...` URIs, so both must stay at the project root. `config.MLFLOW_TRACKING_URI` points at the database regardless of the working directory.
