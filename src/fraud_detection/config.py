"""Central paths and constants for the PaySim fraud detection project.

Every script imports its file locations from here, so moving data or
artifacts only requires a change in this module.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------
# Project layout
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# --------------------------------------------------
# External datasets (too large to keep in the project)
# Override with the PAYSIM_RAW_PATH / PAYSIM_FEATURES_PATH
# environment variables.
# --------------------------------------------------

RAW_DATA_PATH = Path(
    os.environ.get("PAYSIM_RAW_PATH", r"E:\paysim.csv")
)
FEATURES_DATA_PATH = Path(
    os.environ.get("PAYSIM_FEATURES_PATH", r"E:\paysim_features.csv")
)

# --------------------------------------------------
# Shared train/test split and training graph
# --------------------------------------------------

SHARED_TRAIN_PATH = PROCESSED_DATA_DIR / "shared_train.csv"
SHARED_TEST_PATH = PROCESSED_DATA_DIR / "shared_test.csv"
SHARED_GRAPH_PATH = PROCESSED_DATA_DIR / "shared_train_graph.pt"
SHARED_GRAPH_TRANSACTIONS_PATH = (
    PROCESSED_DATA_DIR / "shared_train_graph_transactions.csv"
)

# --------------------------------------------------
# Model artifacts
# --------------------------------------------------

XGBOOST_MODEL_PATH = MODELS_DIR / "fraud_xgboost_model.joblib"
XGBOOST_VALIDATED_MODEL_PATH = (
    MODELS_DIR / "fraud_xgboost_validated_model.joblib"
)
SHARED_XGBOOST_MODEL_PATH = (
    MODELS_DIR / "shared_fraud_xgboost_model.joblib"
)
SHARED_GNN_MODEL_PATH = MODELS_DIR / "shared_fraud_gnn_model.pt"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"

# --------------------------------------------------
# Predictions, reports and figures
# --------------------------------------------------

SHARED_XGB_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "shared_xgb_test_predictions.csv"
)
SHARED_GNN_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "shared_gnn_test_predictions.csv"
)
SHARED_BLENDED_PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "shared_blended_test_predictions.csv"
)

SHARED_BLEND_COMPARISON_PATH = REPORTS_DIR / "shared_blend_comparison.csv"
THRESHOLD_RESULTS_PATH = REPORTS_DIR / "threshold_results.csv"
VALIDATED_THRESHOLD_RESULTS_PATH = (
    REPORTS_DIR / "validated_threshold_results.csv"
)
EVALUATION_RESULTS_PATH = REPORTS_DIR / "evaluation_results.txt"

CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

RANDOM_STATE = 42

# Baseline XGBoost track: 10% stratified sample, 80/20 split
BASELINE_SAMPLE_FRACTION = 0.10
BASELINE_TEST_FRACTION = 0.20

# Shared track: 80/20 split of the full dataset,
# training side keeps all fraud plus a legitimate sample
SHARED_TEST_FRACTION = 0.20
SHARED_TRAIN_LEGITIMATE_SAMPLE = 100_000

MLFLOW_EXPERIMENT_NAME = "PaySim_Fraud_Detection"

# Existing runs live in mlflow.db and their artifacts in mlruns/,
# both at the project root. Do not move them: run records store
# absolute artifact paths.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix(),
)

# --------------------------------------------------
# Streaming track: Feast feature store, stream model,
# scoring service, drift monitoring
# --------------------------------------------------

# Feast repository (feature_store.yaml) and its local data
# (offline parquet source, registry, SQLite online store).
FEATURE_REPO_DIR = PROJECT_ROOT / "feature_repo"
FEAST_DATA_DIR = FEATURE_REPO_DIR / "data"
RECIPIENT_STATS_PATH = FEAST_DATA_DIR / "recipient_stats.parquet"

# PaySim "step" is one simulated hour. Steps are mapped onto real
# timestamps so Feast can do point-in-time joins.
PAYSIM_START_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)

STREAM_XGBOOST_MODEL_PATH = MODELS_DIR / "stream_xgboost_model.joblib"
GNN_SERVING_ARTIFACT_PATH = MODELS_DIR / "gnn_serving_artifact.joblib"
STREAM_MODEL_ARCHIVE_DIR = MODELS_DIR / "archive"

# Sample of training features kept as the drift reference.
DRIFT_REFERENCE_PATH = PROCESSED_DATA_DIR / "drift_reference_features.parquet"

# Fraction of shared_test.csv scored when training the stream model.
# Half of the sample picks the threshold and blend weight, the other
# half is the untouched test set.
STREAM_EVAL_FRACTION = float(os.environ.get("STREAM_EVAL_FRACTION", "0.25"))
STREAM_BLEND_GNN_WEIGHTS = [0.0, 0.1, 0.25, 0.5]
STREAM_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]

MLFLOW_STREAM_EXPERIMENT_NAME = "PaySim_Stream_Model"
MLFLOW_DRIFT_EXPERIMENT_NAME = "PaySim_Drift_Monitoring"
STREAM_MODEL_REGISTRY_NAME = "paysim-stream-xgboost"

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
)
KAFKA_TRANSACTIONS_TOPIC = os.environ.get(
    "KAFKA_TRANSACTIONS_TOPIC", "transactions"
)
KAFKA_ALERTS_TOPIC = os.environ.get("KAFKA_ALERTS_TOPIC", "fraud_alerts")
KAFKA_SERVICE_GROUP_ID = "paysim-fraud-service"

# Replay: the online store is bootstrapped with all history before
# REPLAY_START_STEP, then the producer streams from that step on.
REPLAY_START_STEP = int(os.environ.get("REPLAY_START_STEP", "600"))

# Scoring service state (scored transactions, alert cases, feedback)
SERVING_DATA_DIR = DATA_DIR / "serving"
SERVING_DB_PATH = Path(
    os.environ.get("SERVING_DB_PATH", SERVING_DATA_DIR / "fraud_cases.db")
)

# Drift monitoring and automatic retraining
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
DRIFT_LATEST_SUMMARY_PATH = DRIFT_REPORTS_DIR / "latest_summary.json"
DRIFT_LATEST_REPORT_PATH = DRIFT_REPORTS_DIR / "latest_report.html"
RETRAIN_STATE_PATH = DRIFT_REPORTS_DIR / "retrain_state.json"
DRIFT_WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "2000"))
DRIFT_MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", "500"))
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.5"))
RETRAIN_COOLDOWN_HOURS = float(os.environ.get("RETRAIN_COOLDOWN_HOURS", "12"))
# Retrained model replaces the current one only if its test average
# precision is at least the current model's minus this tolerance.
RETRAIN_PROMOTION_TOLERANCE = 0.002


def ensure_output_dirs():
    """Create every output directory if it does not exist yet."""
    for directory in (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        FEAST_DATA_DIR,
        SERVING_DB_PATH.parent,
        DRIFT_REPORTS_DIR,
        STREAM_MODEL_ARCHIVE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
