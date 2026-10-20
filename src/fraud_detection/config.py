"""Central paths and constants for the PaySim fraud detection project.

Every script imports its file locations from here, so moving data or
artifacts only requires a change in this module.
"""

import os
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
MLFLOW_TRACKING_URI = (
    "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
)


def ensure_output_dirs():
    """Create every output directory if it does not exist yet."""
    for directory in (
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
