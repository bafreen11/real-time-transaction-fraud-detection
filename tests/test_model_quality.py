"""Model quality gates on held-out data (skipped without the shared split).

Reads only the first rows of shared_test.csv, which none of the models
were trained on.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, recall_score

from fraud_detection.config import (
    GNN_SERVING_ARTIFACT_PATH,
    SHARED_GNN_PREDICTIONS_PATH,
    SHARED_TEST_PATH,
    SHARED_XGBOOST_MODEL_PATH,
)
from fraud_detection.features.engineering import create_features
from fraud_detection.models.gnn_serving import GNNServingScorer

ROWS = 200_000

pytestmark = pytest.mark.skipif(
    not SHARED_TEST_PATH.exists(),
    reason="shared split not built (run scripts/data/create_shared_split.py)",
)


@pytest.fixture(scope="module")
def test_rows():
    return pd.read_csv(SHARED_TEST_PATH, nrows=ROWS)


def test_shared_xgboost_catches_fraud(test_rows):
    saved = joblib.load(SHARED_XGBOOST_MODEL_PATH)
    X = create_features(test_rows)[saved["feature_columns"]]
    y = test_rows["isFraud"].to_numpy()

    probabilities = saved["model"].predict_proba(X)[:, 1]

    assert y.sum() > 100  # enough fraud for the numbers to mean something
    assert average_precision_score(y, probabilities) >= 0.95
    assert recall_score(y, probabilities >= 0.5) >= 0.95


@pytest.mark.skipif(
    not SHARED_GNN_PREDICTIONS_PATH.exists(),
    reason="shared GNN predictions not saved",
)
def test_torch_free_gnn_reproduces_trained_gnn(test_rows):
    """The numpy GNN used in serving gives the same scores as torch."""
    scorer = GNNServingScorer(joblib.load(GNN_SERVING_ARTIFACT_PATH))
    saved = pd.read_csv(SHARED_GNN_PREDICTIONS_PATH, nrows=5_000)

    served = scorer.score(test_rows.head(5_000))
    assert np.abs(served - saved["gnn_fraud_probability"].to_numpy()).max() < 1e-4
