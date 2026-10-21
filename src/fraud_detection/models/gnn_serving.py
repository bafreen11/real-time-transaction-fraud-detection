"""Torch-free scoring with the shared-track GraphSAGE model.

scripts/feast/export_gnn_for_serving.py runs the trained FraudGNN
encoder once over the training graph and saves the account embeddings,
the edge-feature scaler and the classifier weights. Scoring a
transaction is then a lookup plus a small MLP in numpy, which keeps
torch out of the serving image.

Accounts that were not in the training graph get zero embeddings,
exactly as in scripts/shared/train_shared_gnn.py.
"""

import numpy as np
import pandas as pd

# Raw edge features and the log-transformed subset, as in
# scripts/shared/train_shared_gnn.py.
GNN_EDGE_FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
GNN_LOG_COLUMNS = GNN_EDGE_FEATURE_COLUMNS[1:]


def prepare_gnn_edge_features(df):
    features = df[GNN_EDGE_FEATURE_COLUMNS].astype(float).copy()
    for col in GNN_LOG_COLUMNS:
        features[col] = np.log1p(features[col].clip(lower=0))
    return features


class GNNServingScorer:
    """Scores transactions with exported GNN embeddings and weights."""

    def __init__(self, artifact):
        self.embeddings = artifact["embeddings"]
        self.account_to_id = artifact["account_to_id"]
        self.scaler = artifact["scaler"]
        self.w1, self.b1 = artifact["w1"], artifact["b1"]
        self.w2, self.b2 = artifact["w2"], artifact["b2"]

    def _lookup(self, accounts):
        ids = np.array(
            [self.account_to_id.get(a, -1) for a in accounts],
            dtype=np.int64,
        )
        vectors = self.embeddings[np.maximum(ids, 0)].copy()
        vectors[ids < 0] = 0.0
        return vectors

    def score(self, df):
        """Fraud probability for each row of df (nameOrig, nameDest and
        the six raw numeric fields)."""
        edge_features = self.scaler.transform(prepare_gnn_edge_features(df))

        combined = np.hstack([
            self._lookup(df["nameOrig"].astype(str)),
            self._lookup(df["nameDest"].astype(str)),
            edge_features,
        ]).astype(np.float32)

        hidden = np.maximum(combined @ self.w1.T + self.b1, 0.0)
        logits = (hidden @ self.w2.T + self.b2).ravel()
        return 1.0 / (1.0 + np.exp(-logits))


def account_known(scorer, accounts):
    return pd.Series(accounts).astype(str).map(
        lambda a: a in scorer.account_to_id
    ).to_numpy()
