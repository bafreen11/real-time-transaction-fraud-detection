"""Blended XGBoost + GNN scoring for the streaming track."""

import joblib

from fraud_detection.models.gnn_serving import GNNServingScorer


def blend_scores(xgboost_probability, gnn_probability, gnn_weight):
    return (1 - gnn_weight) * xgboost_probability + gnn_weight * gnn_probability


class StreamScorer:
    """Loads the stream model artifact and the exported GNN.

    The artifact (scripts/feast/train_stream_model.py) holds the model,
    feature_columns, threshold, gnn_weight and version.
    """

    def __init__(self, model_path, gnn_artifact_path):
        self.model_path = model_path
        self.gnn = GNNServingScorer(joblib.load(gnn_artifact_path))
        self.reload()

    def reload(self):
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.threshold = float(artifact["threshold"])
        self.gnn_weight = float(artifact["gnn_weight"])
        self.version = artifact.get("version", "unknown")

    def score(self, transactions, features):
        """Score rows. transactions holds nameOrig/nameDest and the raw
        fields (for the GNN); features is build_stream_features() output.
        Returns one result dict per row.
        """
        xgb = self.model.predict_proba(features[self.feature_columns])[:, 1]
        gnn = self.gnn.score(transactions)
        blended = blend_scores(xgb, gnn, self.gnn_weight)

        results = []
        for x, g, b in zip(xgb, gnn, blended):
            is_fraud = bool(b >= self.threshold)
            results.append({
                "xgboost_probability": round(float(x), 6),
                "gnn_probability": round(float(g), 6),
                "fraud_score": round(float(b), 6),
                "gnn_weight": self.gnn_weight,
                "threshold": self.threshold,
                "predicted_label": int(is_fraud),
                "decision": "FRAUD_ALERT" if is_fraud else "LEGITIMATE",
                "model_version": self.version,
            })
        return results
