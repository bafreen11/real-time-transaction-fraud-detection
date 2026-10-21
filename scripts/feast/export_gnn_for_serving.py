import joblib
import numpy as np
import pandas as pd
import torch

from fraud_detection.config import (
    GNN_SERVING_ARTIFACT_PATH,
    SHARED_GNN_MODEL_PATH,
    SHARED_GNN_PREDICTIONS_PATH,
    SHARED_GRAPH_TRANSACTIONS_PATH,
    SHARED_TEST_PATH,
    ensure_output_dirs,
)
from fraud_detection.models.gnn import FraudGNN
from fraud_detection.models.gnn_serving import GNNServingScorer

CHECK_ROWS = 20_000

# --------------------------------------------------
# 1. Load the trained shared-track GNN
# --------------------------------------------------

checkpoint = torch.load(SHARED_GNN_MODEL_PATH, weights_only=False)
account_to_id = checkpoint["account_to_id"]
edge_feature_columns = checkpoint["edge_feature_columns"]

model = FraudGNN(
    2, checkpoint["hidden_channels"], len(edge_feature_columns)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --------------------------------------------------
# 2. Rebuild the training graph exactly as train_shared_gnn.py does
# --------------------------------------------------

train_df = pd.read_csv(
    SHARED_GRAPH_TRANSACTIONS_PATH, usecols=["nameOrig", "nameDest"]
)
num_nodes = len(account_to_id)

source_ids = train_df["nameOrig"].map(account_to_id).to_numpy()
target_ids = train_df["nameDest"].map(account_to_id).to_numpy()

counts = np.column_stack([
    np.bincount(source_ids, minlength=num_nodes),
    np.bincount(target_ids, minlength=num_nodes),
])
node_features = torch.tensor(np.log1p(counts), dtype=torch.float32)

edge_index = torch.tensor(
    np.vstack([source_ids, target_ids]), dtype=torch.long
)
message_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

with torch.no_grad():
    embeddings = model.encode(node_features, message_edge_index).numpy()

print("Account embeddings:", embeddings.shape)

# --------------------------------------------------
# 3. Export embeddings, scaler and classifier weights
# --------------------------------------------------

first_layer = model.classifier[0]
last_layer = model.classifier[3]

artifact = {
    "embeddings": embeddings.astype(np.float32),
    "account_to_id": account_to_id,
    "scaler": checkpoint["scaler"],
    "edge_feature_columns": edge_feature_columns,
    "w1": first_layer.weight.detach().numpy(),
    "b1": first_layer.bias.detach().numpy(),
    "w2": last_layer.weight.detach().numpy(),
    "b2": last_layer.bias.detach().numpy(),
}

# --------------------------------------------------
# 4. Check against the saved GNN test predictions
# --------------------------------------------------

scorer = GNNServingScorer(artifact)
test_rows = pd.read_csv(SHARED_TEST_PATH, nrows=CHECK_ROWS)
saved = pd.read_csv(SHARED_GNN_PREDICTIONS_PATH, nrows=CHECK_ROWS)

max_difference = float(np.abs(
    scorer.score(test_rows) - saved["gnn_fraud_probability"].to_numpy()
).max())
print(f"Max difference vs saved predictions ({CHECK_ROWS} rows):",
      max_difference)

if max_difference > 1e-4:
    raise ValueError("Exported GNN does not reproduce saved predictions.")

ensure_output_dirs()
joblib.dump(artifact, GNN_SERVING_ARTIFACT_PATH)
print("Saved GNN serving artifact:", GNN_SERVING_ARTIFACT_PATH)
