import numpy as np
import pandas as pd
import torch

from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from fraud_detection.config import (
    RANDOM_STATE,
    SHARED_GNN_MODEL_PATH as MODEL_PATH,
    SHARED_GNN_PREDICTIONS_PATH as PREDICTIONS_PATH,
    SHARED_GRAPH_TRANSACTIONS_PATH as TRAIN_PATH,
    SHARED_TEST_PATH as TEST_PATH,
    ensure_output_dirs,
)
from fraud_detection.models.gnn import FraudGNN

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EPOCHS = 15
LEARNING_RATE = 0.003
HIDDEN_CHANNELS = 32

torch.manual_seed(RANDOM_STATE)
device = torch.device("cpu")

print("Training device:", device)

# --------------------------------------------------
# 2. Load training and test transactions
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["isFraud"] = train_df["isFraud"].astype(int)
test_df["isFraud"] = test_df["isFraud"].astype(int)

print("Training transactions:", len(train_df))
print("Test transactions:", len(test_df))
print("Training fraud:", int(train_df["isFraud"].sum()))
print("Test fraud:", int(test_df["isFraud"].sum()))

# --------------------------------------------------
# 3. Build account IDs using training data only
# --------------------------------------------------

all_accounts = pd.concat(
    [train_df["nameOrig"], train_df["nameDest"]],
    ignore_index=True,
).unique()

account_to_id = {
    account: idx for idx, account in enumerate(all_accounts)
}

num_nodes = len(all_accounts)

train_df["source_id"] = train_df["nameOrig"].map(account_to_id)
train_df["target_id"] = train_df["nameDest"].map(account_to_id)

# Test accounts absent from training are mapped to -1.
# These accounts receive zero node embeddings at prediction time.
test_df["source_id"] = (
    test_df["nameOrig"].map(account_to_id).fillna(-1).astype(int)
)
test_df["target_id"] = (
    test_df["nameDest"].map(account_to_id).fillna(-1).astype(int)
)

# --------------------------------------------------
# 4. Build node features from training transactions only
# --------------------------------------------------

sender_counts = (
    train_df["source_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

recipient_counts = (
    train_df["target_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

node_features = torch.tensor(
    np.log1p(
        np.column_stack([
            sender_counts.to_numpy(),
            recipient_counts.to_numpy(),
        ])
    ),
    dtype=torch.float32,
    device=device,
)

# --------------------------------------------------
# 5. Prepare transaction features
# --------------------------------------------------

edge_feature_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

log_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

def prepare_edge_features(df):
    features = df[edge_feature_columns].copy()

    for col in log_columns:
        features[col] = np.log1p(
            features[col].clip(lower=0)
        )

    return features

train_raw_features = prepare_edge_features(train_df)
test_raw_features = prepare_edge_features(test_df)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_raw_features)
test_features = scaler.transform(test_raw_features)

train_features = torch.tensor(
    train_features, dtype=torch.float32, device=device
)
test_features = torch.tensor(
    test_features, dtype=torch.float32, device=device
)

# --------------------------------------------------
# 6. Build message-passing graph from training edges
# --------------------------------------------------

train_edge_index = torch.tensor(
    train_df[["source_id", "target_id"]].to_numpy().T,
    dtype=torch.long,
    device=device,
)

message_edge_index = torch.cat(
    [train_edge_index, train_edge_index.flip(0)],
    dim=1,
)

# --------------------------------------------------
# 7. Create GraphSAGE edge classifier
# --------------------------------------------------

model = FraudGNN(
    2, HIDDEN_CHANNELS, len(edge_feature_columns)
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

train_sources = torch.tensor(
    train_df["source_id"].to_numpy(),
    dtype=torch.long, device=device,
)
train_targets = torch.tensor(
    train_df["target_id"].to_numpy(),
    dtype=torch.long, device=device,
)
y_train = torch.tensor(
    train_df["isFraud"].to_numpy(),
    dtype=torch.float32, device=device,
)

positive_count = float(y_train.sum())
negative_count = float(len(y_train) - positive_count)

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1)],
    dtype=torch.float32, device=device,
)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Account nodes:", num_nodes)
print("Positive-class weight:", pos_weight.item())

# --------------------------------------------------
# 8. Train
# --------------------------------------------------

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    embeddings = model.encode(
        node_features, message_edge_index
    )

    logits = model.predict_edges(
        embeddings,
        train_sources,
        train_targets,
        train_features,
    )

    loss = loss_function(logits, y_train)
    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Training loss: {loss.item():.4f}"
    )

# --------------------------------------------------
# 9. Predict held-out test transactions
# --------------------------------------------------

# Map unknown accounts to a safe index temporarily.
# Their embeddings are replaced with zeros below.
test_sources_np = test_df["source_id"].to_numpy()
test_targets_np = test_df["target_id"].to_numpy()

known_source = test_sources_np >= 0
known_target = test_targets_np >= 0

safe_sources = torch.tensor(
    np.maximum(test_sources_np, 0),
    dtype=torch.long, device=device,
)
safe_targets = torch.tensor(
    np.maximum(test_targets_np, 0),
    dtype=torch.long, device=device,
)

model.eval()

with torch.no_grad():
    embeddings = model.encode(
        node_features, message_edge_index
    )

    source_embeddings = embeddings[safe_sources].clone()
    target_embeddings = embeddings[safe_targets].clone()

    source_embeddings[~torch.tensor(known_source, device=device)] = 0
    target_embeddings[~torch.tensor(known_target, device=device)] = 0

    combined = torch.cat(
        [source_embeddings, target_embeddings, test_features],
        dim=1,
    )

    test_logits = model.classifier(combined).squeeze(1)
    test_probabilities = torch.sigmoid(test_logits).cpu().numpy()

y_test = test_df["isFraud"].to_numpy()
y_pred = (test_probabilities >= 0.5).astype(int)

print("\n========== SHARED TEST RESULTS ==========")
print(
    "Average precision:",
    average_precision_score(y_test, test_probabilities),
)
print(
    "Fraud precision:",
    precision_score(y_test, y_pred, zero_division=0),
)
print(
    "Fraud recall:",
    recall_score(y_test, y_pred, zero_division=0),
)
print(
    "Fraud F1:",
    f1_score(y_test, y_pred, zero_division=0),
)
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and prediction outputs.
ensure_output_dirs()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "account_to_id": account_to_id,
        "scaler": scaler,
        "edge_feature_columns": edge_feature_columns,
        "hidden_channels": HIDDEN_CHANNELS,
    },
    MODEL_PATH,
)

# Preserve the original test row position so the XGBoost
# and GNN outputs can be aligned exactly.
test_df.reset_index(drop=True, inplace=True)

pd.DataFrame({
    "transaction_index": np.arange(len(test_df)),
    "isFraud": y_test,
    "gnn_fraud_probability": test_probabilities,
}).to_csv(PREDICTIONS_PATH, index=False)

print("\nSaved GNN model:", MODEL_PATH)
print("Saved GNN predictions:", PREDICTIONS_PATH)
print("Shared GNN training and evaluation finished.")
