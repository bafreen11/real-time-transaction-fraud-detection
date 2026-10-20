import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = r"E:\Transactions Fraud Detection"
DATA_PATH = PROJECT_DIR + r"\graph_transactions.csv"

RANDOM_STATE = 42
EPOCHS = 15
LEARNING_RATE = 0.003
HIDDEN_CHANNELS = 32

torch.manual_seed(RANDOM_STATE)

device = torch.device("cpu")
print("Training device:", device)

# --------------------------------------------------
# 2. Load graph transaction data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    "source_id",
    "target_id",
    "isFraud",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df["isFraud"] = df["isFraud"].astype("int64")

print("Transactions loaded:", len(df))
print("Fraud transactions:", int(df["isFraud"].sum()))

# --------------------------------------------------
# 3. Split transactions before creating graph features
# --------------------------------------------------

train_idx, test_idx = train_test_split(
    df.index.to_numpy(),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["isFraud"],
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

print("Training transactions:", len(train_df))
print("Testing transactions:", len(test_df))
print("Training fraud count:", int(train_df["isFraud"].sum()))
print("Testing fraud count:", int(test_df["isFraud"].sum()))

# --------------------------------------------------
# 4. Build account features using TRAINING transactions only
# --------------------------------------------------
edge_feature_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

train_edge_features = train_df[edge_feature_columns].copy()
test_edge_features = test_df[edge_feature_columns].copy()

# Reduce skew in transaction amounts and balances.
log_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

for col in log_columns:
    train_edge_features[col] = np.log1p(
        train_edge_features[col].clip(lower=0)
    )
    test_edge_features[col] = np.log1p(
        test_edge_features[col].clip(lower=0)
    )

# Fit only on training data to prevent test-data leakage.
scaler = StandardScaler()
train_edge_features = scaler.fit_transform(train_edge_features)
test_edge_features = scaler.transform(test_edge_features)

train_edge_features = torch.tensor(
    train_edge_features, dtype=torch.float32
)
test_edge_features = torch.tensor(
    test_edge_features, dtype=torch.float32
)
num_nodes = int(
    max(df["source_id"].max(), df["target_id"].max())
) + 1

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

# Log transform reduces the impact of accounts with many transactions.
node_features = torch.tensor(
    pd.DataFrame({
        "sender_count": sender_counts.to_numpy(),
        "recipient_count": recipient_counts.to_numpy(),
    }).to_numpy(),
    dtype=torch.float32,
)

node_features = torch.log1p(node_features)

# --------------------------------------------------
# 5. Build message-passing graph from TRAINING edges
# --------------------------------------------------

train_edge_index = torch.tensor(
    train_df[["source_id", "target_id"]].to_numpy().T,
    dtype=torch.long,
)

# Make message passing bidirectional.
# Labels are NOT used as model input features.
message_edge_index = torch.cat(
    [train_edge_index, train_edge_index.flip(0)],
    dim=1,
)

x = node_features.to(device)
edge_index = message_edge_index.to(device)

# --------------------------------------------------
# 6. Define GraphSAGE edge-classification model
# --------------------------------------------------

class FraudGNN(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(
            input_channels, hidden_channels
        )
        self.conv2 = SAGEConv(
            hidden_channels, hidden_channels
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 6, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return x

    def predict_edges(self, embeddings, source_ids, target_ids, edge_features):
        source_embedding = embeddings[source_ids]
        target_embedding = embeddings[target_ids]

        edge_features = torch.cat(
            [source_embedding, target_embedding, edge_features,],
            dim=1,
        )

        return self.classifier(edge_features).squeeze(1)


model = FraudGNN(
    input_channels=2,
    hidden_channels=HIDDEN_CHANNELS,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# 7. Prepare transaction labels and IDs
# --------------------------------------------------

train_sources = torch.tensor(
    train_df["source_id"].to_numpy(),
    dtype=torch.long,
    device=device,
)

train_targets = torch.tensor(
    train_df["target_id"].to_numpy(),
    dtype=torch.long,
    device=device,
)

y_train = torch.tensor(
    train_df["isFraud"].to_numpy(),
    dtype=torch.float32,
    device=device,
)

test_sources = torch.tensor(
    test_df["source_id"].to_numpy(),
    dtype=torch.long,
    device=device,
)

test_targets = torch.tensor(
    test_df["target_id"].to_numpy(),
    dtype=torch.long,
    device=device,
)

y_test = test_df["isFraud"].to_numpy()

# Compensate for fraud being the minority class.
positive_count = float(y_train.sum())
negative_count = float(len(y_train) - positive_count)

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1)],
    dtype=torch.float32,
    device=device,
)

loss_function = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

print("Positive-class weight:", pos_weight.item())

# --------------------------------------------------
# 8. Train the GNN
# --------------------------------------------------

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    embeddings = model.encode(x, edge_index)

    train_logits = model.predict_edges(
        embeddings,
        train_sources,
        train_targets,
        train_edge_features,
    )

    loss = loss_function(train_logits, y_train)

    loss.backward()
    optimizer.step()

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Training loss: {loss.item():.4f}"
    )

# --------------------------------------------------
# 9. Evaluate on held-out transactions
# --------------------------------------------------

model.eval()

with torch.no_grad():
    embeddings = model.encode(x, edge_index)

    test_logits = model.predict_edges(
        embeddings,
        test_sources,
        test_targets,
        test_edge_features,
    )

    test_probabilities = torch.sigmoid(test_logits).cpu().numpy()

# Use 0.5 as an initial threshold.
# Threshold tuning should be performed using validation data.
test_predictions = (test_probabilities >= 0.5).astype(int)

average_precision = average_precision_score(
    y_test, test_probabilities
)

precision = precision_score(
    y_test, test_predictions, zero_division=0
)

recall = recall_score(
    y_test, test_predictions, zero_division=0
)

f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

print("\n========== GNN TEST RESULTS ==========")
print(f"Average precision (PR-AUC): {average_precision:.6f}")
print(f"Fraud precision:             {precision:.6f}")
print(f"Fraud recall:                {recall:.6f}")
print(f"Fraud F1-score:              {f1:.6f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

# --------------------------------------------------
# 10. Save model and test probabilities
# --------------------------------------------------

model_path = PROJECT_DIR + r"\fraud_gnn_model.pt"
predictions_path = PROJECT_DIR + r"\gnn_test_predictions.csv"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_channels": 2,
        "hidden_channels": HIDDEN_CHANNELS,
        "num_nodes": num_nodes,
    },
    model_path,
)

pd.DataFrame({
    "transaction_index": test_df.index.to_numpy(),
    "isFraud": y_test,
    "gnn_fraud_probability": test_probabilities,
}).to_csv(predictions_path, index=False)

print("\nGNN model saved to:", model_path)
print("Test predictions saved to:", predictions_path)
print("GNN training and evaluation finished.")
