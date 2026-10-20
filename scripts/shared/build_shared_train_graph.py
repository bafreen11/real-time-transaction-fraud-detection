import pandas as pd
import torch
from torch_geometric.data import Data

from fraud_detection.config import (
    SHARED_GRAPH_PATH as GRAPH_PATH,
    SHARED_GRAPH_TRANSACTIONS_PATH as TRANSACTIONS_PATH,
    SHARED_TRAIN_PATH as TRAIN_PATH,
)

print("Loading shared training transactions...")
df = pd.read_csv(TRAIN_PATH)

print("Training rows:", len(df))
print("Training fraud rows:", int(df["isFraud"].sum()))

# Build account IDs using training transactions only.
all_accounts = pd.concat(
    [df["nameOrig"], df["nameDest"]],
    ignore_index=True,
).unique()

account_to_id = {
    account: idx for idx, account in enumerate(all_accounts)
}

df["source_id"] = df["nameOrig"].map(account_to_id)
df["target_id"] = df["nameDest"].map(account_to_id)

# Build transaction edges.
edge_index = torch.tensor(
    df[["source_id", "target_id"]].to_numpy().T,
    dtype=torch.long,
)

num_nodes = len(all_accounts)

sender_counts = (
    df["source_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

recipient_counts = (
    df["target_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

node_features = torch.tensor(
    pd.DataFrame({
        "sender_count": sender_counts.to_numpy(),
        "recipient_count": recipient_counts.to_numpy(),
    }).to_numpy(),
    dtype=torch.float32,
)

node_features = torch.log1p(node_features)

edge_labels = torch.tensor(
    df["isFraud"].to_numpy(),
    dtype=torch.long,
)

graph = Data(
    x=node_features,
    edge_index=edge_index,
    edge_label=edge_labels,
)

torch.save(graph, GRAPH_PATH)
df.to_csv(TRANSACTIONS_PATH, index=False)

print("\nShared training graph created.")
print("Account nodes:", graph.num_nodes)
print("Training edges:", graph.num_edges)
print("Node features:", tuple(graph.x.shape))
print("Saved graph:", GRAPH_PATH)
print("Saved transactions:", TRANSACTIONS_PATH)
