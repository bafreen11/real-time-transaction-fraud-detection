
import pandas as pd
import torch
from torch_geometric.data import Data

# Step 1: Load the PaySim dataset
file_path = r"E:\paysim.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Step 2: Select a manageable sample for our first graph
# Keep all fraud rows and sample legitimate rows.
fraud_df = df[df["isFraud"] == 1]
legitimate_df = df[df["isFraud"] == 0]

sample_size = min(100_000, len(legitimate_df))

legitimate_sample = legitimate_df.sample(
    n=sample_size,
    random_state=42
)

graph_df = pd.concat(
    [fraud_df, legitimate_sample],
    ignore_index=True
)

print("Graph transaction rows:", len(graph_df))
print("Fraud transactions:", int(graph_df["isFraud"].sum()))

# Step 3: Convert account IDs into numeric node IDs
all_accounts = pd.concat(
    [graph_df["nameOrig"], graph_df["nameDest"]],
    ignore_index=True
).unique()

account_to_id = {
    account: idx for idx, account in enumerate(all_accounts)
}

graph_df["source_id"] = graph_df["nameOrig"].map(account_to_id)
graph_df["target_id"] = graph_df["nameDest"].map(account_to_id)

# Step 4: Create graph edges
edge_index = torch.tensor(
    graph_df[["source_id", "target_id"]].to_numpy().T,
    dtype=torch.long
)

# Step 5: Create simple account-level node features
# Count how often each account appears as a sender or recipient.
num_nodes = len(all_accounts)

sender_counts = (
    graph_df["source_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

recipient_counts = (
    graph_df["target_id"]
    .value_counts()
    .reindex(range(num_nodes), fill_value=0)
    .sort_index()
)

node_features = torch.tensor(
    pd.DataFrame({
        "sender_count": sender_counts.to_numpy(),
        "recipient_count": recipient_counts.to_numpy()
    }).to_numpy(),
    dtype=torch.float
)

# Step 6: Store transaction labels on the edges
edge_labels = torch.tensor(
    graph_df["isFraud"].to_numpy(),
    dtype=torch.long
)

# Step 7: Create the PyTorch Geometric graph object
data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_label=edge_labels
)

# Step 8: Save graph data and transaction information
output_path = r"E:\Transactions Fraud Detection\transaction_graph.pt"
torch.save(data, output_path)

graph_df.to_csv(
    r"E:\Transactions Fraud Detection\graph_transactions.csv",
    index=False
)

print("\nGraph created successfully!")
print("Number of account nodes:", data.num_nodes)
print("Number of transaction edges:", data.num_edges)
print("Node feature shape:", tuple(data.x.shape))
print("Edge index shape:", tuple(data.edge_index.shape))
print("Edge label shape:", tuple(data.edge_label.shape))
print("Saved graph to:", output_path)
