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
    SHARED_GNN_STRATEGY_COMPARISON_PATH as STRATEGY_COMPARISON_PATH,
    SHARED_GNN_VALIDATION_PREDICTIONS_PATH as VALIDATION_PREDICTIONS_PATH,
    SHARED_GRAPH_TRANSACTIONS_PATH as TRAIN_PATH,
    SHARED_TEST_PATH as TEST_PATH,
    ensure_output_dirs,
)
from fraud_detection.data.splits import (
    shared_validation_split,
    shared_validation_weights,
)
from fraud_detection.models.gnn import (
    FraudGNN,
    account_count_features,
    message_edge_index,
)
from fraud_detection.models.gnn_serving import (
    GNN_EDGE_FEATURE_COLUMNS,
    prepare_gnn_edge_features,
)

# --------------------------------------------------
# Edge rules for message passing
# --------------------------------------------------
# 1. A transaction being scored is never in the graph used to compute
#    its accounts' embeddings. With the "disjoint" strategy, each epoch
#    splits the fit transactions at random into message edges
#    (MESSAGE_EDGE_FRACTION) and supervision edges; the loss uses only
#    supervision edges.
# 2. Node features (log sender/recipient counts) are counted over the
#    message edges only.
# 3. Accounts with no message edge get zero embeddings, as unseen
#    accounts do at test time.
# 4. Validation and test transactions never enter message passing
#    during training. Validation is scored on the graph of fit edges,
#    test on the graph of all training edges. Labels are never features.
#
# The "transductive" strategy (the original approach: every fit edge is
# both a message and a supervision edge) is trained for comparison. The
# strategy with the higher validation average precision is kept.
# Validation metrics are weighted to the natural fraud rate
# (splits.shared_validation_weights); shared_train.csv is ~6% fraud.

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

EPOCHS = 15
LEARNING_RATE = 0.003
HIDDEN_CHANNELS = 32
MESSAGE_EDGE_FRACTION = 0.7
STRATEGIES = ["disjoint", "transductive"]

device = torch.device("cpu")

print("Training device:", device)

# --------------------------------------------------
# 2. Load training and test transactions
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["isFraud"] = train_df["isFraud"].astype(int)
test_df["isFraud"] = test_df["isFraud"].astype(int)

# Same validation rows as train_shared_xgboost.py
fit_rows, val_rows = shared_validation_split(train_df["isFraud"])

print("Fit transactions:", len(fit_rows),
      "| fraud:", int(train_df["isFraud"].iloc[fit_rows].sum()))
print("Validation transactions:", len(val_rows),
      "| fraud:", int(train_df["isFraud"].iloc[val_rows].sum()))
print("Test transactions:", len(test_df),
      "| fraud:", int(test_df["isFraud"].sum()))

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

train_sources = train_df["nameOrig"].map(account_to_id).to_numpy()
train_targets = train_df["nameDest"].map(account_to_id).to_numpy()

# Test accounts absent from training are mapped to -1.
# These accounts receive zero node embeddings at prediction time.
test_sources = (
    test_df["nameOrig"].map(account_to_id).fillna(-1).astype(int).to_numpy()
)
test_targets = (
    test_df["nameDest"].map(account_to_id).fillna(-1).astype(int).to_numpy()
)

# --------------------------------------------------
# 4. Prepare transaction (edge) features
# --------------------------------------------------

edge_feature_columns = GNN_EDGE_FEATURE_COLUMNS

# The scaler is fitted on fit transactions only.
scaler = StandardScaler()
scaler.fit(prepare_gnn_edge_features(train_df.iloc[fit_rows]))

train_features = torch.tensor(
    scaler.transform(prepare_gnn_edge_features(train_df)),
    dtype=torch.float32, device=device,
)
test_features = torch.tensor(
    scaler.transform(prepare_gnn_edge_features(test_df)),
    dtype=torch.float32, device=device,
)

y_train = torch.tensor(
    train_df["isFraud"].to_numpy(), dtype=torch.float32, device=device
)

positive_count = float(y_train[fit_rows].sum())
negative_count = float(len(fit_rows) - positive_count)

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1)],
    dtype=torch.float32, device=device,
)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print("Account nodes:", num_nodes)
print("Positive-class weight:", pos_weight.item())

# --------------------------------------------------
# 5. Training and scoring
# --------------------------------------------------


def graph_from(rows):
    """Node features, message edges and known-node mask built from the
    given training rows only."""
    node_features = account_count_features(
        train_sources[rows], train_targets[rows], num_nodes
    ).to(device)
    edges = message_edge_index(
        train_sources[rows], train_targets[rows]
    ).to(device)
    known_nodes = node_features.sum(dim=1) > 0
    return node_features, edges, known_nodes


def train_gnn(strategy):
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    model = FraudGNN(
        2, HIDDEN_CHANNELS, len(edge_feature_columns)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining strategy: {strategy}")

    for epoch in range(1, EPOCHS + 1):
        if strategy == "disjoint":
            shuffled = rng.permutation(fit_rows)
            message_count = int(len(shuffled) * MESSAGE_EDGE_FRACTION)
            message_rows = shuffled[:message_count]
            supervision_rows = shuffled[message_count:]
        else:
            message_rows = fit_rows
            supervision_rows = fit_rows

        node_features, edges, known_nodes = graph_from(message_rows)
        supervision = torch.tensor(supervision_rows, device=device)

        model.train()
        optimizer.zero_grad()

        embeddings = model.encode(node_features, edges)
        logits = model.predict_edges(
            embeddings,
            torch.tensor(train_sources[supervision_rows], device=device),
            torch.tensor(train_targets[supervision_rows], device=device),
            train_features[supervision],
            known_nodes,
        )

        loss = loss_function(logits, y_train[supervision])
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| Training loss: {loss.item():.4f}"
        )

    return model


def score(model, message_rows, sources, targets, edge_features):
    """Fraud probabilities for edges that are not in message_rows."""
    node_features, edges, known_nodes = graph_from(message_rows)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(node_features, edges)
        logits = model.predict_edges(
            embeddings,
            torch.tensor(sources, device=device),
            torch.tensor(targets, device=device),
            edge_features,
            known_nodes,
        )
    return torch.sigmoid(logits).cpu().numpy()


def metrics(y_true, probabilities, threshold=0.5, sample_weight=None):
    predictions = (probabilities >= threshold).astype(int)
    weights = {"sample_weight": sample_weight}
    return {
        "average_precision": average_precision_score(
            y_true, probabilities, **weights
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0, **weights
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0, **weights
        ),
        "f1": f1_score(y_true, predictions, zero_division=0, **weights),
    }


# --------------------------------------------------
# 6. Compare strategies on the validation set
# --------------------------------------------------

y_val = train_df["isFraud"].to_numpy()[val_rows]
val_weights = shared_validation_weights(y_val)
val_features = train_features[torch.tensor(val_rows, device=device)]

models = {}
val_probabilities = {}
comparison = []

for strategy in STRATEGIES:
    models[strategy] = train_gnn(strategy)
    val_probabilities[strategy] = score(
        models[strategy],
        fit_rows,
        train_sources[val_rows],
        train_targets[val_rows],
        val_features,
    )
    comparison.append({
        "strategy": strategy,
        **{
            f"val_{name}": value
            for name, value in metrics(
                y_val, val_probabilities[strategy],
                sample_weight=val_weights,
            ).items()
        },
    })

comparison_df = pd.DataFrame(comparison)

print("\n========== GNN STRATEGY COMPARISON "
      "(VALIDATION, WEIGHTED TO THE NATURAL FRAUD RATE) ==========")
print(comparison_df.to_string(
    index=False, float_format=lambda value: f"{value:.6f}"
))

selected = comparison_df.loc[
    comparison_df["val_average_precision"].idxmax(), "strategy"
]
comparison_df["selected"] = comparison_df["strategy"] == selected
print("Selected strategy (highest validation AP):", selected)

model = models[selected]

# --------------------------------------------------
# 7. Predict held-out test transactions
# --------------------------------------------------

# All training transactions form the graph; test edges are never in it.
test_probabilities = score(
    model,
    np.arange(len(train_df)),
    test_sources,
    test_targets,
    test_features,
)

y_test = test_df["isFraud"].to_numpy()
y_pred = (test_probabilities >= 0.5).astype(int)

print("\n========== SHARED TEST RESULTS (selected strategy) ==========")
for name, value in metrics(y_test, test_probabilities).items():
    print(f"{name}: {value:.6f}")
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
        "training_strategy": selected,
    },
    MODEL_PATH,
)

comparison_df.to_csv(STRATEGY_COMPARISON_PATH, index=False)

# transaction_index = row position in shared_train.csv
pd.DataFrame({
    "transaction_index": val_rows,
    "isFraud": y_val,
    "gnn_fraud_probability": val_probabilities[selected],
}).to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

# Preserve the original test row position so the XGBoost
# and GNN outputs can be aligned exactly.
pd.DataFrame({
    "transaction_index": np.arange(len(test_df)),
    "isFraud": y_test,
    "gnn_fraud_probability": test_probabilities,
}).to_csv(PREDICTIONS_PATH, index=False)

print("\nSaved GNN model:", MODEL_PATH)
print("Saved strategy comparison:", STRATEGY_COMPARISON_PATH)
print("Saved GNN validation predictions:", VALIDATION_PREDICTIONS_PATH)
print("Saved GNN test predictions:", PREDICTIONS_PATH)
print("Shared GNN training and evaluation finished.")
