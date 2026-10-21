import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv


class FraudGNN(nn.Module):
    """GraphSAGE encoder over accounts with an edge (transaction) classifier.

    Each transaction is scored from its sender and recipient embeddings
    concatenated with its own edge features.
    """

    def __init__(self, input_channels, hidden_channels, edge_feature_count=6):
        super().__init__()

        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_channels * 2 + edge_feature_count, hidden_channels
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def predict_edges(
        self, embeddings, source_ids, target_ids, edge_features,
        known_nodes=None,
    ):
        """Edge logits from sender/recipient embeddings and edge features.

        known_nodes is an optional boolean tensor over all nodes. Accounts
        that are not known, and ids of -1, get zero embeddings: the same
        treatment unseen accounts get at test time.
        """
        source_embedding = embeddings[source_ids.clamp(min=0)]
        target_embedding = embeddings[target_ids.clamp(min=0)]

        if known_nodes is not None:
            source_embedding = source_embedding * _known_mask(
                source_ids, known_nodes
            )
            target_embedding = target_embedding * _known_mask(
                target_ids, known_nodes
            )

        combined = torch.cat(
            [source_embedding, target_embedding, edge_features],
            dim=1,
        )

        return self.classifier(combined).squeeze(1)


def _known_mask(ids, known_nodes):
    known = (ids >= 0) & known_nodes[ids.clamp(min=0)]
    return known.unsqueeze(1).to(torch.float32)


def account_count_features(source_ids, target_ids, num_nodes):
    """Node features: log1p of each account's sender and recipient counts
    over the given edges, shape (num_nodes, 2)."""
    counts = np.column_stack([
        np.bincount(source_ids, minlength=num_nodes),
        np.bincount(target_ids, minlength=num_nodes),
    ])
    return torch.tensor(np.log1p(counts), dtype=torch.float32)


def message_edge_index(source_ids, target_ids):
    """Undirected edge_index (both directions) for message passing."""
    edge_index = torch.tensor(
        np.vstack([source_ids, target_ids]), dtype=torch.long
    )
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)
