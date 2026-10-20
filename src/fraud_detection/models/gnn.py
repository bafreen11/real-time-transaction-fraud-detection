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
        self, embeddings, source_ids, target_ids, edge_features
    ):
        source_embedding = embeddings[source_ids]
        target_embedding = embeddings[target_ids]

        combined = torch.cat(
            [source_embedding, target_embedding, edge_features],
            dim=1,
        )

        return self.classifier(combined).squeeze(1)
