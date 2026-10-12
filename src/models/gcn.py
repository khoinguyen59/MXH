"""Graph Convolutional Network (GCN) Architecture.

Implements the standard spectral GCN (Kipf & Welling, 2017) with ReLU activation
and dropout regularization.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.models.base_gnn import BaseGNN


class GCN(BaseGNN):
    """2-layer or 3-layer Graph Convolutional Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Initialize GCN model.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden representation dimension.
            out_channels: Target class dimension.
            num_layers: Number of GCN convolutional layers (default: 2).
            dropout: Dropout probability (default: 0.5).
        """
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        if num_layers == 2:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        elif num_layers >= 3:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass computing log-softmax class probabilities."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Extract latent representation before the final classification projection."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        return x
