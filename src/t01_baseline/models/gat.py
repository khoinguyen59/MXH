"""Graph Attention Network (GAT) Architecture.

Implements multi-head masked self-attention over node neighborhoods (Veličković et al., ICLR 2018).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.base_gnn import BaseGNN


class GAT(BaseGNN):
    """Multi-Head Graph Attention Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """Initialize GAT model.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden representation dimension per head.
            out_channels: Target class dimension.
            heads: Number of attention heads (default: 4).
            num_layers: Number of GAT layers (default: 2).
            dropout: Dropout probability on features and attention weights (default: 0.5).
        """
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels * heads,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.heads = heads
        self.num_layers = num_layers
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass computing log-softmax class probabilities."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Extract latent representation before final attention projection."""
        x = self.conv1(x, edge_index)
        return F.elu(x)
