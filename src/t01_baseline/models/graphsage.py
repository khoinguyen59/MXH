"""GraphSAGE Architecture.

Implements inductive representation learning on graphs (Hamilton et al., NeurIPS 2017)
utilizing neighborhood aggregation with Mean pooling.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.models.base_gnn import BaseGNN


class GraphSAGE(BaseGNN):
    """Inductive GraphSAGE network with Mean aggregator."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        """Initialize GraphSAGE model.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden representation dimension.
            out_channels: Target class dimension.
            num_layers: Number of SAGE layers (default: 2).
            dropout: Dropout probability (default: 0.5).
            aggr: Neighborhood aggregation scheme ('mean', 'max', 'lstm').
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
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        elif num_layers >= 3:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass computing log-softmax class probabilities."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)

    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Extract latent representation before final classification projection."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        return x
