"""Universal T01 Model Adapter for Group 09 GNN Models.

Provides a defensive wrapper around GCN models/checkpoints loaded from
Nhóm 1's T01 handover package (`data/t01_handover/`), adding dynamic `edge_weight`
support, `cached=False`, and configurable `return_type` ('log_probs', 'probs', 'raw')
for seamless compatibility with PyG Explainers (GNNExplainer, PGExplainer, Saliency).
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class T01GCNModelWrapper(torch.nn.Module):
    """PyG-compliant defensive wrapper around T01 GCN architectures."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # cached=False ensures dynamic edge weight updates during explanation optimization
        self.c1 = GCNConv(in_dim, hidden, cached=False)
        self.c2 = GCNConv(hidden, n_classes, cached=False)
        self.p = dropout
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_classes = n_classes

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_type: str = "log_probs",
    ) -> torch.Tensor:
        """Forward pass supporting continuous edge attribution weights and distribution types."""
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.p, self.training)
        x = self.c2(x, edge_index, edge_weight=edge_weight)

        if return_type in ("raw", "logits"):
            return x
        elif return_type in ("probs", "prob"):
            return F.softmax(x, dim=-1)
        elif return_type in ("log_probs", "log_prob"):
            return F.log_softmax(x, dim=-1)
        else:
            raise ValueError(
                f"Unsupported return_type: '{return_type}'. "
                "Must be 'raw', 'logits', 'probs', or 'log_probs'."
            )
