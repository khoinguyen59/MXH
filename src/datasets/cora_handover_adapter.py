"""Adapter Service for Cora Handover Artifacts.

Provides non-invasive ingestion and in-memory defensive wrapping of the Cora
dataset and trained GCN checkpoint. Ensures PyG-compliant `edge_weight` support
without altering any raw handover files.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class CoraGCNModelWrapper(torch.nn.Module):
    """PyG-compliant defensive wrapper around the baseline Cora GCN architecture.

    Adds full support for `edge_weight: Optional[torch.Tensor] = None` during
    forward execution, ensuring seamless gradient and weight propagation for
    XAI explainers (GNNExplainer, PGExplainer, Saliency, SubgraphX).
    """

    def __init__(
        self,
        in_dim: int = 1433,
        hidden: int = 64,
        n_classes: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # cached=False is required for dynamic edge masking during XAI optimization
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
        """Forward pass supporting optional continuous edge attribution weights and distribution types."""
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


@dataclass
class CoraHandoverBundle:
    """Container holding loaded and wrapped Cora graph artifacts."""

    data: Data
    model: CoraGCNModelWrapper
    val_nodes: List[int]
    multiseed_subset: List[int]
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    original_pred: torch.Tensor


class CoraHandoverAdapter:
    """Ingestion service for Cora handover files."""

    def __init__(self, handover_dir: str = "data/t01_handover/cora") -> None:
        self.handover_dir = Path(handover_dir)
        self.data_path = self.handover_dir / "data.pt"
        self.ckpt_path = self.handover_dir / "gcn_ckpt.pt"
        self.val_nodes_path = self.handover_dir / "val_nodes.json"
        self.config_path = self.handover_dir / "config.json"

    def verify_files_exist(self) -> bool:
        """Verify all required handover files are present."""
        required = [self.data_path, self.ckpt_path, self.val_nodes_path, self.config_path]
        return all(p.exists() for p in required)

    def load(self, device: str = "cpu") -> CoraHandoverBundle:
        """Load and wrap Cora artifacts into a ready-to-use bundle.

        Args:
            device: Target torch device ('cpu' or 'cuda').

        Returns:
            CoraHandoverBundle containing Data, wrapped Model, val_nodes, and metadata.
        """
        if not self.verify_files_exist():
            raise FileNotFoundError(
                f"Missing handover files in {self.handover_dir}. "
                "Ensure handover archive is extracted to data/t01_handover/."
            )

        # 1. Ingest Raw Graph Data
        raw_data = torch.load(self.data_path, map_location=device, weights_only=False)
        data = Data(
            x=raw_data["x"].to(device),
            edge_index=raw_data["edge_index"].to(device),
            y=raw_data["y"].to(device),
            train_mask=raw_data["train_mask"].to(device),
            val_mask=raw_data["val_mask"].to(device),
            test_mask=raw_data["test_mask"].to(device),
        )

        # 2. Ingest Config & Metadata
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(self.val_nodes_path, "r", encoding="utf-8") as f:
            nodes_meta = json.load(f)
            val_nodes = nodes_meta.get("nodes", [])
            multiseed_subset = nodes_meta.get("multiseed_subset", [])

        # 3. Ingest Trained Checkpoint & Construct Defensive Model Wrapper
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        model = CoraGCNModelWrapper(
            in_dim=ckpt.get("in_dim", 1433),
            hidden=ckpt.get("hidden", 64),
            n_classes=ckpt.get("n_classes", 7),
            dropout=ckpt.get("dropout", 0.5),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        # Freeze parameters for XAI inference
        for param in model.parameters():
            param.requires_grad_(False)

        metrics = ckpt.get("metrics", {})
        original_pred = ckpt.get("pred", torch.empty(0)).to(device)

        return CoraHandoverBundle(
            data=data,
            model=model,
            val_nodes=val_nodes,
            multiseed_subset=multiseed_subset,
            config=config,
            metrics=metrics,
            original_pred=original_pred,
        )
