"""Adapter Service for BA-Shapes Handover Artifacts.

Provides non-invasive ingestion and in-memory defensive wrapping of the BA-Shapes
synthetic benchmark and trained GCN checkpoint from Nhóm 1 (`data/t01_handover/bashapes`).
Adds full PyG-compliant `edge_weight` support without modifying any raw handover files.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch_geometric.data import Data

from src.t02_explainers.adapters.t01_model_adapter import T01GCNModelWrapper


@dataclass
class BAShapesHandoverBundle:
    """Container holding loaded and wrapped BA-Shapes graph artifacts."""

    data: Data
    model: T01GCNModelWrapper
    val_nodes: List[int]
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    original_pred: torch.Tensor


class BAShapesHandoverAdapter:
    """Ingestion service for BA-Shapes handover files."""

    def __init__(self, handover_dir: str = "data/t01_handover/bashapes") -> None:
        self.handover_dir = Path(handover_dir)
        self.data_path = self.handover_dir / "data.pt"
        self.ckpt_path = self.handover_dir / "gcn_ckpt.pt"
        self.val_nodes_path = self.handover_dir / "val_nodes.json"
        self.config_path = self.handover_dir / "config.json"

    def verify_files_exist(self) -> bool:
        """Verify all required handover files are present."""
        required = [self.data_path, self.ckpt_path, self.val_nodes_path, self.config_path]
        return all(p.exists() for p in required)

    def load(self, device: str = "cpu") -> BAShapesHandoverBundle:
        """Load and wrap BA-Shapes artifacts into a ready-to-use bundle.

        Args:
            device: Target torch device ('cpu' or 'cuda').

        Returns:
            BAShapesHandoverBundle containing Data, wrapped Model, val_nodes, and metadata.
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
            gt_edge_mask=raw_data.get("gt_edge_mask", torch.empty(0)).to(device),
        )

        # 2. Ingest Config & Metadata
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(self.val_nodes_path, "r", encoding="utf-8") as f:
            nodes_meta = json.load(f)
            val_nodes = nodes_meta.get("nodes", [])

        # 3. Ingest Trained Checkpoint & Construct Defensive Model Wrapper
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        model = T01GCNModelWrapper(
            in_dim=ckpt.get("in_dim", 10),
            hidden=ckpt.get("hidden", 64),
            n_classes=ckpt.get("n_classes", 4),
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

        return BAShapesHandoverBundle(
            data=data,
            model=model,
            val_nodes=val_nodes,
            config=config,
            metrics=metrics,
            original_pred=original_pred,
        )
