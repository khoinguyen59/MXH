"""TwiBot-20 Twitter Bot Detection Social Network Loader.

Loads the TwiBot-20 benchmark graph containing user interactions, follow relations,
and metadata features for social bot identification, partitioned with anti-leakage splitting.
"""

from pathlib import Path
import json
import torch
import numpy as np
from torch_geometric.data import Data

from src.datasets.anti_leakage_splitter import AntiLeakageSplitter
from src.datasets.base_loader import BaseDatasetLoader


class TwiBot20Loader(BaseDatasetLoader):
    """Loads and standardizes the TwiBot-20 Twitter benchmark."""

    def __init__(self, data_dir: str, random_seed: int = 42) -> None:
        """Initialize TwiBot-20 loader.

        Args:
            data_dir: Base directory for dataset storage.
            random_seed: Seed for reproducible partitioning.
        """
        super().__init__(data_dir=data_dir, random_seed=random_seed)
        self.raw_folder = Path(data_dir) / "twibot20"

    def load(self) -> Data:
        """Load processed TwiBot-20 graph or generate a representative graph if raw files are missing.

        Returns:
            PyG Data object with standardized node features and stratified split masks.
        """
        processed_file = self.raw_folder / "processed_graph.pt"
        if processed_file.exists():
            data = torch.load(processed_file)
        else:
            # Check if edge.csv and node.json exist
            edge_file = self.raw_folder / "edge.csv"
            node_file = self.raw_folder / "node.json"
            label_file = self.raw_folder / "label.csv"

            if edge_file.exists() and node_file.exists():
                data = self._process_raw_files(edge_file, node_file, label_file)
            else:
                # If dataset not yet downloaded, load canonical social network topology
                data = self._download_or_scaffold()

        splitter = AntiLeakageSplitter(random_seed=self.random_seed)
        train_mask, val_mask, test_mask = splitter.split_nodes(data, stratify=True)
        data = splitter.fit_transform_features(data, train_mask)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        self.data = data
        return data

    def _download_or_scaffold(self) -> Data:
        """Scaffold download handler for TwiBot-20 benchmark."""
        self.raw_folder.mkdir(parents=True, exist_ok=True)
        # Note: Actual full download requires ~2GB storage; provides clear instructions
        processed_file = self.raw_folder / "processed_graph.pt"
        if not processed_file.exists():
            # Create a placeholder info file for dataset download source
            readme = self.raw_folder / "DOWNLOAD_INSTRUCTIONS.md"
            if not readme.exists():
                readme.write_text(
                    "# TwiBot-20 Dataset\n"
                    "Download source: https://github.com/YingtongDou/TwiBot-20\n"
                    "Place node.json, edge.csv, label.csv into this directory.\n"
                )
            raise FileNotFoundError(
                f"TwiBot-20 data files not found in {self.raw_folder}. "
                "Please place raw files or run the data acquisition script."
            )
        return torch.load(processed_file)

    def _process_raw_files(self, edge_path: Path, node_path: Path, label_path: Path) -> Data:
        """Process raw JSON and CSV tables into a unified PyG Data object."""
        # Read edges
        edges = []
        with open(edge_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    edges.append((int(parts[0]), int(parts[1])))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Read labels and features from node.json
        with open(node_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)

        num_nodes = len(nodes)
        features = np.zeros((num_nodes, 16), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.int64)

        for i, node in enumerate(nodes):
            labels[i] = 1 if node.get("label") == "bot" else 0
            features[i, 0] = float(node.get("followers_count", 0))
            features[i, 1] = float(node.get("friends_count", 0))
            features[i, 2] = float(node.get("statuses_count", 0))

        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            num_nodes=num_nodes,
        )
        return data
