"""Base Abstract Class for Dataset Ingestion Services.

Enforces standardized interface across synthetic benchmarks and real-world social networks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from torch_geometric.data import Data


class BaseDatasetLoader(ABC):
    """Abstract base class for all graph dataset ingestion modules."""

    def __init__(self, data_dir: str, random_seed: int = 42) -> None:
        """Initialize the dataset loader.

        Args:
            data_dir: Path to directory where datasets are stored or downloaded.
            random_seed: Random seed for deterministic reproducibility.
        """
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.data: Data = None

    @abstractmethod
    def load(self) -> Data:
        """Load, process, and partition the dataset.

        Returns:
            PyG Data object ready for training and evaluation.
        """
        pass

    def summary(self) -> Dict[str, Any]:
        """Compute basic structural statistics of the loaded graph.

        Returns:
            Dictionary containing node count, edge count, feature dimension, and class count.
        """
        if self.data is None:
            raise ValueError("Dataset has not been loaded yet. Call load() first.")

        num_nodes = self.data.num_nodes
        num_edges = self.data.num_edges
        num_features = self.data.num_features if self.data.x is not None else 0
        num_classes = len(self.data.y.unique()) if self.data.y is not None else 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": num_features,
            "num_classes": num_classes,
            "is_directed": self.data.is_directed(),
            "has_isolated_nodes": self.data.has_isolated_nodes(),
            "has_self_loops": self.data.has_self_loops(),
        }
