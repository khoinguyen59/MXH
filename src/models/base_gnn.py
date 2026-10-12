"""Base Abstract GNN Backbone Architecture.

Provides unified training, evaluation, feature representation, and model persistence
interfaces across all GNN architectures (GCN, GraphSAGE, GAT).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class BaseGNN(nn.Module, ABC):
    """Abstract base class for all Graph Neural Network backbones."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        """Initialize the base GNN model.

        Args:
            in_channels: Dimension of input node features.
            hidden_channels: Hidden representation dimension.
            out_channels: Number of target classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

    @abstractmethod
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass computing log-probabilities or class logits.

        Args:
            x: Node feature tensor of shape [num_nodes, in_channels].
            edge_index: Graph connectivity tensor of shape [2, num_edges].
            edge_weight: Optional edge weight tensor of shape [num_edges].

        Returns:
            Log-softmax tensor of shape [num_nodes, out_channels].
        """
        pass

    @abstractmethod
    def get_node_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Extract intermediate latent representation of nodes before final classification head.

        Args:
            x: Node feature tensor.
            edge_index: Graph connectivity tensor.

        Returns:
            Latent representation tensor of shape [num_nodes, hidden_channels].
        """
        pass

    def fit_model(
        self,
        data: Data,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 20,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, list]:
        """Train the GNN model with early stopping on validation loss.

        Args:
            data: PyG Data object containing train_mask and val_mask.
            epochs: Maximum training epochs.
            lr: Learning rate for Adam optimizer.
            weight_decay: L2 regularization factor.
            patience: Early stopping patience epochs.
            device: Hardware execution device.

        Returns:
            History dictionary recording training and validation loss and accuracy per epoch.
        """
        self.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # Validation evaluation
            train_acc, train_loss = self.evaluate(data, data.train_mask)
            val_acc, val_loss = self.evaluate(data, data.val_mask)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history

    @torch.no_grad()
    def evaluate(self, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
        """Compute classification accuracy and negative log-likelihood loss on a given mask.

        Args:
            data: PyG Data object.
            mask: Boolean tensor indicating target subset of nodes.

        Returns:
            Tuple of (accuracy, loss).
        """
        self.eval()
        out = self(data.x, data.edge_index)
        loss = F.nll_loss(out[mask], data.y[mask]).item()
        pred = out[mask].argmax(dim=-1)
        correct = (pred == data.y[mask]).sum().item()
        acc = correct / mask.sum().item() if mask.sum().item() > 0 else 0.0
        return acc, loss

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights to disk.

        Args:
            path: Target file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "dropout": self.dropout,
        }, path)

    def load_checkpoint(self, path: Path, device: torch.device = torch.device("cpu")) -> None:
        """Load model weights from disk.

        Args:
            path: File path of saved checkpoint.
            device: Target device to map tensors to.
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["state_dict"])
