"""Nap goi ban giao T01 chuan hoa (torch + torch_geometric).
Dung:
    from load_t01 import load
    d, model, pred, nodes = load("bashapes")  # hoac "cora", "cora_dense"
"""
import json
import pathlib
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

HERE = pathlib.Path(__file__).resolve().parent


class GCN(torch.nn.Module):
    """GCN da tang voi ho tro lien tuc trong so canh edge_weight cho Explainers."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.p = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, return_type: str = "log_probs") -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.p, training=self.training)
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        if return_type == "raw":
            return x
        elif return_type == "probs":
            return F.softmax(x, dim=-1)
        return F.log_softmax(x, dim=-1)


def load(ds: str, root: pathlib.Path = HERE) -> Tuple[Data, torch.nn.Module, torch.Tensor, List[int]]:
    """Nap dataset PyG Data, mo hinh GCN (eval, frozen), tensor du doan pred, va validation nodes."""
    d = torch.load(root / ds / "data.pt", weights_only=False)
    ck = torch.load(root / ds / "gcn_ckpt.pt", weights_only=False)
    num_layers = ck.get("num_layers", 3 if ds == "bashapes" else 2)
    model = GCN(
        in_channels=ck["in_dim"],
        hidden_channels=ck["hidden"],
        out_channels=ck["n_classes"],
        dropout=ck["dropout"],
        num_layers=num_layers,
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    val_nodes_file = root / ds / "val_nodes.json"
    nodes = json.load(open(val_nodes_file, encoding="utf-8"))["nodes"] if val_nodes_file.exists() else []
    return d, model, ck["pred"], nodes


if __name__ == "__main__":
    for ds in ("bashapes", "cora"):
        d, m, pred, nodes = load(ds)
        with torch.no_grad():
            p2 = m(d.x, d.edge_index).argmax(dim=-1)
        assert torch.equal(p2, pred), f"{ds}: du doan nap lai khong khop!"
        acc = (p2[d.test_mask] == d.y[d.test_mask]).float().mean().item()
        print(f"[{ds}] Nap OK | Test Acc: {acc:.4f} | Validation nodes: {len(nodes)}")
