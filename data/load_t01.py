"""Nạp gói bàn giao T01 (chỉ cần torch + torch_geometric). Dùng: from load_t01 import load; d, model, pred, nodes = load('bashapes')"""
import json, pathlib, torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

HERE = pathlib.Path(__file__).resolve().parent


class GCN(torch.nn.Module):
    def __init__(self, i, h, o, p):
        super().__init__(); self.c1, self.c2, self.p = GCNConv(i, h), GCNConv(h, o), p

    def forward(self, x, edge_index):
        x = F.dropout(x, self.p, self.training); x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, self.p, self.training); return self.c2(x, edge_index)


def load(ds, root=HERE):
    d = torch.load(root / ds / "data.pt", weights_only=False)
    ck = torch.load(root / ds / "gcn_ckpt.pt", weights_only=False)
    model = GCN(ck["in_dim"], ck["hidden"], ck["n_classes"], ck["dropout"])
    model.load_state_dict(ck["state_dict"]); model.eval()           # luôn eval(), không cập nhật trọng số
    for p in model.parameters():
        p.requires_grad_(False)
    nodes = json.load(open(root / ds / "val_nodes.json"))["nodes"]
    return d, model, ck["pred"], nodes


if __name__ == "__main__":
    for ds in ("bashapes", "cora"):
        d, m, pred, nodes = load(ds)
        with torch.no_grad():
            p2 = m(d["x"], d["edge_index"]).argmax(1)
        assert torch.equal(p2, pred), "dự đoán nạp lại không khớp"
        acc = (p2[d["test_mask"]] == d["y"][d["test_mask"]]).float().mean().item()
        print(ds, "nạp OK; test acc", round(acc, 4), "| node validation:", len(nodes))
