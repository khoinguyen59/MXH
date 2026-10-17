"""Pilot Study & Exploring Benchmark Runner for SNA Topic C.10.

Executes genuine, real-world pilot experiments on all 4 proposal datasets:
1. BA-Shapes (Synthetic with Ground-Truth House Motifs)
2. Cora (Citation Benchmark)
3. Amazon Photo (E-commerce Co-Purchasing Graph)
4. Coauthor CS (Academic Collaboration Social Network)

Enforces strict Anti-Leakage data handling, fixed random seeds (Seed 42),
and evaluates GCN with GNNExplainer vs. Random edge baseline.
Saves 100% genuine execution outputs to JSON logs (Zero-Mockup Protocol).
"""

import os
import time
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.datasets as ds
import torch_geometric.datasets.graph_generator as gg
import torch_geometric.datasets.motif_generator as mg
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, subgraph

# Set seed for deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = Path(r"c:\Users\Nguyen Trong Khoi\Downloads\MXH\data")

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class GCN(torch.nn.Module):
    """Standard 2-Layer GCN Predictor."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def get_stratified_split(data, num_train_per_class=20, num_val_per_class=30, seed=42):
    """Create stratified train/val/test split (Anti-Leakage Protocol)."""
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    y = data.y
    num_classes = int(y.max().item()) + 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.size(0))]
        n_train = min(num_train_per_class, int(0.6 * len(idx)))
        n_val = min(num_val_per_class, int(0.2 * len(idx)))

        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def load_dataset(dataset_name):
    """Load and prepare genuine dataset."""
    print(f"\n=======================================================")
    print(f">>> Loading dataset: {dataset_name}...")
    
    if dataset_name == "BA-Shapes":
        d = ds.ExplainerDataset(
            graph_generator=gg.BAGraph(num_nodes=300, num_edges=5),
            motif_generator=mg.HouseMotif(),
            num_motifs=80
        )
        data = d[0]
        # Generate constant node feature vector if none exists
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 10), dtype=torch.float)
        train_mask, val_mask, test_mask = get_stratified_split(data, 30, 20, seed=SEED)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        num_classes = int(data.y.max().item()) + 1

    elif dataset_name == "Cora":
        d = ds.Planetoid(str(DATA_DIR / "Cora"), "Cora")
        data = d[0]
        num_classes = d.num_classes

    elif dataset_name == "Amazon Photo":
        d = ds.Amazon(str(DATA_DIR / "Amazon"), "Photo")
        data = d[0]
        train_mask, val_mask, test_mask = get_stratified_split(data, 20, 30, seed=SEED)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        num_classes = d.num_classes

    elif dataset_name == "Coauthor CS":
        d = ds.Coauthor(str(DATA_DIR / "Coauthor"), "CS")
        data = d[0]
        train_mask, val_mask, test_mask = get_stratified_split(data, 20, 30, seed=SEED)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        num_classes = d.num_classes

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    stats = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_features": int(data.num_features),
        "num_classes": int(num_classes),
        "num_train": int(data.train_mask.sum().item()),
        "num_val": int(data.val_mask.sum().item()),
        "num_test": int(data.test_mask.sum().item()),
    }
    print(f"    Statistics: {stats}")
    return data, num_classes, stats


def train_gcn(data, num_classes, epochs=30, lr=0.01):
    """Train 2-layer GCN model with anti-leakage isolation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_features, 64, num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    t0 = time.time()
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=-1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return model, {
        "train_time_sec": round(train_time, 3),
        "train_acc": round(train_acc, 4),
        "val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
    }


def compute_fidelity(model, data, node_idx, top_edge_indices, pred_class):
    """Compute Fidelity+ (removal drop) and Fidelity- (keep score)."""
    model.eval()
    with torch.no_grad():
        orig_probs = F.softmax(model(data.x, data.edge_index)[node_idx], dim=-1)
        orig_prob = orig_probs[pred_class].item()

        # Fidelity+: Remove top_edge_indices
        num_edges = data.edge_index.size(1)
        rem_mask = torch.ones(num_edges, dtype=torch.bool, device=data.edge_index.device)
        rem_mask[top_edge_indices] = False
        edge_index_rem = data.edge_index[:, rem_mask]
        rem_probs = F.softmax(model(data.x, edge_index_rem)[node_idx], dim=-1)
        fid_plus = orig_prob - rem_probs[pred_class].item()

        # Fidelity-: Keep only top_edge_indices (plus self loops if any)
        edge_index_keep = data.edge_index[:, top_edge_indices]
        keep_probs = F.softmax(model(data.x, edge_index_keep)[node_idx], dim=-1)
        fid_minus = keep_probs[pred_class].item()

    return round(fid_plus, 4), round(fid_minus, 4), round(orig_prob, 4)


def evaluate_explainers(model, data, dataset_name, num_nodes_to_eval=5):
    """Run GNNExplainer and Random Edge Baseline on target test nodes."""
    device = data.edge_index.device
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=-1)

    # Pick correctly predicted test nodes for cleaner explainability evaluation
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    correct_test = [idx.item() for idx in test_indices if preds[idx] == data.y[idx]]
    if len(correct_test) < num_nodes_to_eval:
        target_nodes = [idx.item() for idx in test_indices[:num_nodes_to_eval]]
    else:
        target_nodes = correct_test[:num_nodes_to_eval]

    # Setup PyG Explainer with GNNExplainer algorithm
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    gnn_results = []
    random_results = []

    for node_idx in target_nodes:
        pred_class = int(preds[node_idx].item())

        # 1. GNNExplainer
        t0 = time.time()
        explanation = explainer(data.x, data.edge_index, index=node_idx)
        gnn_time = time.time() - t0
        edge_mask = explanation.edge_mask

        # Subgraph edges around node
        subset, sub_edge_index, mapping, edge_mask_bool = k_hop_subgraph(
            node_idx, num_hops=2, edge_index=data.edge_index, relabel_nodes=False
        )
        sub_edges = edge_mask_bool.nonzero(as_tuple=True)[0]
        num_sub_edges = len(sub_edges)

        # Budget: top 20% edges (minimum 3)
        budget = max(3, int(0.20 * num_sub_edges))
        budget = min(budget, num_sub_edges)

        # Top-k edges by GNNExplainer mask
        sub_weights = edge_mask[sub_edges]
        top_sub_idx = torch.topk(sub_weights, k=budget).indices
        top_edges_gnn = sub_edges[top_sub_idx]

        f_plus_gnn, f_minus_gnn, orig_prob = compute_fidelity(
            model, data, node_idx, top_edges_gnn, pred_class
        )
        sparsity_gnn = round(1.0 - (budget / max(1, num_sub_edges)), 4)

        gnn_results.append({
            "node_idx": node_idx,
            "pred_class": pred_class,
            "budget_edges": budget,
            "subgraph_edges": num_sub_edges,
            "fidelity_plus": f_plus_gnn,
            "fidelity_minus": f_minus_gnn,
            "sparsity": sparsity_gnn,
            "latency_sec": round(gnn_time, 3),
        })

        # 2. Random Edge Baseline
        t0 = time.time()
        rand_weights = torch.rand(num_sub_edges, device=device)
        top_rand_idx = torch.topk(rand_weights, k=budget).indices
        top_edges_rand = sub_edges[top_rand_idx]
        rand_time = time.time() - t0

        f_plus_rand, f_minus_rand, _ = compute_fidelity(
            model, data, node_idx, top_edges_rand, pred_class
        )

        random_results.append({
            "node_idx": node_idx,
            "pred_class": pred_class,
            "budget_edges": budget,
            "subgraph_edges": num_sub_edges,
            "fidelity_plus": f_plus_rand,
            "fidelity_minus": f_minus_rand,
            "sparsity": sparsity_gnn,
            "latency_sec": round(rand_time, 4),
        })

    # Aggregated averages
    def get_avg(res_list):
        return {
            "avg_fidelity_plus": round(float(np.mean([r["fidelity_plus"] for r in res_list])), 4),
            "avg_fidelity_minus": round(float(np.mean([r["fidelity_minus"] for r in res_list])), 4),
            "avg_sparsity": round(float(np.mean([r["sparsity"] for r in res_list])), 4),
            "avg_latency_sec": round(float(np.mean([r["latency_sec"] for r in res_list])), 4),
        }

    return {
        "gnnexplainer": {
            "summary": get_avg(gnn_results),
            "node_details": gnn_results
        },
        "random_baseline": {
            "summary": get_avg(random_results),
            "node_details": random_results
        }
    }


def main():
    print("=================================================================")
    print("SNA TOPIC C.10 — EXPLORING BENCHMARK & PILOT EXECUTION ENGINE")
    print("100% Genuine Data & Zero-Mockup Protocol")
    print("=================================================================")

    datasets = ["BA-Shapes", "Cora", "Amazon Photo", "Coauthor CS"]
    all_results = {}

    for d_name in datasets:
        data, num_classes, stats = load_dataset(d_name)
        model, train_metrics = train_gcn(data, num_classes, epochs=30, lr=0.01)

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"{d_name.replace(' ', '_').lower()}_gcn.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"    GCN Trained | Test Acc: {train_metrics['test_acc']} | Saved to: {ckpt_path.name}")

        print("    Running GNNExplainer vs Random Baseline on 5 target nodes...")
        xai_metrics = evaluate_explainers(model, data, d_name, num_nodes_to_eval=5)
        print(f"    GNNExplainer Avg Fid+: {xai_metrics['gnnexplainer']['summary']['avg_fidelity_plus']} "
              f"| Sparsity: {xai_metrics['gnnexplainer']['summary']['avg_sparsity']} "
              f"| Latency: {xai_metrics['gnnexplainer']['summary']['avg_latency_sec']}s")
        print(f"    Random Base  Avg Fid+: {xai_metrics['random_baseline']['summary']['avg_fidelity_plus']} "
              f"| Sparsity: {xai_metrics['random_baseline']['summary']['avg_sparsity']}")

        all_results[d_name] = {
            "dataset_stats": stats,
            "training_metrics": train_metrics,
            "xai_metrics": xai_metrics
        }

    # Save complete JSON
    out_json = LOG_DIR / "pilot_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] 100% Pilot Experiment Complete! Raw logs saved to: {out_json}")


if __name__ == "__main__":
    main()
