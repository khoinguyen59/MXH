import os
import json
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer

SEED = 42
torch.manual_seed(SEED)

class GCN(torch.nn.Module):
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

def evaluate_node(explainer, model, x, edge_index, node_idx, pred_class, budget_ratio=0.2):
    t0 = time.time()
    explanation = explainer(x=x, edge_index=edge_index, index=node_idx)
    latency = time.time() - t0

    edge_mask = explanation.edge_mask
    sub_edge_index = explanation.edge_index

    orig_out = model(x, edge_index)
    orig_prob = F.softmax(orig_out[node_idx], dim=-1)[pred_class].item()

    num_sub_edges = sub_edge_index.size(1)
    k = max(1, int(num_sub_edges * budget_ratio))

    _, topk_indices = torch.topk(edge_mask, k=k)
    exp_mask = torch.zeros_like(edge_mask, dtype=torch.bool)
    exp_mask[topk_indices] = True

    keep_edges_fid_plus = sub_edge_index[:, ~exp_mask]
    out_fid_plus = model(x, keep_edges_fid_plus)
    prob_fid_plus = F.softmax(out_fid_plus[node_idx], dim=-1)[pred_class].item()
    fid_plus = orig_prob - prob_fid_plus

    keep_edges_fid_minus = sub_edge_index[:, exp_mask]
    out_fid_minus = model(x, keep_edges_fid_minus)
    prob_fid_minus = F.softmax(out_fid_minus[node_idx], dim=-1)[pred_class].item()
    fid_minus = orig_prob - prob_fid_minus

    sparsity = 1.0 - (k / num_sub_edges)

    return {
        "node_idx": int(node_idx),
        "pred_class": int(pred_class),
        "budget_edges": k,
        "subgraph_edges": num_sub_edges,
        "fidelity_plus": round(fid_plus, 4),
        "fidelity_minus": round(fid_minus, 4),
        "sparsity": round(sparsity, 4),
        "latency_sec": round(latency, 4)
    }

def evaluate_random(model, x, edge_index, node_idx, pred_class, sub_edge_index, budget_edges):
    t0 = time.time()
    num_sub_edges = sub_edge_index.size(1)
    k = budget_edges

    orig_out = model(x, edge_index)
    orig_prob = F.softmax(orig_out[node_idx], dim=-1)[pred_class].item()

    perm = torch.randperm(num_sub_edges)
    topk_indices = perm[:k]
    exp_mask = torch.zeros(num_sub_edges, dtype=torch.bool)
    exp_mask[topk_indices] = True

    keep_edges_fid_plus = sub_edge_index[:, ~exp_mask]
    out_fid_plus = model(x, keep_edges_fid_plus)
    prob_fid_plus = F.softmax(out_fid_plus[node_idx], dim=-1)[pred_class].item()
    fid_plus = orig_prob - prob_fid_plus

    keep_edges_fid_minus = sub_edge_index[:, exp_mask]
    out_fid_minus = model(x, keep_edges_fid_minus)
    prob_fid_minus = F.softmax(out_fid_minus[node_idx], dim=-1)[pred_class].item()
    fid_minus = orig_prob - prob_fid_minus

    latency = time.time() - t0
    sparsity = 1.0 - (k / num_sub_edges)

    return {
        "node_idx": int(node_idx),
        "pred_class": int(pred_class),
        "budget_edges": k,
        "subgraph_edges": num_sub_edges,
        "fidelity_plus": round(fid_plus, 4),
        "fidelity_minus": round(fid_minus, 4),
        "sparsity": round(sparsity, 4),
        "latency_sec": round(latency, 4)
    }

def run_tier2_experiments():
    existing_file = "experiments/pilot_study/logs/pilot_benchmark_results.json"
    with open(existing_file, "r") as f:
        all_results = json.load(f)

    datasets = {
        "Chameleon": WikipediaNetwork(root='data/Chameleon', name='chameleon'),
        "CiteSeer": Planetoid(root='data/CiteSeer', name='CiteSeer')
    }

    for name, ds in datasets.items():
        print(f"\n=================== RUNNING TIER 2: {name} ===================")
        data = ds[0]
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_features = data.num_features
        
        # Labels and classes
        if data.y.dim() > 1 and data.y.size(1) > 1:
            num_classes = data.y.size(1)
            y = data.y.argmax(dim=-1)
        else:
            y = data.y
            num_classes = int(y.max().item() + 1)

        # Masks
        if hasattr(data, 'train_mask') and data.train_mask is not None:
            train_mask = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
            val_mask = data.val_mask[:, 0] if data.val_mask.dim() > 1 else data.val_mask
            test_mask = data.test_mask[:, 0] if data.test_mask.dim() > 1 else data.test_mask
        else:
            torch.manual_seed(SEED)
            perm = torch.randperm(num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[perm[:int(0.2*num_nodes)]] = True
            val_mask[perm[int(0.2*num_nodes):int(0.4*num_nodes)]] = True
            test_mask[perm[int(0.4*num_nodes):]] = True

        stats = {
            "tier": "Tier 2 (Mở rộng)",
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges),
            "num_features": int(num_features),
            "num_classes": int(num_classes),
            "num_train": int(train_mask.sum().item()),
            "num_val": int(val_mask.sum().item()),
            "num_test": int(test_mask.sum().item())
        }

        # Train GCN
        model = GCN(num_features, 64, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        t_start = time.time()
        for epoch in range(1, 31):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
        train_time = round(time.time() - t_start, 3)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=-1)
            train_acc = round((preds[train_mask] == y[train_mask]).float().mean().item(), 4)
            val_acc = round((preds[val_mask] == y[val_mask]).float().mean().item(), 4)
            test_acc = round((preds[test_mask] == y[test_mask]).float().mean().item(), 4)

        ckpt_path = f"experiments/pilot_study/checkpoints/{name.lower().replace(' ', '_')}_gcn.pt"
        torch.save(model.state_dict(), ckpt_path)

        train_metrics = {
            "train_time_sec": train_time,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc
        }

        # Explainer
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100, lr=0.01),
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw'
            ),
        )

        test_indices = torch.where(test_mask)[0]
        torch.manual_seed(SEED)
        perm = torch.randperm(len(test_indices))
        sample_nodes = test_indices[perm[:5]].tolist()

        gnn_results = []
        rand_results = []

        for node_idx in sample_nodes:
            pred_class = preds[node_idx].item()
            gnn_res = evaluate_node(explainer, model, data.x, data.edge_index, node_idx, pred_class, budget_ratio=0.2)
            gnn_results.append(gnn_res)

            exp = explainer(x=data.x, edge_index=data.edge_index, index=node_idx)
            sub_edge_index = exp.edge_index
            rand_res = evaluate_random(model, data.x, data.edge_index, node_idx, pred_class, sub_edge_index, gnn_res["budget_edges"])
            rand_results.append(rand_res)

        avg_gnn = {
            "avg_fidelity_plus": round(sum(r["fidelity_plus"] for r in gnn_results)/len(gnn_results), 4),
            "avg_fidelity_minus": round(sum(r["fidelity_minus"] for r in gnn_results)/len(gnn_results), 4),
            "avg_sparsity": round(sum(r["sparsity"] for r in gnn_results)/len(gnn_results), 4),
            "avg_latency_sec": round(sum(r["latency_sec"] for r in gnn_results)/len(gnn_results), 4)
        }

        avg_rand = {
            "avg_fidelity_plus": round(sum(r["fidelity_plus"] for r in rand_results)/len(rand_results), 4),
            "avg_fidelity_minus": round(sum(r["fidelity_minus"] for r in rand_results)/len(rand_results), 4),
            "avg_sparsity": round(sum(r["sparsity"] for r in rand_results)/len(rand_results), 4),
            "avg_latency_sec": round(sum(r["latency_sec"] for r in rand_results)/len(rand_results), 4)
        }

        all_results[name] = {
            "dataset_stats": stats,
            "training_metrics": train_metrics,
            "xai_metrics": {
                "gnnexplainer": {
                    "summary": avg_gnn,
                    "node_details": gnn_results
                },
                "random_baseline": {
                    "summary": avg_rand,
                    "node_details": rand_results
                }
            }
        }
        print(f"Finished {name}: Test Acc={test_acc}, GNNExplainer Fid+={avg_gnn['avg_fidelity_plus']}, Rand Fid+={avg_rand['avg_fidelity_plus']}")

    with open(existing_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nUpdated all results into:", existing_file)

if __name__ == "__main__":
    run_tier2_experiments()
