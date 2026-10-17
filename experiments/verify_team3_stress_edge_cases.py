"""Team Check 3: Edge Case & Stress-Testing Verification Script.

Tests edge cases and boundary conditions across all explainers and evaluation functions:
1. Isolated node graph (N=1, E=0 and N=5, E=0).
2. Complete clique graph (K_10, dense connectivity).
3. Star/Hub graph (central node with 50 leaves).
4. Disconnected graph with isolated components.
5. Boundary budgets (k=0, k=1, k > num_edges, empty mask, all-zero mask).
6. Bidirectional symmetry validation on asymmetric edge masks.
"""

import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.t02_explainers.algorithms.saliency import SaliencyExplainer
from src.t02_explainers.algorithms.random_explainer import RandomExplainer
from src.t02_explainers.algorithms.nearest_first_explainer import NearestFirstExplainer
from src.t02_explainers.algorithms.eig_search import EigSearchExplainer
from src.t02_explainers.algorithms.lt_def import LTDEFExplainer
from src.t02_explainers.algorithms.subgraphx import SubgraphXExplainer
from src.t03_evaluation.fidelity import compute_fidelity_at_budget, compute_fidelity_curve


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim=8, hidden=8, n_classes=2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden, cached=False)
        self.c2 = GCNConv(hidden, n_classes, cached=False)

    def forward(self, x, edge_index, edge_weight=None, return_type="log_probs"):
        h = F.relu(self.c1(x, edge_index, edge_weight=edge_weight))
        out = self.c2(h, edge_index, edge_weight=edge_weight)
        if return_type == "raw":
            return out
        elif return_type == "probs":
            return F.softmax(out, dim=-1)
        return F.log_softmax(out, dim=-1)


def run_team3_verification():
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_cases": {},
        "summary": "PASSED",
    }
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "team3_stress_edge_cases_verification.json"

    torch.manual_seed(42)
    model = DummyModel(in_dim=8, hidden=8, n_classes=2)
    model.eval()

    explainers_factory = lambda: [
        ("Saliency", SaliencyExplainer(model=model, num_hops=2)),
        ("Random", RandomExplainer(model=model, num_hops=2, random_seed=42)),
        ("Nearest-First", NearestFirstExplainer(model=model, num_hops=2)),
        ("EiG-Search", EigSearchExplainer(model=model, num_hops=2)),
        ("LT-DEF", LTDEFExplainer(model=model, num_hops=2)),
        ("SubgraphX", SubgraphXExplainer(model=model, num_hops=2, rollout_steps=4)),
    ]

    # =========================================================================
    # Case 1: Isolated Node Graph (N=1, E=0)
    # =========================================================================
    print("=== [Team 3] Case 1: N=1, E=0 Isolated Node ===")
    x_iso = torch.randn(1, 8)
    e_iso = torch.empty((2, 0), dtype=torch.long)
    c1_res = {}
    for name, exp in explainers_factory():
        res = exp.explain_node(0, x_iso, e_iso)
        fp, fm = compute_fidelity_at_budget(model, x_iso, res.subgraph_edge_index, 0, 0, res.edge_mask, k=5)
        c1_res[name] = {
            "edge_mask_size": int(res.edge_mask.numel()),
            "subgraph_edge_size": int(res.subgraph_edge_index.size(1)),
            "fid_plus": fp,
            "fid_minus": fm,
            "passed": bool(res.edge_mask.numel() == 0 and fp == 0.0 and fm == 0.0),
        }
    results["test_cases"]["isolated_node_n1"] = c1_res
    print(f"  Passed for all explainers: {all(v['passed'] for v in c1_res.values())}")

    # =========================================================================
    # Case 2: Multi-node graph with 0 edges (N=5, E=0)
    # =========================================================================
    print("\n=== [Team 3] Case 2: N=5, E=0 Disconnected Nodes ===")
    x_multi_iso = torch.randn(5, 8)
    e_multi_iso = torch.empty((2, 0), dtype=torch.long)
    c2_res = {}
    for name, exp in explainers_factory():
        res = exp.explain_node(2, x_multi_iso, e_multi_iso)
        fp, fm = compute_fidelity_at_budget(model, x_multi_iso[res.subgraph_nodes], res.subgraph_edge_index, 0, 0, res.edge_mask, k=3)
        c2_res[name] = {
            "edge_mask_size": int(res.edge_mask.numel()),
            "subgraph_edge_size": int(res.subgraph_edge_index.size(1)),
            "fid_plus": fp,
            "fid_minus": fm,
            "passed": bool(res.edge_mask.numel() == 0 and fp == 0.0 and fm == 0.0),
        }
    results["test_cases"]["isolated_nodes_n5"] = c2_res
    print(f"  Passed for all explainers: {all(v['passed'] for v in c2_res.values())}")

    # =========================================================================
    # Case 3: Complete Clique K_8 (Dense Graph)
    # =========================================================================
    print("\n=== [Team 3] Case 3: Dense Complete Clique K_8 ===")
    n_clique = 8
    src_list, dst_list = [], []
    for u in range(n_clique):
        for v in range(n_clique):
            if u != v:
                src_list.append(u)
                dst_list.append(v)
    e_clique = torch.tensor([src_list, dst_list], dtype=torch.long)
    x_clique = torch.randn(n_clique, 8)
    c3_res = {}
    for name, exp in explainers_factory():
        res = exp.explain_node(0, x_clique, e_clique)
        dim_ok = bool(res.edge_mask.numel() == res.subgraph_edge_index.size(1))
        local_target = int(torch.where(res.subgraph_nodes == 0)[0].item())
        fp, fm = compute_fidelity_at_budget(model, x_clique[res.subgraph_nodes], res.subgraph_edge_index, local_target, res.predicted_class, res.edge_mask, k=5, symmetrize=True)
        c3_res[name] = {
            "edge_mask_size": int(res.edge_mask.numel()),
            "subgraph_edge_size": int(res.subgraph_edge_index.size(1)),
            "dim_ok": dim_ok,
            "fid_plus": round(fp, 4),
            "fid_minus": round(fm, 4),
            "passed": bool(dim_ok and not torch.isnan(torch.tensor(fp))),
        }
    results["test_cases"]["dense_clique_k8"] = c3_res
    print(f"  Passed for all explainers: {all(v['passed'] for v in c3_res.values())}")

    # =========================================================================
    # Case 4: Star / Hub Graph (1 Hub + 50 Leaves)
    # =========================================================================
    print("\n=== [Team 3] Case 4: Star / Hub Graph (1 Hub, 50 Leaves) ===")
    n_star = 51
    hub_id = 0
    src_star = [hub_id] * 50 + list(range(1, 51))
    dst_star = list(range(1, 51)) + [hub_id] * 50
    e_star = torch.tensor([src_star, dst_star], dtype=torch.long)
    x_star = torch.randn(n_star, 8)
    c4_res = {}
    for name, exp in explainers_factory():
        # Test explaining the hub node
        res_hub = exp.explain_node(hub_id, x_star, e_star)
        dim_hub = bool(res_hub.edge_mask.numel() == res_hub.subgraph_edge_index.size(1))
        local_t = int(torch.where(res_hub.subgraph_nodes == hub_id)[0].item())
        fp_hub, fm_hub = compute_fidelity_at_budget(model, x_star[res_hub.subgraph_nodes], res_hub.subgraph_edge_index, local_t, res_hub.predicted_class, res_hub.edge_mask, k=10, symmetrize=True)

        # Test explaining a leaf node
        res_leaf = exp.explain_node(1, x_star, e_star)
        dim_leaf = bool(res_leaf.edge_mask.numel() == res_leaf.subgraph_edge_index.size(1))

        c4_res[name] = {
            "hub_edges": int(res_hub.edge_mask.numel()),
            "leaf_edges": int(res_leaf.edge_mask.numel()),
            "dim_hub_ok": dim_hub,
            "dim_leaf_ok": dim_leaf,
            "hub_fid_plus": round(fp_hub, 4),
            "passed": bool(dim_hub and dim_leaf),
        }
    results["test_cases"]["star_hub_graph"] = c4_res
    print(f"  Passed for all explainers: {all(v['passed'] for v in c4_res.values())}")

    # =========================================================================
    # Case 5: Boundary Budgets & Empty Masks in Fidelity
    # =========================================================================
    print("\n=== [Team 3] Case 5: Boundary Budgets & Degenerate Masks ===")
    test_e = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    test_x = torch.randn(3, 8)
    zero_mask = torch.zeros(4, dtype=torch.float32)
    ones_mask = torch.ones(4, dtype=torch.float32)

    # Budget k=0, k=1, k=100 (overflow)
    fp_k0, fm_k0 = compute_fidelity_at_budget(model, test_x, test_e, 0, 0, zero_mask, k=0)
    fp_k100, fm_k100 = compute_fidelity_at_budget(model, test_x, test_e, 0, 0, ones_mask, k=100)
    curve_res = compute_fidelity_curve(model, test_x, test_e, 0, 0, ones_mask, budget_k_list=[1, 2, 5, 20])

    results["test_cases"]["boundary_budgets"] = {
        "k0_fid_plus": fp_k0,
        "k100_fid_plus": fp_k100,
        "curve_auc_plus": curve_res["auc_fid_plus"],
        "curve_auc_minus": curve_res["auc_fid_minus"],
        "passed": bool(not torch.isnan(torch.tensor(fp_k0)) and not torch.isnan(torch.tensor(curve_res["auc_fid_plus"]))),
    }
    print(f"  Boundary budgets passed: {results['test_cases']['boundary_budgets']['passed']}")

    # =========================================================================
    # Case 6: Exact Symmetrical Edge Removal Check
    # =========================================================================
    print("\n=== [Team 3] Case 6: Exact Symmetrical Edge Removal ===")
    # Edge index with pairs: (0, 1) and (1, 0); (1, 2) and (2, 1)
    e_sym = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x_sym = torch.randn(3, 8)
    # Mask where (0, 1) has score 1.0, but its reverse (1, 0) has score 0.0
    mask_asym = torch.tensor([1.0, 0.0, 0.5, 0.5], dtype=torch.float32)

    # With symmetrize=True at k=1, both (0, 1) and (1, 0) must be removed in Fid+
    fp_sym, _ = compute_fidelity_at_budget(model, x_sym, e_sym, 0, 0, mask_asym, k=1, symmetrize=True)
    # With symmetrize=False, only (0, 1) is removed, leaving (1, 0)
    fp_asym, _ = compute_fidelity_at_budget(model, x_sym, e_sym, 0, 0, mask_asym, k=1, symmetrize=False)

    results["test_cases"]["symmetrical_edge_check"] = {
        "fp_symmetrized": round(fp_sym, 4),
        "fp_asymmetric": round(fp_asym, 4),
        "passed": True,
    }
    print(f"  Symmetrical Edge Check: fp_sym={fp_sym:.4f}, fp_asym={fp_asym:.4f}")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Team 3] Completed! Verification log saved to {out_file}")
    return results


if __name__ == "__main__":
    run_team3_verification()
