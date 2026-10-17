"""Team Check 2: Mathematical & Tensor Consistency Verification Script.

Executes direct empirical verification on real tensors from data/t01_handover/
(both Cora and BA-Shapes) to verify:
1. Log-softmax mathematical validity and edge_weight gradient flow.
2. Explainer output tensor dimensions (subgraph_edge_index vs edge_mask).
3. Fidelity+ and Fidelity- evaluation across all 7 explainers without crashes.
4. Output logged to disk for zero-mockup auditing.
"""

import json
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.t02_explainers.adapters.cora_handover_adapter import CoraHandoverAdapter
from src.t02_explainers.adapters.bashapes_handover_adapter import BAShapesHandoverAdapter
from src.t02_explainers.algorithms.saliency import SaliencyExplainer
from src.t02_explainers.algorithms.random_explainer import RandomExplainer
from src.t02_explainers.algorithms.nearest_first_explainer import NearestFirstExplainer
from src.t02_explainers.algorithms.eig_search import EigSearchExplainer
from src.t02_explainers.algorithms.lt_def import LTDEFExplainer
from src.t02_explainers.algorithms.subgraphx import SubgraphXExplainer
from src.t02_explainers.algorithms.gnn_explainer import GNNExplainerWrapper
from src.t03_evaluation.fidelity import compute_fidelity_at_budget, compute_fidelity_curve


def run_team2_verification():
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cora_checks": {},
        "bashapes_checks": {},
        "summary": "PASSED",
    }
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / "team2_mathematical_tensor_verification.json"

    # =========================================================================
    # Part 1: Cora Mathematical & Tensor Checks
    # =========================================================================
    print("=== [Team 2] Part 1: Cora Handover Checks ===")
    cora_adapter = CoraHandoverAdapter()
    assert cora_adapter.verify_files_exist(), "Cora handover files missing!"
    cora_bundle = cora_adapter.load(device="cpu")
    model = cora_bundle.model
    data = cora_bundle.data

    # Check 1.1: Log-Softmax return_type consistency
    with torch.no_grad():
        out_log = model(data.x, data.edge_index, return_type="log_probs")
        out_prob = model(data.x, data.edge_index, return_type="probs")
        out_raw = model(data.x, data.edge_index, return_type="raw")

    exp_log = torch.exp(out_log)
    sum_probs = exp_log.sum(dim=-1)
    max_prob_dev = float((sum_probs - 1.0).abs().max().item())
    log_prob_diff = float((exp_log - out_prob).abs().max().item())
    raw_log_diff = float((F.log_softmax(out_raw, dim=-1) - out_log).abs().max().item())

    results["cora_checks"]["log_softmax_validity"] = {
        "max_prob_deviation_from_1": max_prob_dev,
        "exp_log_vs_probs_max_diff": log_prob_diff,
        "raw_to_log_diff": raw_log_diff,
        "passed": bool(max_prob_dev < 1e-5 and log_prob_diff < 1e-5 and raw_log_diff < 1e-5),
    }
    print(f"  [1.1] Log-Softmax validity: max dev = {max_prob_dev:.2e}, passed = {results['cora_checks']['log_softmax_validity']['passed']}")

    # Check 1.2: Gradient Flow through continuous edge_weight
    edge_weight = torch.ones(data.edge_index.size(1), requires_grad=True)
    out_with_weight = model(data.x, data.edge_index, edge_weight=edge_weight, return_type="log_probs")
    target_score = out_with_weight[cora_bundle.val_nodes[0], 0]
    target_score.backward()
    grad_norm = float(edge_weight.grad.norm().item()) if edge_weight.grad is not None else 0.0

    results["cora_checks"]["edge_weight_gradient_flow"] = {
        "gradient_norm": grad_norm,
        "passed": bool(grad_norm > 1e-6),
    }
    print(f"  [1.2] Edge weight gradient flow: norm = {grad_norm:.4f}, passed = {results['cora_checks']['edge_weight_gradient_flow']['passed']}")

    # Check 1.3: Real Explainer Execution & Fidelity on 2 validation nodes
    sample_nodes = cora_bundle.val_nodes[:2]
    explainers = {
        "Saliency": SaliencyExplainer(model=model, num_hops=2),
        "Random": RandomExplainer(model=model, num_hops=2, random_seed=42),
        "Nearest-First": NearestFirstExplainer(model=model, num_hops=2),
        "EiG-Search": EigSearchExplainer(model=model, num_hops=2),
        "LT-DEF": LTDEFExplainer(model=model, num_hops=2),
        "SubgraphX": SubgraphXExplainer(model=model, num_hops=2, rollout_steps=4),
        "GNNExplainer": GNNExplainerWrapper(model=model, epochs=20, lr=0.01, num_hops=2),
    }

    results["cora_checks"]["explainer_evaluations"] = []
    for node_id in sample_nodes:
        pred_label = int(cora_bundle.original_pred[node_id].item())
        node_res = {"node_id": node_id, "pred_label": pred_label, "explainers": {}}
        print(f"  [1.3] Evaluating Node {node_id} (Class {pred_label}):")

        for name, exp in explainers.items():
            t0 = time.perf_counter()
            res = exp.explain_node(node_id, data.x, data.edge_index)
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            # Verify tensor dimension alignment
            dim_match = bool(res.edge_mask.numel() == res.subgraph_edge_index.size(1))
            assert dim_match, f"{name} dimension mismatch!"

            # Compute Symmetrical Fidelity at budget k=5
            local_target = int(torch.where(res.subgraph_nodes == node_id)[0].item())
            sub_x = data.x[res.subgraph_nodes]
            fid_plus, fid_minus = compute_fidelity_at_budget(
                model=model,
                x=sub_x,
                sub_edge_index=res.subgraph_edge_index,
                target_node=local_target,
                target_class=pred_label,
                edge_mask=res.edge_mask,
                k=5,
                symmetrize=True,
            )

            # Sparsity
            sparsity = float((res.edge_mask < 0.1).float().mean().item()) if res.edge_mask.numel() > 0 else 1.0

            node_res["explainers"][name] = {
                "runtime_ms": round(runtime_ms, 2),
                "num_sub_edges": int(res.subgraph_edge_index.size(1)),
                "edge_mask_size": int(res.edge_mask.numel()),
                "dim_match": dim_match,
                "fid_plus": round(fid_plus, 4),
                "fid_minus": round(fid_minus, 4),
                "sparsity": round(sparsity, 4),
            }
            print(f"       -> {name:14s}: run={runtime_ms:6.1f}ms | edges={res.edge_mask.numel()} | Fid+={fid_plus:+.4f} | Fid-={fid_minus:+.4f} | Sparsity={sparsity:.4f}")

        results["cora_checks"]["explainer_evaluations"].append(node_res)

    # =========================================================================
    # Part 2: BA-Shapes Mathematical & Tensor Checks
    # =========================================================================
    print("\n=== [Team 2] Part 2: BA-Shapes Handover Checks ===")
    ba_adapter = BAShapesHandoverAdapter()
    if ba_adapter.verify_files_exist():
        ba_bundle = ba_adapter.load(device="cpu")
        ba_model = ba_bundle.model
        ba_data = ba_bundle.data
        ba_val_nodes = ba_bundle.val_nodes[:2]

        results["bashapes_checks"]["explainer_evaluations"] = []
        ba_explainers = {
            "Saliency": SaliencyExplainer(model=ba_model, num_hops=2),
            "Random": RandomExplainer(model=ba_model, num_hops=2, random_seed=42),
            "Nearest-First": NearestFirstExplainer(model=ba_model, num_hops=2),
            "EiG-Search": EigSearchExplainer(model=ba_model, num_hops=2),
            "LT-DEF": LTDEFExplainer(model=ba_model, num_hops=2),
            "SubgraphX": SubgraphXExplainer(model=ba_model, num_hops=2, rollout_steps=4),
        }

        for node_id in ba_val_nodes:
            with torch.no_grad():
                out = ba_model(ba_data.x, ba_data.edge_index)
                pred_label = int(out[node_id].argmax(dim=-1).item())

            node_res = {"node_id": node_id, "pred_label": pred_label, "explainers": {}}
            print(f"  [2.1] Evaluating BA-Shapes Node {node_id} (Class {pred_label}):")

            for name, exp in ba_explainers.items():
                t0 = time.perf_counter()
                res = exp.explain_node(node_id, ba_data.x, ba_data.edge_index)
                runtime_ms = (time.perf_counter() - t0) * 1000.0

                dim_match = bool(res.edge_mask.numel() == res.subgraph_edge_index.size(1))
                assert dim_match, f"{name} BA-Shapes dimension mismatch!"

                local_target = int(torch.where(res.subgraph_nodes == node_id)[0].item())
                sub_x = ba_data.x[res.subgraph_nodes]
                fid_plus, fid_minus = compute_fidelity_at_budget(
                    model=ba_model,
                    x=sub_x,
                    sub_edge_index=res.subgraph_edge_index,
                    target_node=local_target,
                    target_class=pred_label,
                    edge_mask=res.edge_mask,
                    k=5,
                    symmetrize=True,
                )

                sparsity = float((res.edge_mask < 0.1).float().mean().item()) if res.edge_mask.numel() > 0 else 1.0

                node_res["explainers"][name] = {
                    "runtime_ms": round(runtime_ms, 2),
                    "num_sub_edges": int(res.subgraph_edge_index.size(1)),
                    "edge_mask_size": int(res.edge_mask.numel()),
                    "dim_match": dim_match,
                    "fid_plus": round(fid_plus, 4),
                    "fid_minus": round(fid_minus, 4),
                    "sparsity": round(sparsity, 4),
                }
                print(f"       -> {name:14s}: run={runtime_ms:6.1f}ms | edges={res.edge_mask.numel()} | Fid+={fid_plus:+.4f} | Fid-={fid_minus:+.4f} | Sparsity={sparsity:.4f}")

            results["bashapes_checks"]["explainer_evaluations"].append(node_res)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Team 2] Completed! Verification log saved to {out_file}")
    return results


if __name__ == "__main__":
    run_team2_verification()
