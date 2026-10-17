"""Runner script for Khôi's Task T02: Sample Explainer Run on BA-Shapes.

Executes GNNExplainer and Saliency on sample validation nodes and exports:
- edge_scores.csv (BG10)
- run_metadata.json (BG11)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import torch.nn.functional as F

from src.datasets.ba_shapes import BAShapesLoader
from src.models.gcn import GCN
from src.explainers.edge_explainer_service import (
    get_candidate_edges,
    explain_edges_gnn_explainer,
    explain_edges_saliency,
    export_explanation_results,
)

sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 60)
    print("TASK T02 (KHÔI) - SAMPLE EXPLAINER RUN ON BA-SHAPES")
    print("=" * 60)

    # 1. Load BA-Shapes data
    data_dir = "data/ba_shapes"
    loader = BAShapesLoader(data_dir=data_dir, random_seed=42)
    data = loader.load()
    print(f"Loaded BA-Shapes: {data.num_nodes} nodes, {data.edge_index.size(1)} directed edges")

    # 2. Load trained GCN checkpoint or train quickly
    ckpt_path = "checkpoints/gcn_ba_shapes_seed42.pt"
    in_channels = data.x.size(1)
    hidden_channels = 16
    out_channels = int(data.y.max().item()) + 1
    model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

    if os.path.exists(ckpt_path):
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(saved, dict) and "state_dict" in saved:
            state_dict = saved["state_dict"]
            hidden_channels = saved.get("hidden_channels", hidden_channels)
            model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)
        else:
            state_dict = saved
        model.load_state_dict(state_dict)
        print(f"Loaded GCN checkpoint from {ckpt_path}")
    else:
        print("Checkpoint not found, fitting model for 30 epochs...")
        model.fit_model(data, epochs=30, lr=0.01)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved GCN checkpoint to {ckpt_path}")

    model.eval()

    # Verify model accuracy on validation set
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        val_acc = (preds[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        print(f"GCN Validation Accuracy: {val_acc * 100:.2f}%")

    # 3. Select 5 sample validation nodes (prioritizing motif nodes per our audit recommendation!)
    val_indices = torch.where(data.val_mask)[0]
    motif_val_nodes = [idx.item() for idx in val_indices if data.y[idx].item() in [1, 2]]
    base_val_nodes = [idx.item() for idx in val_indices if data.y[idx].item() == 0]

    # Select 3 motif nodes and 2 base nodes
    selected_nodes = motif_val_nodes[:3] + base_val_nodes[:2]
    print(f"\nSelected 5 sample validation nodes for T02: {selected_nodes}")
    for n in selected_nodes:
        pred_c = int(preds[n].item())
        true_c = int(data.y[n].item())
        prob = float(probs[n, pred_c].item())
        print(f"  - Node {n}: True Class = {true_c}, Pred Class = {pred_c}, Prob = {prob:.4f}")

    # 4. Run Explainers
    records = []
    checkpoint_id = "gcn_ba_shapes_seed42"

    for n in selected_nodes:
        pred_c = int(preds[n].item())
        prob = float(probs[n, pred_c].item())
        cand_edges, sub_edge_index, _ = get_candidate_edges(n, data.edge_index, data.num_nodes, num_hops=2)
        m = len(cand_edges)
        print(f"\n[Node {n}] Candidate edges (undirected, 2-hop, no self-loops): {m}")

        # Run GNNExplainer
        run_id_gnn = f"ba_shapes_node{n}_gnnexplainer"
        print(f"  -> Running GNNExplainer (epochs=50)...", end=" ", flush=True)
        res_gnn = explain_edges_gnn_explainer(
            model=model,
            data=data,
            node_id=n,
            target_class=pred_c,
            candidate_edges=cand_edges,
            config={"epochs": 50, "lr": 0.02, "num_hops": 2},
            seed=42
        )
        print(f"Done in {res_gnn['runtime_sec']:.3f}s (status: {res_gnn['status']})")

        records.append({
            "run_id": run_id_gnn,
            "dataset": "BA-Shapes",
            "node_id": n,
            "target_class": pred_c,
            "predicted_class": pred_c,
            "original_prob": prob,
            "checkpoint_id": checkpoint_id,
            "model_architecture": "GCN-2L",
            "seed_gnn": 42,
            "explainer_method": "GNNExplainer",
            "seed_explainer": 42,
            "config": {"epochs": 50, "lr": 0.02, "num_hops": 2},
            "candidate_edges": cand_edges,
            "scores": res_gnn["scores"],
            "runtime_sec": res_gnn["runtime_sec"],
            "status": res_gnn["status"],
            "error_message": res_gnn["error_message"]
        })

        # Run Saliency
        run_id_sal = f"ba_shapes_node{n}_saliency"
        print(f"  -> Running Saliency (abs gradient)...", end=" ", flush=True)
        res_sal = explain_edges_saliency(
            model=model,
            data=data,
            node_id=n,
            target_class=pred_c,
            candidate_edges=cand_edges,
            config={"num_hops": 2},
            seed=42
        )
        print(f"Done in {res_sal['runtime_sec']:.3f}s (status: {res_sal['status']})")

        records.append({
            "run_id": run_id_sal,
            "dataset": "BA-Shapes",
            "node_id": n,
            "target_class": pred_c,
            "predicted_class": pred_c,
            "original_prob": prob,
            "checkpoint_id": checkpoint_id,
            "model_architecture": "GCN-2L",
            "seed_gnn": 42,
            "explainer_method": "Saliency",
            "seed_explainer": 42,
            "config": {"num_hops": 2},
            "candidate_edges": cand_edges,
            "scores": res_sal["scores"],
            "runtime_sec": res_sal["runtime_sec"],
            "status": res_sal["status"],
            "error_message": res_sal["error_message"]
        })

    # 5. Export results
    out_dir = "experiments/pilot_study/results"
    scores_path, meta_path = export_explanation_results(records, output_dir=out_dir)
    print("\n" + "=" * 60)
    print("EXPORT SUCCESSFUL (BG10 & BG11 COMPLIANT)")
    print(f"  - Edge scores CSV: {scores_path}")
    print(f"  - Run metadata JSON: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
