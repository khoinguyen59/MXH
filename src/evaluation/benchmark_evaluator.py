"""Benchmark Evaluator Module.

Coordinates multi-seed explanation evaluation across test nodes, computes aggregate
statistics (mean +/- std), and generates publication-grade comparative reports.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.evaluation.metrics import (
    compute_fidelity_minus,
    compute_fidelity_plus,
    compute_motif_roc_auc,
    compute_sparsity,
)
from src.explainers.base_explainer import BaseExplainer


class BenchmarkEvaluator:
    """Evaluates one or more GNN explainers over target graph test nodes."""

    def __init__(
        self,
        model: torch.nn.Module,
        data: Data,
        threshold: float = 0.5,
    ) -> None:
        """Initialize benchmark evaluator.

        Args:
            model: Trained GNN model.
            data: Graph dataset containing test_mask, x, edge_index, y.
            threshold: Saliency threshold for edge mask binarization.
        """
        self.model = model
        self.data = data
        self.threshold = threshold

    def evaluate_explainer(
        self,
        explainer: BaseExplainer,
        target_nodes: List[int],
        ground_truth_motif_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate an explainer across a list of target test nodes.

        Args:
            explainer: BaseExplainer instance.
            target_nodes: List of node indices to explain.
            ground_truth_motif_mask: Optional ground-truth edge mask for Motif-AUC.

        Returns:
            Dictionary with aggregated mean scores: Fidelity+, Fidelity-, Sparsity, Runtime, Motif-AUC.
        """
        fid_plus_list: List[float] = []
        fid_minus_list: List[float] = []
        sparsity_list: List[float] = []
        runtime_list: List[float] = []
        auc_list: List[float] = []

        for node_idx in target_nodes:
            res = explainer.explain_node(node_idx, self.data.x, self.data.edge_index)
            sub_nodes, sub_edge_index, mapping, edge_mask_bool = (
                explainer.get_k_hop_computation_graph(node_idx, self.data.edge_index, self.data.num_nodes)
            )
            local_target = int(mapping.item())
            sub_x = self.data.x[sub_nodes]

            # 1. Fidelity+
            f_plus = compute_fidelity_plus(
                self.model,
                sub_x,
                sub_edge_index,
                local_target,
                res.predicted_class,
                res.edge_mask,
                self.threshold,
            )
            fid_plus_list.append(f_plus)

            # 2. Fidelity-
            f_minus = compute_fidelity_minus(
                self.model,
                sub_x,
                sub_edge_index,
                local_target,
                res.predicted_class,
                res.edge_mask,
                self.threshold,
            )
            fid_minus_list.append(f_minus)

            # 3. Sparsity
            sp = compute_sparsity(res.edge_mask, self.threshold)
            sparsity_list.append(sp)

            # 4. Runtime
            runtime_list.append(res.execution_time)

            # 5. Motif-AUC (if ground truth available)
            if ground_truth_motif_mask is not None:
                sub_gt = ground_truth_motif_mask[edge_mask_bool]
                auc = compute_motif_roc_auc(res.edge_mask, sub_gt)
                if auc is not None:
                    auc_list.append(auc)

        results = {
            "fidelity_plus_mean": float(np.mean(fid_plus_list)),
            "fidelity_plus_std": float(np.std(fid_plus_list)),
            "fidelity_minus_mean": float(np.mean(fid_minus_list)),
            "fidelity_minus_std": float(np.std(fid_minus_list)),
            "sparsity_mean": float(np.mean(sparsity_list)),
            "sparsity_std": float(np.std(sparsity_list)),
            "runtime_mean_sec": float(np.mean(runtime_list)),
            "runtime_std_sec": float(np.std(runtime_list)),
            "motif_auc_mean": float(np.mean(auc_list)) if auc_list else None,
            "motif_auc_std": float(np.std(auc_list)) if auc_list else None,
        }

        return results

    @staticmethod
    def format_results_table(summary_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Format explainer benchmark dictionary into a presentation DataFrame.

        Args:
            summary_dict: Mapping of explainer_name -> metrics_dict.

        Returns:
            Pandas DataFrame formatted with mean +/- std strings.
        """
        rows = []
        for name, metrics in summary_dict.items():
            f_plus = f"{metrics['fidelity_plus_mean']:.3f} +/- {metrics['fidelity_plus_std']:.3f}"
            f_minus = f"{metrics['fidelity_minus_mean']:.3f} +/- {metrics['fidelity_minus_std']:.3f}"
            sparsity = f"{metrics['sparsity_mean']:.3f} +/- {metrics['sparsity_std']:.3f}"
            runtime = f"{metrics['runtime_mean_sec']:.4f}s"
            auc = (
                f"{metrics['motif_auc_mean']:.3f} +/- {metrics['motif_auc_std']:.3f}"
                if metrics["motif_auc_mean"] is not None
                else "N/A"
            )

            rows.append({
                "Explainer": name,
                "Fidelity+ (Higher)": f_plus,
                "Fidelity- (Lower)": f_minus,
                "Sparsity": sparsity,
                "Motif-AUC": auc,
                "Time / Node (sec)": runtime,
            })

        return pd.DataFrame(rows)
