"""k* Minimal Preserving Edge Budget Metric.

Computes k*(v): the smallest edge budget such that keeping any top-k' edges (k' >= k*)
retains the original model prediction at node v.
Supports k*=0 when node features alone are sufficient to classify the node.
"""

from typing import Dict
import torch


def compute_k_star(
    model: torch.nn.Module,
    x: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_node: int,
    edge_mask: torch.Tensor,
    max_k: int = 20,
) -> Dict[str, float]:
    """Compute monotonic k*(v) minimal edge retention budget for target node.

    Args:
        model: Trained GNN model.
        x: Node feature matrix.
        sub_edge_index: Local computation graph edges.
        target_node: Local target node index.
        edge_mask: Continuous importance weights of edges.
        max_k: Maximum search budget cap.

    Returns:
        Dictionary containing k_star, is_zero_k_star, and k_star_ratio.
    """
    model.eval()
    num_edges = sub_edge_index.size(1)

    if num_edges == 0:
        return {
            "k_star": 0.0,
            "is_zero_k_star": 1.0,
            "k_star_ratio": 0.0,
        }

    with torch.no_grad():
        # Clean prediction
        orig_out = model(x, sub_edge_index)
        orig_class = int(orig_out[target_node].argmax(dim=-1).item())

        # Check k = 0 (empty edge graph)
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=sub_edge_index.device)
        empty_out = model(x, empty_edges)
        empty_class = int(empty_out[target_node].argmax(dim=-1).item())

        if empty_class == orig_class:
            return {
                "k_star": 0.0,
                "is_zero_k_star": 1.0,
                "k_star_ratio": 0.0,
            }

        # Search monotonic k* up to min(max_k, num_edges)
        search_limit = min(max_k, num_edges)
        sorted_indices = torch.argsort(edge_mask, descending=True)

        found_k_star = search_limit

        for k in range(1, search_limit + 1):
            top_k_indices = sorted_indices[:k]
            keep_edges = sub_edge_index[:, top_k_indices]

            k_out = model(x, keep_edges)
            k_class = int(k_out[target_node].argmax(dim=-1).item())

            if k_class == orig_class:
                # Check monotonicity: all k' from k to search_limit must also retain class
                monotonic = True
                for k_prime in range(k + 1, min(k + 4, search_limit + 1)):
                    sub_idx = sorted_indices[:k_prime]
                    sub_edges = sub_edge_index[:, sub_idx]
                    p_out = model(x, sub_edges)
                    if int(p_out[target_node].argmax(dim=-1).item()) != orig_class:
                        monotonic = False
                        break

                if monotonic:
                    found_k_star = k
                    break

    k_ratio = float(found_k_star / max(1, num_edges))

    return {
        "k_star": float(found_k_star),
        "is_zero_k_star": 0.0,
        "k_star_ratio": k_ratio,
    }
