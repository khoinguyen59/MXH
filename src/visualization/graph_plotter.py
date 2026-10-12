"""Graph Visualization Module.

Generates high-resolution publication plots and interactive HTML graphs highlighting
explanation subgraphs, salient motif edges, and node importance scores.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.explainers.base_explainer import ExplanationResult


class GraphPlotter:
    """Renders static and interactive graph explanations."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize graph plotter.

        Args:
            output_dir: Optional directory to save generated figure plots.
        """
        self.output_dir = output_dir or Path("product/reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_explanation(
        self,
        explanation: ExplanationResult,
        full_edge_index: torch.Tensor,
        num_nodes: int,
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Render a publication-ready figure of the k-hop ego-graph with highlighted explanation edges.

        Args:
            explanation: ExplanationResult container.
            full_edge_index: Original graph connectivity.
            num_nodes: Total node count.
            save_path: Optional file path to save PNG figure.
            title: Optional plot title.

        Returns:
            Matplotlib Figure object.
        """
        target_node = explanation.target_node
        sub_nodes = set(explanation.subgraph_nodes.tolist())
        exp_edge_tuples = set()

        if explanation.subgraph_edge_index.numel() > 0:
            for i in range(explanation.subgraph_edge_index.size(1)):
                u = int(explanation.subgraph_edge_index[0, i].item())
                v = int(explanation.subgraph_edge_index[1, i].item())
                exp_edge_tuples.add((u, v))
                exp_edge_tuples.add((v, u))

        # Build local NetworkX graph for plotting
        g = nx.Graph()
        g.add_nodes_from(sub_nodes)

        # Add all edges within sub_nodes from full_edge_index
        background_edges = []
        highlight_edges = []

        for i in range(full_edge_index.size(1)):
            u = int(full_edge_index[0, i].item())
            v = int(full_edge_index[1, i].item())
            if u in sub_nodes and v in sub_nodes and u < v:
                if (u, v) in exp_edge_tuples or (v, u) in exp_edge_tuples:
                    highlight_edges.append((u, v))
                else:
                    background_edges.append((u, v))

        g.add_edges_from(background_edges)
        g.add_edges_from(highlight_edges)

        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        pos = nx.spring_layout(g, seed=42)

        # 1. Draw background edges (grey, dashed/thin)
        nx.draw_networkx_edges(
            g, pos, edgelist=background_edges, edge_color="#D1D5DB", width=1.0, alpha=0.6, ax=ax
        )

        # 2. Draw salient explanation edges (crimson red, bold)
        nx.draw_networkx_edges(
            g, pos, edgelist=highlight_edges, edge_color="#DC2626", width=2.8, alpha=0.95, ax=ax
        )

        # 3. Draw neighbor nodes
        neighbor_nodes = [n for n in g.nodes() if n != target_node]
        nx.draw_networkx_nodes(
            g, pos, nodelist=neighbor_nodes, node_color="#3B82F6", node_size=240, alpha=0.9, ax=ax
        )

        # 4. Draw target node with prominent gold/amber styling
        nx.draw_networkx_nodes(
            g, pos, nodelist=[target_node], node_color="#F59E0B", node_size=550, edgecolors="#111827", linewidths=2.0, ax=ax
        )

        # 5. Draw node labels
        labels = {n: str(n) for n in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels, font_size=8, font_family="sans-serif", ax=ax)

        ax_title = title or f"Explanation Subgraph for Node_{target_node} (Class {explanation.predicted_class})"
        ax.set_title(ax_title, fontsize=12, fontweight="bold", pad=12)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")

        return fig
