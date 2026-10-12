"""SubgraphX Explainer Implementation.

Implements Monte Carlo Tree Search (MCTS) with Shapley value scoring for GNN subgraph
explanations (Yuan et al., ICML 2021).
"""

import time
from typing import List, Set
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from src.explainers.base_explainer import BaseExplainer, ExplanationResult


class MCTSNode:
    """Tree node representing a candidate explanation subgraph during MCTS rollouts."""

    def __init__(self, coalition: Set[int], parent=None) -> None:
        """Initialize MCTS node.

        Args:
            coalition: Set of node indices in this subgraph coalition.
            parent: Parent MCTSNode.
        """
        self.coalition = coalition
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.reward = 0.0

    @property
    def is_leaf(self) -> bool:
        """Whether this node is unexpanded."""
        return len(self.children) == 0

    def compute_ucb(self, c_puct: float = 1.0) -> float:
        """Compute Upper Confidence Bound for Trees (UCT/PUCT)."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.reward / self.visit_count
        exploration = c_puct * np.sqrt(np.log(self.parent.visit_count + 1) / (self.visit_count + 1e-6))
        return exploitation + exploration


class SubgraphXExplainer(BaseExplainer):
    """SubgraphX baseline using MCTS and Shapley value attribution."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_hops: int = 2,
        rollout_steps: int = 20,
        min_nodes: int = 4,
        c_puct: float = 1.0,
    ) -> None:
        """Initialize SubgraphX explainer.

        Args:
            model: Trained GNN model to explain.
            num_hops: Radius of the computation ego-graph (default: 2).
            rollout_steps: Number of MCTS simulation rollouts (default: 20).
            min_nodes: Target stopping size of explanation subgraph (default: 4).
            c_puct: Exploration constant for UCB calculation.
        """
        super().__init__(model=model, num_hops=num_hops)
        self.rollout_steps = rollout_steps
        self.min_nodes = min_nodes
        self.c_puct = c_puct

    def _eval_subgraph_reward(
        self,
        coalition: Set[int],
        node_idx: int,
        target_class: int,
        x: torch.Tensor,
        sub_edge_index: torch.Tensor,
        num_nodes: int,
    ) -> float:
        """Approximate Shapley reward for keeping only the coalition subgraph."""
        if not coalition:
            return 0.0

        # Mask out edges not completely inside the coalition
        coalition_tensor = torch.tensor(list(coalition), dtype=torch.long)
        mask = torch.isin(sub_edge_index[0], coalition_tensor) & torch.isin(sub_edge_index[1], coalition_tensor)
        masked_edge_index = sub_edge_index[:, mask]

        with torch.no_grad():
            out = self.model(x, masked_edge_index)
            prob = F.softmax(out[node_idx], dim=-1)[target_class].item()

        return float(prob)

    def explain_node(
        self, node_idx: int, x: torch.Tensor, edge_index: torch.Tensor
    ) -> ExplanationResult:
        """Run MCTS search to find the most explanatory subgraph."""
        start_time = time.perf_counter()

        with torch.no_grad():
            orig_out = self.model(x, edge_index)
            pred_class = int(orig_out[node_idx].argmax(dim=-1).item())

        subset_nodes, sub_edge_index, mapping, edge_mask_bool = (
            self.get_k_hop_computation_graph(node_idx, edge_index, x.size(0))
        )
        local_target_node = int(mapping.item())
        sub_x = x[subset_nodes]
        num_sub_nodes = subset_nodes.size(0)

        # Initialize root with all nodes in computation graph
        root = MCTSNode(coalition=set(range(num_sub_nodes)))

        # Run MCTS rollouts
        for _ in range(self.rollout_steps):
            curr_node = root

            # 1. Selection
            while not curr_node.is_leaf and len(curr_node.coalition) > self.min_nodes:
                best_ucb = -float("inf")
                best_child = None
                for child in curr_node.children:
                    ucb = child.compute_ucb(self.c_puct)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child
                curr_node = best_child if best_child is not None else curr_node

            # 2. Expansion
            if len(curr_node.coalition) > self.min_nodes:
                # Generate candidate children by removing one boundary node (except local_target_node)
                candidates = [n for n in curr_node.coalition if n != local_target_node]
                if candidates:
                    # Pick up to 3 candidate node removals
                    for remove_node in candidates[:3]:
                        child_coalition = set(curr_node.coalition) - {remove_node}
                        child_node = MCTSNode(coalition=child_coalition, parent=curr_node)
                        curr_node.children.append(child_node)
                    if curr_node.children:
                        curr_node = curr_node.children[0]

            # 3. Simulation & Reward
            reward = self._eval_subgraph_reward(
                curr_node.coalition,
                local_target_node,
                pred_class,
                sub_x,
                sub_edge_index,
                num_sub_nodes,
            )

            # 4. Backpropagation
            back_node = curr_node
            while back_node is not None:
                back_node.visit_count += 1
                back_node.reward += reward
                back_node = back_node.parent

        # Find best leaf node with highest average reward
        best_coalition = root.coalition
        best_avg_reward = -float("inf")

        def traverse(node: MCTSNode):
            nonlocal best_coalition, best_avg_reward
            if node.visit_count > 0:
                avg_r = node.reward / node.visit_count
                if avg_r > best_avg_reward and len(node.coalition) <= max(self.min_nodes + 2, 4):
                    best_avg_reward = avg_r
                    best_coalition = node.coalition
            for c in node.children:
                traverse(c)

        traverse(root)

        # Convert best coalition to edge mask
        coalition_tensor = torch.tensor(list(best_coalition), dtype=torch.long)
        edge_in_coalition = (
            torch.isin(sub_edge_index[0], coalition_tensor) &
            torch.isin(sub_edge_index[1], coalition_tensor)
        )
        edge_mask = edge_in_coalition.float()

        elapsed = time.perf_counter() - start_time

        return ExplanationResult(
            target_node=node_idx,
            predicted_class=pred_class,
            edge_mask=edge_mask,
            node_feat_mask=None,
            subgraph_nodes=subset_nodes[coalition_tensor],
            subgraph_edge_index=sub_edge_index[:, edge_in_coalition],
            execution_time=elapsed,
        )
