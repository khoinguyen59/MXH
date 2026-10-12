"""Interactive Web Application Demo for GNN Explainability (Streamlit).

Allows researchers and practitioners to inspect GNN predictions, visualize explanation
subgraphs, benchmark quantitative metrics, and generate grounded natural language explanations.
"""

from pathlib import Path
import streamlit as st
import torch

from src.datasets.ba_shapes import BAShapesLoader
from src.datasets.cora import CoraLoader
from src.evaluation.metrics import compute_fidelity_minus, compute_fidelity_plus, compute_sparsity
from src.explainers.eig_search import EigSearchExplainer
from src.explainers.gnn_explainer import GNNExplainerWrapper
from src.explainers.lt_def import LTDEFExplainer
from src.explainers.subgraphx import SubgraphXExplainer
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.nle_grounding.rejection_filter import RejectionFilter
from src.nle_grounding.verbalizer import NLEVerbalizer
from src.visualization.graph_plotter import GraphPlotter

st.set_page_config(
    page_title="SNA C.10: Explainable Graph Machine Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_dataset_and_model(dataset_name: str, model_name: str):
    """Cache dataset loading and model initialization."""
    data_dir = Path("data")
    if dataset_name == "BA-Shapes (Synthetic Motifs)":
        loader = BAShapesLoader(data_dir=str(data_dir), num_base_nodes=200, num_motifs=50, random_seed=42)
    else:
        loader = CoraLoader(data_dir=str(data_dir), random_seed=42)

    data = loader.load()

    num_features = data.num_features
    num_classes = int(data.y.max().item()) + 1

    if model_name == "GraphSAGE":
        model = GraphSAGE(num_features, 64, num_classes)
    else:
        model = GCN(num_features, 64, num_classes)

    # Train model lightly for interactive demo if not pre-trained
    model.fit_model(data, epochs=30, lr=0.01)
    return loader, data, model


def main():
    st.title("Explainable Graph Machine Learning Engine")
    st.caption("Course: Social Network Analysis (UIT 2026) | Topic C.10 | Supervisor: TS. Tran Hung Nghiep")

    st.sidebar.header("Configuration Panel")
    dataset_name = st.sidebar.selectbox(
        "Dataset Benchmark",
        ["BA-Shapes (Synthetic Motifs)", "Cora Citation Network"],
    )
    model_name = st.sidebar.selectbox(
        "GNN Backbone",
        ["GCN", "GraphSAGE"],
    )
    explainer_name = st.sidebar.selectbox(
        "Explainer Algorithm",
        ["LT-DEF (Proposed Dual-Channel)", "EiG-Search (Linear-Time)", "GNNExplainer", "SubgraphX (MCTS)"],
    )

    loader, data, model = load_dataset_and_model(dataset_name, model_name)

    test_indices = torch.where(data.test_mask)[0].tolist()
    default_node = test_indices[0] if test_indices else 0
    node_idx = st.sidebar.number_input(
        "Target Test Node Index",
        min_value=0,
        max_value=data.num_nodes - 1,
        value=default_node,
        step=1,
    )

    if st.sidebar.button("Generate Explanation", type="primary"):
        with st.spinner("Extracting explanation subgraph and computing metrics..."):
            if "LT-DEF" in explainer_name:
                explainer = LTDEFExplainer(model=model, num_hops=2, sparsity_target=0.70)
            elif "EiG-Search" in explainer_name:
                explainer = EigSearchExplainer(model=model, num_hops=2, sparsity_target=0.70)
            elif "SubgraphX" in explainer_name:
                explainer = SubgraphXExplainer(model=model, num_hops=2, rollout_steps=10)
            else:
                explainer = GNNExplainerWrapper(model=model, epochs=50)

            explanation = explainer.explain_node(node_idx, data.x, data.edge_index)

            # Compute quantitative metrics
            sub_nodes, sub_edge_index, mapping, edge_mask_bool = (
                explainer.get_k_hop_computation_graph(node_idx, data.edge_index, data.num_nodes)
            )
            local_target = int(mapping.item())
            sub_x = data.x[sub_nodes]

            fid_plus = compute_fidelity_plus(
                model, sub_x, sub_edge_index, local_target, explanation.predicted_class, explanation.edge_mask
            )
            fid_minus = compute_fidelity_minus(
                model, sub_x, sub_edge_index, local_target, explanation.predicted_class, explanation.edge_mask
            )
            sparsity = compute_sparsity(explanation.edge_mask)

            # Generate verbalization
            verbalizer = NLEVerbalizer()
            nle_text = verbalizer.generate_explanation(explanation, data.x)

            # Audit against hallucination
            rejection_filter = RejectionFilter()
            is_accepted, hall_rate, violations = rejection_filter.audit_explanation(nle_text, explanation)

        # Main Layout
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader(f"Subgraph Visualization (Node {node_idx})")
            plotter = GraphPlotter()
            fig = plotter.plot_explanation(explanation, data.edge_index, data.num_nodes)
            st.pyplot(fig)

        with col2:
            st.subheader("Quantitative Evaluation")
            m1, m2 = st.columns(2)
            m1.metric("Fidelity+ (Higher=Better)", f"{fid_plus:.3f}")
            m2.metric("Fidelity- (Lower=Better)", f"{fid_minus:.3f}")

            m3, m4 = st.columns(2)
            m3.metric("Sparsity", f"{sparsity:.1%}")
            m4.metric("Time", f"{explanation.execution_time:.4f}s")

            st.divider()
            st.subheader("Natural Language Explanation (NLE)")
            st.info(nle_text)

            if is_accepted:
                st.success(f"Audit Verified: Zero Hallucination ({hall_rate:.1%})")
            else:
                st.warning(f"Hallucination Warning: {violations}")


if __name__ == "__main__":
    main()
