# Source Code Architecture (`src/`)

> **Compliance Standard:** Scientific Computing Best Practices (Wilson et al., 2017) & NeurIPS/ICLR Code Reproducibility Standards.  
> **Topic Profile:** SNA Topic C.10 — Making Graph Machine Learning Models Explainable (UIT 2026).

---

## 1. Modular Package Taxonomy

```
src/
├── config/                  # Centralized Hyperparameters & Seed Management
│   ├── settings.py          # Fixed 5 random seeds [42, 123, 456, 789, 999], device & paths
│   └── __init__.py
│
├── datasets/                # Data Ingestion & Anti-Leakage Partitioning Service
│   ├── base_loader.py       # Abstract BaseDatasetLoader interface & summary stats
│   ├── anti_leakage_splitter.py # Strict pre-processing train/val/test isolation
│   ├── ba_shapes.py         # Synthetic motif ground truth graph (700 nodes)
│   ├── cora.py              # Homophilic citation network (2,708 nodes)
│   ├── amazon_photo.py      # Dense co-purchase e-commerce network (7,650 nodes)
│   ├── facebook_page.py     # Verified human social network (22,470 nodes, SNA profile)
│   ├── ogbn_arxiv.py        # Directed citation network with temporal split (169,343 nodes)
│   ├── twitch.py            # Real gamer social graph
│   ├── twibot20.py          # Social bot detection benchmark
│   └── __init__.py
│
├── models/                  # GNN Modeling Service & Backbone Architectures
│   ├── base_gnn.py          # Abstract BaseGNN with save/load checkpoints
│   ├── gcn.py               # Graph Convolutional Network (Kipf & Welling, 2017)
│   ├── gat.py               # Graph Attention Network (Veličković et al., 2018)
│   ├── graphsage.py         # GraphSAGE Neighborhood Sampler (Hamilton et al., 2017)
│   └── __init__.py
│
├── explainers/              # Explainability Engine & Baseline Attribution
│   ├── base_explainer.py    # Abstract BaseExplainer & ExplanationResult container
│   ├── gnn_explainer.py     # Optimization-based mutual information mask (Ying et al.)
│   ├── pg_explainer.py      # Parametric global explainer (Luo et al.)
│   ├── saliency.py          # Deterministic first-order gradient attribution (Captum)
│   ├── random_explainer.py  # Uniform random noise-floor baseline
│   ├── nearest_first_explainer.py # Topological distance heuristic baseline
│   ├── subgraphx.py         # MCTS Monte-Carlo Tree Search explainer
│   ├── eig_search.py        # Empirical Information Gain search
│   ├── lt_def.py            # Latent Topology Diffusion explainer
│   └── __init__.py
│
├── evaluation/              # Scientific Metrics & Disagreement Quantification
│   ├── fidelity.py          # Fid+ (unfaithfulness drop), Fid- (necessity) & AUC curves
│   ├── agreement.py         # Chance-corrected Kappa@k, Jaccard, Rank tau & Ceiling
│   ├── k_star.py            # Monotonic minimal edge budget k* to retain prediction
│   ├── variance_decomposition.py # 3-way ANOVA variance attribution across sources
│   ├── statistical_tests.py # Holm-Bonferroni correction & TOST equivalence tests
│   ├── metrics.py           # Core metric calculations (Sparsity, ROC-AUC)
│   ├── benchmark_evaluator.py # Multi-seed aggregate benchmark runner
│   └── __init__.py
│
├── nle_grounding/           # Graph-to-Text Grounding (Secondary / Scoped Out)
│   ├── graph_serializer.py  # Subgraph motif serialization
│   ├── verbalizer.py        # Natural language template rendering
│   └── rejection_filter.py  # Faithfulness hallucination rejection
│
├── visualization/           # Visualization & Interactive Demo Service
│   ├── graph_plotter.py     # High-resolution vector & raster motif rendering
│   └── app_demo.py          # Interactive Streamlit Web Demonstration
│
└── pipeline/                # End-to-End Orchestrated Pipeline
    ├── run_pipeline.py      # Unified CLI execution orchestrator
    └── update_graphify.py   # Dual AST Knowledge Graph tracking tool
```

---

## 2. Architectural Invariants & Data Flow

1. **Strict Decoupling:** Every module has single responsibility and communicates strictly via typed interfaces (`ExplanationResult`, `Data`).
2. **Anti-Leakage Splitting:** Datasets are partitioned into Train / Validation / Test **before** any feature scaling or graph transformation.
3. **The Disagreement Protocol:** All explainers are evaluated on the exact same candidate receptive field ($L$-hop message-passing subgraph), evaluated against both `RandomExplainer` and `NearestFirstExplainer` baselines.
4. **Reproducibility Guarantee:** Random seeds are fixed across `[42, 123, 456, 789, 999]`. All metric aggregations report $\mu \pm \sigma$.
