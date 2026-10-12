# Topic C.10: Making Graph Machine Learning Models Explainable

[![CI Pipeline](https://github.com/khoi-nt/SNA-C10-Explainable-GNN/actions/workflows/ci.yml/badge.svg)](https://github.com/khoi-nt/SNA-C10-Explainable-GNN/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.4+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Course:** Social Network Analysis (SNA) – University of Information Technology (UIT, VNU-HCM), 2026  
> **Supervisor:** Dr. Tran Hung Nghiep (`nghiepth@uit.edu.vn`)  
> **Topic Category:** Level C (Research-Oriented, Difficulty: 4/5)  
> **Team:** Group 10 – XAI-GNN  
> **Members:** Nguyen Trong Khoi (Team Leader) and team  

---

## 1. Executive Summary

Graph Neural Networks (GNNs) achieve state-of-the-art predictive performance on social network graphs (e.g., bot detection, community role classification). However, their multi-hop message-passing architectures operate as opaque black boxes intertwining local topology with high-dimensional node features.

This repository implements the **Linear-Time Dual-Explanation Framework (LT-DEF)**:
1. **Structural Explanation:** Extracts causal edge-induced subgraphs in linear time $\mathcal{O}(|\mathcal{E}|)$ via mutual information optimization, eliminating exponential MCTS bottlenecks.
2. **Semantic Explanation:** Serializes explanation subgraphs via ego-network encoding into local open-weight LLMs, generating human-interpretable natural language rationales validated by a rejection sampling filter to achieve $< 2\%$ hallucination.
3. **Rigorous Academic Benchmarking:** Implements strict data leakage prevention (Special Lecture 01), evaluating across 5 fixed random seeds ($\mu \pm \sigma$) on 4 diverse graph benchmarks.

---

## 2. Decoupled Modular Architecture

```text
c:\Users\Nguyen Trong Khoi\Downloads\MXH/
├── .github/                           # Automated CI/CD (Linting, Anti-Mockup Audit, Pytest)
├── AGENTS.md                          # Master Governance Specification & 12 Invariants
├── README.md                          # Project Overview & Execution Guide
├── requirements.txt                   # Reproducible library versions
│
├── product/                           # CÁC SẢN PHẨM NỘP (Course Deliverables)
│   ├── proposal/                      # 1. Mini Proposal (DOCX, PDF, Markdown & visual audits)
│   ├── slides/                        # 2. Presentation Slide Decks (16:9 5-min pitch & defense)
│   └── reports/                       # 3. Scientific Technical Report (~15-page Springer LNCS)
│
├── docs/                              # TOÀN BỘ TÀI LIỆU DỰ ÁN (Comprehensive Documentation)
│   ├── tai_lieu_goc/                  # Syllabus, lecture slides & template files (.docx, .pdf)
│   ├── tai_lieu_da_export/            # Exported Markdown documentation & topic C.10 specs
│   ├── templates/                     # Official Springer LNCS & HCMUIT thesis templates
│   ├── papers/                        # 14 peer-reviewed research papers (PDF & full Markdown)
│   ├── design/                        # Visual & UI Specification for graph XAI and demo app
│   ├── diary/                         # Daily engineering work logs & AST graph tracking
│   └── research_notes/                # Comparative baseline & methodology notes
│
├── src/                               # SOURCE CODE CHÍNH (Decoupled Python Packages)
│   ├── config/                        # Global paths, seeds [42, 123, 456, 789, 999], settings
│   ├── datasets/                      # Data Ingestion Service & Anti-Leakage Splitters
│   ├── models/                        # GNN Modeling Service (GCN, GraphSAGE, GAT)
│   ├── explainers/                    # Explainability Engine (GNNExplainer, SubgraphX, LT-DEF)
│   ├── evaluation/                    # Quantitative Metric Service (Fidelity, Sparsity, Latency)
│   ├── nle_grounding/                 # Graph-to-Text Serializer & Rejection Sampling Filter
│   ├── visualization/                 # Subgraph Plotter & Streamlit Interactive Demo
│   └── pipeline/                      # End-to-End Automated Pipeline Orchestrator
│
├── experiments/                       # Kịch bản chạy thực nghiệm độc lập
├── tests/                             # Bộ kiểm thử tự động (Unit & Integration tests)
├── data/                              # Dữ liệu đồ thị thực nghiệm (BA-Shapes, Cora, Twitch, TwiBot-20)
├── checkpoints/                       # Trọng số mô hình đã huấn luyện
├── repos/                             # Cloned official GitHub reference repositories
└── scratch/                           # Thư mục chứa các script nháp tạm thời (được gitignore)
```

---

## 3. Quick Start & Execution

### 3.1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/khoi-nt/SNA-C10-Explainable-GNN.git
cd SNA-C10-Explainable-GNN

# Install dependencies
pip install -r requirements.txt
```

### 3.2. Run Unit Tests & Anti-Leakage Verification
```bash
pytest tests/ -v
```

### 3.3. Launch End-to-End Orchestrated Pipeline
```bash
python -m src.pipeline.run_pipeline --dataset ba_shapes --model gcn --explainer lt_def
```

### 3.4. Launch Interactive Web Demo (Streamlit)
```bash
streamlit run src/visualization/app_demo.py
```

---

## 4. Key Research Deliverables

- **Official Mini Proposal (Word & PDF):** [product/proposal/MINI_PROPOSAL_C10.docx](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/product/proposal/MINI_PROPOSAL_C10.docx) | [product/proposal/MINI_PROPOSAL_C10.pdf](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/product/proposal/MINI_PROPOSAL_C10.pdf)
- **Daily Engineering Diary:** [docs/diary/2026/09/2026-09-27-SNA-C10.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/diary/2026/09/2026-09-27-SNA-C10.md)
- **Paper Repository Catalog:** [docs/papers/README.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/papers/README.md)
- **Official Specification Anchor:** [docs/tai_lieu_da_export/04_de_bai_c10_chinh_thuc.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/tai_lieu_da_export/04_de_bai_c10_chinh_thuc.md)
- **Visual & UI Specification:** [docs/design/DESIGN.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/design/DESIGN.md)

---

## 5. Academic Integrity Statement

This project strictly adheres to the Academic Integrity Principles enforced by Dr. Tran Hung Nghiep:
- **Zero Mockup Rule:** 100% of reported metrics are obtained from executable code running on real datasets. Zero synthetic/hallucinated numbers.
- **Anti-Leakage Data Pipeline:** Strict train/validation/test split isolation prior to feature normalization.
- **Multi-Seed Reproducibility:** All empirical results are averaged over 5 distinct random seeds (`[42, 123, 456, 789, 999]`) and reported with standard deviations ($\mu \pm \sigma$).
