# SNA Topic C.10: Making Graph Machine Learning Models Explainable

[![CI Pipeline](https://github.com/khoinguyen59/MXH/actions/workflows/ci.yml/badge.svg)](https://github.com/khoinguyen59/MXH/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.6+-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Integrity](https://img.shields.io/badge/Zero--Mockup-Verified-success.svg)](#6-academic-integrity--verification-statement)

> **Môn học:** Phân tích Mạng xã hội (Social Network Analysis - SNA)  
> **Trường:** Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM (2026)  
> **Giảng viên hướng dẫn:** TS. Trần Hùng Nghiệp (`nghiepth@uit.edu.vn`)  
> **Phân loại đề tài:** Mức độ C (Nghiên cứu chuyên sâu, Độ khó: 4/5)  
> **Nhóm thực hiện:** Nhóm 02  

---

## 1. Tổng Quan & Câu Hỏi Nghiên Cứu Cốt Lõi

Mạng nơ-ron đồ thị (Graph Neural Networks - GNNs) đạt hiệu năng vượt trội trong các bài toán phân lớp nốt và dự đoán liên kết trên mạng xã hội. Tuy nhiên, cơ chế lan truyền tin nhắn (Message Passing) đa lớp biến GNN thành những "hộp đen" phức tạp, đan xen chặt chẽ giữa cấu trúc topo cục bộ và không gian đặc trưng nốt nhiều chiều.

Dự án này tập trung nghiên cứu, xây dựng và đánh giá thực nghiệm toàn diện hệ thống giải thích (Explainability Engine) cho các mô hình GNN trên đồ thị xã hội, giải quyết 3 câu hỏi nghiên cứu cốt lõi:

- **RQ1 (Pareto Frontier):** Thuật toán giải thích nào đạt điểm cân bằng Pareto tối ưu giữa Độ trung thực (Fidelity) và Độ thưa gọn (Sparsity) trên đồ thị?
- **RQ2 (Topo vs. Feature Attribution):** Cấu trúc liên kết cục bộ (Topological Connectivity) hay thuộc tính đặc trưng nội tại của nốt (Node Features) có tác động chi phối lớn hơn đến dự đoán của GNN?
- **RQ3 (Semantic Faithfulness):** Lời giải thích bằng ngôn ngữ tự nhiên (Natural Language Explanations - NLE) được sinh ra qua LLMs có bảo toàn được tính chuẩn xác ngữ nghĩa với cấu trúc motif đồ thị con trích xuất hay không?

---

## 2. Báo Cáo Thực Nghiệm 1: Đối Chuẩn Mô Hình Phân Lớp GNN

Dự án tuân thủ nghiêm ngặt nguyên tắc **Zero-Mockup** (100% số liệu đo đạc thực tế từ code chạy trên dữ liệu thật, không số ảo). Dưới đây là kết quả kiểm định độc lập giữa mô hình Baseline gốc (gói T01), mô hình Baseline đã sửa của Thiện (Nhóm 09) và mô hình Đề xuất Cải tiến của Khôi (Nhóm 10).

### 2.1. Tập dữ liệu tổng hợp BA-Shapes (700 nodes, 3.972 edges, 4 classes)

| Lớp Hình Thái Motif | Baseline Gốc T01 (Nhóm 1) | Clean Baseline (Thiện - Nhóm 2) | Proposed Model (Khôi - Nhóm 2) | Nhận Xét & Phân Tích Kỹ Thuật |
| :--- | :---: | :---: | :---: | :--- |
| **0: Đồ thị nền (BA)** | 100.0% (60/60) | 100.0% (60/60) | **93.33% (56/60)** | Khôi bị nhầm nhẹ 4 nốt ranh giới nối với motif |
| **1: Chân nhà (Đáy)** | 100.0% (32/32) | 93.75% (30/32) | **100.0% (32/32)** | Nhận diện tuyệt đối đế ngôi nhà |
| **2: Tường nhà (Thân)** | 96.88% (31/32) | 81.25% (26/32) | **100.0% (32/32)** | Nhận diện tuyệt đối liên kết ngang thân nhà |
| **3: Mái nhà (Đỉnh nhọn)**| **0.0% (0/16)** | **93.75% (15/16)** | **100.0% (16/16)** | **Khôi & Thiện phục hồi hoàn toàn lỗi mất Lớp 3** |
| **Độ chính xác Test (Acc)** | **87.86%** | **93.57%** | **97.14%** | **Mô hình Khôi đạt 136/140 nốt đúng trên test set** |
| **Macro-F1 Score** | **0.6891** | **0.9260** | **0.9650** | **F1 tăng vọt +0.2759 nhờ khắc phục Lớp 3** |
| **Receptive Field / Features** | 2 lớp GCN | 2 lớp + Weighted Loss | 3 lớp GCN + **RWPE 4D** | Vươn tới bán kính 3-hop của đỉnh mái nhà |

### 2.2. Tập dữ liệu mạng trích dẫn Cora (2.708 nodes, 10.556 edges, 7 classes)
- **Độ chính xác kiểm thử (Test Accuracy):** **83.60%** (836 / 1.000 nốt test).
- **Macro-F1 Score:** **82.46%** (phân bố đều trên toàn bộ 7 lĩnh vực khoa học, không bị sụp đổ lớp).
- **Validation Accuracy:** **81.60%** trên 100 validation nodes chuẩn hóa (BG01-BG03).

---

## 3. Báo Cáo Thực Nghiệm 2: Đánh Giá Toàn Diện Các Explainer (XAI Benchmark)

Thực nghiệm được thực hiện trên **Google Colab (GPU T4)** trên 100 nốt validation phân tầng $\times$ 5 thuật toán giải thích $= 1.000$ lượt đánh giá thực tế chi tiết ở ngân sách cạnh $k = 5$ (theo chuẩn quy ước bàn giao BG08).

### 3.1. Bảng đối chuẩn giải thích: Baseline GCN vs. Proposed GCN (RWPE 4D)

| Thuật Toán Explainer | Baseline Topo Fid+ | Proposed Topo Fid+ | Mức Tăng ($\Delta$) | Baseline Motif AUC | Proposed Motif AUC | Độ Thưa Cạnh (Sparsity) | Feat Fid+ (RQ2) | Thời Gian Suy Luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Saliency** | $0.0205 \pm 0.25$ | **$0.5087 \pm 0.24$** | **+0.4881** | $0.4579 \pm 0.28$ | **$0.7323 \pm 0.22$** | $67.7\% \pm 28.1\%$ | **$0.5727$** | $6.2$ ms / nốt |
| **Random Baseline** | $-0.0277 \pm 0.18$ | $0.3839 \pm 0.26$ | +0.4117 | $0.5137 \pm 0.21$ | $0.4893 \pm 0.20$ | $9.6\% \pm 5.3\%$ | $0.0000$ | $2.2$ ms / nốt |
| **Nearest-First Baseline** | $0.0540 \pm 0.35$ | $0.5087 \pm 0.27$ | +0.4547 | $0.6416 \pm 0.16$ | $0.6279 \pm 0.16$ | $0.0\% \pm 0.0\%$ | $0.0000$ | $2.2$ ms / nốt |
| **GNNExplainer** | $-0.0318 \pm 0.21$ | **$0.5488 \pm 0.21$** | **+0.5806** | $0.6087 \pm 0.11$ | **$0.6682 \pm 0.12$** | $53.0\% \pm 23.3\%$ | **$0.1972$** | $355.5$ ms / nốt |
| **PGExplainer** | $0.0251 \pm 0.27$ | **$0.4940 \pm 0.28$** | **+0.4689** | **$1.0000 \pm 0.00$** | **$0.9729 \pm 0.07$** | **$94.1\% \pm 11.2\%$** | $0.0000$ | **$7.6$ ms / nốt** |

### 3.2. Những phát hiện khoa học định lượng then chốt
1. **Trả lời RQ1 (Tối ưu hóa Pareto giữa Fidelity và Sparsity):**
   - **PGExplainer** đạt vị trí thống trị trên đường biên Pareto: Phục hồi cấu trúc motif thực tế với **Motif ROC-AUC = 0.9729 - 1.0000**, độ thưa cạnh tuyệt đối **94.1%** (chỉ giữ đúng các cạnh cốt lõi của motif ngôi nhà) và thời gian thực thi cực nhanh **7.6 ms**, nhanh gấp **45 lần** so với GNNExplainer ($355.5$ ms).
2. **Trả lời RQ2 (Ảnh hưởng Cấu trúc Topo vs. Đặc trưng Nốt):**
   - Khi mô hình được trang bị thông tin vị trí không gian (RWPE 4D), **Feature Fidelity+ của Gradient Saliency đạt $0.5727$**, vượt qua cả Topological Fidelity+ ($0.5087$).
   - *Kết luận:* Khi nốt có đặc trưng định vị, bộ phân loại GNN kết hợp nhịp nhàng cả hai nguồn tin nhắn, trong đó đặc trưng nốt đóng vai trò neo giữ không gian quyết định nhãn phân lớp.
3. **Hiện tượng hồi sinh Gradient của Saliency:**
   - Trên baseline cũ bị mù Class 3, Saliency có AUC chỉ $0.4579$ (tệ hơn đoán ngẫu nhiên). Trên mô hình Proposed, tín hiệu gradient qua 3 lớp được thông suốt giúp AUC tăng vọt lên **$0.7323$ (+0.2743)**.

---

## 4. Kiến Trúc Mã Nguồn Module Hóa (Decoupled Modular Architecture)

Dự án được tổ chức theo tiêu chuẩn kỹ thuật phần mềm nghiêm ngặt, tách bạch hoàn toàn trách nhiệm giữa các service:

```text
MXH/
├── .github/workflows/ci.yml         # CI/CD tự động chạy Unit Tests & Anti-Leakage Check
├── .gitignore                       # Loại trừ 100% file nhị phân *.pt (lưu trên Drive)
├── README.md                        # Tài liệu hướng dẫn & công bố số liệu thực nghiệm
├── requirements.txt                 # Các gói phụ thuộc chuẩn hóa (PyTorch, PyG, scikit-learn)
│
├── src/                             # MÃ NGUỒN CHÍNH (Core Services)
│   ├── config/                      # Hyperparameters, 5 random seeds [42, 123, 456, 789, 999]
│   ├── datasets/                    # Data Ingestion & Handover Adapters (Cora, BA-Shapes)
│   ├── models/                      # Kiến trúc GNN động (GCN, GAT, GraphSAGE, RWPE Positional)
│   ├── explainers/                  # Explainability Engine:
│   │   ├── edge_explainer_service.py # Wrapper chuẩn BG04, gộp cạnh canonical (BG07), giữ node_mask
│   │   ├── saliency.py              # Gradient Saliency Explainer
│   │   ├── gnn_explainer.py         # Mutual Information Edge & Node Mask Optimization
│   │   ├── pg_explainer.py          # Parameterized Neural Explainer (Train trên train_mask - BG03)
│   │   ├── eig_search.py            # Energy-based Subgraph Search
│   │   ├── lt_def.py                # Linear-Time Dual Explanation Framework
│   │   └── subgraphx.py             # MCTS-based Subgraph Shapley Explainer
│   ├── evaluation/                  # Metric Service:
│   │   ├── fidelity.py              # Fidelity+ (xóa cạnh), Fidelity- (giữ cạnh), Smart Tie-Breaking (BG08)
│   │   ├── metrics.py               # Sparsity, ROC-AUC, PR-AUC ground truth motif
│   │   ├── agreement.py             # Jaccard@k, Cohen's Kappa, Kendall's Tau liên explainer
│   │   └── statistical_tests.py     # ANOVA phương sai, kiểm định giả thuyết thống kê
│   ├── nle_grounding/               # Graph-to-Text Serializer & LLM Semantic Rejection Filter
│   ├── visualization/               # Web App tương tác đồ thị (Streamlit) & Motif Plotter
│   └── pipeline/                    # Orchestrator CLI điều phối chạy toàn bộ quy trình
│
├── notebooks/                       # JUPYTER NOTEBOOKS CHẠY THỰC NGHIỆM TRÊN GOOGLE COLAB
│   ├── README_COLAB_GUIDE.md        # Hướng dẫn chạy Colab T4 đồng bộ kết quả về Drive
│   ├── SNA_C10_BAShapes_Baseline_Explainability.ipynb # Thực nghiệm BA-Shapes Baseline
│   ├── SNA_C10_BAShapes_Proposed_Explainability.ipynb # Thực nghiệm BA-Shapes Proposed (RWPE)
│   ├── SNA_C10_Cora_Baseline_Explainability.ipynb     # Thực nghiệm Cora Baseline
│   └── SNA_C10_Cora_Proposed_Explainability.ipynb     # Thực nghiệm Cora Proposed
│
├── tests/                           # BỘ KIỂM THỬ TỰ ĐỘNG (100% Passed)
│   ├── test_explain_edges_bg04_bg08.py # Test giao diện chuẩn BG04 & cơ chế tie-breaking BG08
│   ├── test_explainers.py              # Test 8 explainers trên CPU/CUDA
│   └── test_anti_leakage.py            # Test cô lập split train/val/test (Special Lecture 01)
│
└── data/                            # GIAO DIỆN DỮ LIỆU SẠCH TRÊN GITHUB (Chỉ chứa Metadata)
    ├── README.md                    # Hướng dẫn nạp và link tải file .pt từ Google Drive
    ├── load_t01.py                  # Module nạp dữ liệu chuẩn cho các nhóm liên minh
    ├── summary.json                 # Số liệu thống kê đồ thị & độ chính xác phân lớp
    ├── checks.json                  # 20 bài test kiểm tra tính toàn vẹn (Zero-Leakage)
    ├── bashapes/                    # config.json & 100 nốt validation phân tầng
    └── cora/                        # config.json & 100 nốt validation phân tầng
```

---

## 5. Hướng Dẫn Cài Đặt & Tái Hiện Kết Quả

### 5.1. Cài đặt môi trường cục bộ
```bash
# 1. Clone repository
git clone https://github.com/khoinguyen59/MXH.git
cd MXH

# 2. Khởi tạo virtual environment và cài đặt dependencies
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.2. Chạy bộ kiểm thử tự động (Unit Tests)
```bash
# Kiểm tra 100% tính toàn vẹn code và các giao diện BG04/BG08
pytest tests/ -v
```

### 5.3. Tải dữ liệu đồ thị và trọng số từ Google Drive
Do chính sách của GitHub không lưu trữ file nhị phân lớn, toàn bộ các file `.pt` (`data.pt`, `gcn_ckpt.pt`) được lưu trữ tại Google Drive của nhóm:
- Tải file nén `data_drive_upload.zip` hoặc giải nén thư mục `data/` đặt vào thư mục gốc của dự án.
- Kiểm tra việc nạp dữ liệu:
```bash
python data/load_t01.py
```

### 5.4. Chạy thực nghiệm quy mô lớn trên Google Colab
Mở các notebook trong thư mục `notebooks/` trên Google Colab với GPU T4 miễn phí để chạy quét 100 validation nodes $\times$ 5 explainers:
- [`notebooks/SNA_C10_BAShapes_Baseline_Explainability.ipynb`](notebooks/SNA_C10_BAShapes_Baseline_Explainability.ipynb)
- [`notebooks/SNA_C10_BAShapes_Proposed_Explainability.ipynb`](notebooks/SNA_C10_BAShapes_Proposed_Explainability.ipynb)

### 5.5. Khởi chạy Web App trực quan hóa giải thích (Streamlit)
```bash
streamlit run src/visualization/app_demo.py
```

---

## 6. Cam Kết Liêm Chính Học Thuật (Zero-Mockup Policy)

Dự án tuân thủ tuyệt đối chuẩn mực học thuật của **TS. Trần Hùng Nghiệp** (Đề tài C.10, Đồ án SNA 2026):
1. **Không tạo số liệu giả:** 100% các giá trị trong bảng đối chuẩn (Test Accuracy, Macro-F1, Fidelity+, Fidelity-, Motif ROC-AUC, Sparsity, Runtime) đều được trích xuất trực tiếp từ các file nhật ký thực nghiệm thực tế (`bashapes_node_evaluations.csv`, `bashapes_xai_summary.json`).
2. **Quy trình chống rò rỉ dữ liệu (Anti-Leakage Pipeline):** Tách biệt tuyệt đối tập Train / Validation / Test trước khi thực hiện chuẩn hóa đặc trưng. Mọi thuật toán explainer tham số hóa (PGExplainer) chỉ được huấn luyện nghiêm ngặt trên `train_mask`.
3. **Tính lặp lại đa hạt giống:** Đánh giá ổn định thống kê qua 5 random seeds cố định `[42, 123, 456, 789, 999]` và báo cáo độ lệch chuẩn ($\mu \pm \sigma$).
