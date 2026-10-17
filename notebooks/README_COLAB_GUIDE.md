# HƯỚNG DẪN VẬN HÀNH THỰC NGHIỆM: LOCAL DEV VS. GOOGLE COLAB

> **Dự án:** SNA Topic C.10 - Making Graph Machine Learning Models Explainable (UIT 2026)  
> **Nguyên tắc phân bổ tài nguyên:** Phân tách nghiêm ngặt môi trường Máy cá nhân (Local) và Google Colab.  

---

## 1. PHÂN CÔNG TÀI NGUYÊN NGHIÊM NGẶT

```
                      MÔ HÌNH PHÂN CÔNG TÀI NGUYÊN (LOCAL VS. COLAB)
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│  MÁY CÁ NHÂN (LOCAL ENVIRONMENT)             │  GOOGLE COLAB (HIGH-PERFORMANCE RUNTIME)     │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • Lập trình, tái cấu trúc, refactoring mã.   │ • Huấn luyện GNN quy mô lớn nhiều epoch.    │
│ • Chạy kiểm thử tự động (pytest tests/).     │ • Quét 5 fixed random seeds đầy đủ.          │
│ • Chạy smoke test / dry-run data nhỏ:        │ • Chạy các tập dữ liệu lớn: Amazon Photo,    │
│   - Số seed = 1 (seed 42).                   │   Coauthor CS, ogbn-arxiv, Facebook Page.    │
│   - Số node mục tiêu: 2 - 5 node.            │ • Chạy Explainer tốn kém (MCTS, multi-opt).  │
│   - Mục đích: Đảm bảo code chạy không lỗi.   │ • Kết quả tự động nén & sync Google Drive.   │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 2. DANH MỤC 4 NOTEBOOK NỘP BÀI CHUẨN MỰC (ACADEMIC SUBMISSION NOTEBOOKS)

Để đáp ứng quy chuẩn nộp bài của Giảng viên (TS. Trần Hùng Nghiệp), toàn bộ mã nguồn thực nghiệm được tổ chức thành đúng **4 notebook độc lập**, tương ứng ma trận $(2 \text{ Datasets}) \times (2 \text{ Model Directions})$, loại bỏ hoàn toàn các ký hiệu nội bộ (`t01`, `clean`, `handover`):

### 1. Phân hệ BA-Shapes (Synthetic Benchmark)
1. **[`SNA_C10_BAShapes_Baseline_Explainability.ipynb`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/notebooks/SNA_C10_BAShapes_Baseline_Explainability.ipynb):**
   - Đánh giá các thuật toán giải thích trên Mô hình GNN Cơ sở (Baseline GCN).
   - Đo lường Fidelity+, Fidelity-, Sparsity và Motif-ROC-AUC đối chiếu House motif chân lý.
2. **[`SNA_C10_BAShapes_Proposed_Explainability.ipynb`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/notebooks/SNA_C10_BAShapes_Proposed_Explainability.ipynb):**
   - Đánh giá các thuật toán giải thích trên Mô hình GNN Đề xuất (Proposed GCN với RWPE 4 chiều & receptive field 3 lớp).
   - Chứng minh thực nghiệm triệt tiêu hiện tượng Negative Fidelity và nâng cao Motif-ROC-AUC.

### 2. Phân hệ Cora (Academic Citation Network)
3. **[`SNA_C10_Cora_Baseline_Explainability.ipynb`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/notebooks/SNA_C10_Cora_Baseline_Explainability.ipynb):**
   - Đánh giá các thuật toán giải thích trên Mô hình GNN Cơ sở (Baseline GCN) với Planetoid split.
   - Đo lường độ trung thực cấu trúc (Structural Fidelity) và độ thưa (Sparsity).
4. **[`SNA_C10_Cora_Proposed_Explainability.ipynb`](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/notebooks/SNA_C10_Cora_Proposed_Explainability.ipynb):**
   - Đánh giá toàn diện trên Mô hình GNN Đề xuất (Proposed GCN).
   - Đo lường đồng thời **Độ trung thực Cấu trúc (Edge Fidelity)** và **Độ trung thực Thuộc tính (Feature Fidelity)** để giải đáp câu hỏi nghiên cứu **RQ2**.

---

## 3. QUY TRÌNH THỰC HIỆN 1-CLICK TRÊN GOOGLE COLAB

1. **Mở Colab:** Truy cập [Google Colab](https://colab.research.google.com/) -> Chọn Upload notebook mong muốn trong danh mục trên.
2. **Chọn Runtime GPU:** `Runtime -> Change runtime type -> T4 GPU`.
3. **Đồng bộ mã nguồn:** Kéo thả thư mục `src/` và `data/` vào Files, hoặc kết nối Google Drive.
4. **Bấm "Run all":** Notebook tự động chạy, in kết quả, xuất bảng so sánh, và tự động tải file zip kết quả về máy.
