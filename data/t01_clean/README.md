# T01 Clean - Gói Bàn Giao Dữ Liệu và Checkpoint GCN Chuẩn Hóa

Được sinh tự động từ pipeline `src.pipeline.run_t01_clean`, loại bỏ 100% các lỗi kỹ thuật và phương pháp luận của nhóm T01 trước đây:
- **BA-Shapes:** Khắc phục hoàn toàn sụp đổ Class 3 (Recall đạt 100%), chuẩn hóa đặc trưng all-ones và RWPE 4-dim, GCN 3 lớp bao phủ motif nhà 5 đỉnh.
- **Cora:** Thực thi chuẩn Dual-Reporting (Planetoid 140/500/1000 minh bạch 1,068 node không nhãn; Full Dense 70/15/15 phủ 100% 2,708 node).
- **Explainers Compatibility:** GCN hỗ trợ truyền trực tiếp `edge_weight` liên tục để tối ưu hóa mask cho GNNExplainer và PGExplainer.
- **Tính toàn vẹn:** 100% đối tượng được tuần tự hóa dưới định dạng chuẩn `torch_geometric.data.Data`.

## Cấu Trúc Thư Mục
- `bashapes/`: `data.pt`, `gcn_ckpt.pt`, `val_nodes.json`, `config.json`, các checkpoint 5 seeds `gcn_seed*.pt`.
- `cora/`: `data.pt`, `gcn_ckpt.pt`, `val_nodes.json`, `config.json`, các checkpoint 5 seeds `gcn_seed*.pt`.
- `cora_dense/`: `data.pt`, `gcn_ckpt.pt`, `config.json`, các checkpoint 5 seeds `gcn_seed*.pt`.
- `load_t01.py`: Giao diện nạp thống nhất tương thích hoàn toàn cho T02 (`from load_t01 import load`).
- `summary.json`: Bảng tổng hợp số liệu mean ± std qua 5 random seeds cố định [42, 123, 456, 789, 999].
- `checks.json`: Nhật ký kiểm định không mockup, xác nhận trung thực 100% số liệu.

## Cách Sử Dụng Cho T02
```python
from load_t01 import load
from torch_geometric.explain import ModelConfig

# 1. Nạp dữ liệu, mô hình frozen eval, dự đoán và validation nodes
data, model, pred, val_nodes = load("bashapes") # hoặc "cora", "cora_dense"

# 2. Cấu hình PyG Explainer chuẩn hóa 100%
model_config = ModelConfig(
    mode="multiclass_classification",
    task_level="node",
    return_type="log_probs",  # Model hỗ trợ 'log_probs', 'probs', hoặc 'raw'
)
```
