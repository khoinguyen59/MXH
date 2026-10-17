# T01 – Gói bàn giao dữ liệu và checkpoint GCN (BG01–BG03)

Thư mục: `bashapes/`, `cora/`, mỗi thư mục gồm `data.pt`, `gcn_ckpt.pt`, `config.json`, `val_nodes.json`. Kiểm tra: `checks.json`, `summary.json`.

## Nạp (cần torch, torch_geometric==2.8.0.post1)
```python
from load_t01 import load
d, model, pred, val_nodes = load("bashapes")      # hoặc "cora"
# d: x, edge_index, y, train_mask, val_mask, test_mask, gt_edge_mask (BA-Shapes; Cora = None),
#    undirected, pair_id, stats
# model: GCN 2 lớp, đã eval(), trọng số đã khóa; pred: lớp dự đoán cho mọi node
```
`python load_t01.py` kiểm tra dự đoán nạp lại khớp 100% với `pred`.

## Quy ước (BG01–BG03)
- **Không đổi** node ID và thứ tự cột của `edge_index`; không tự sắp xếp lại.
- `gt_edge_mask[j] = True` nếu cột j của `edge_index` thuộc motif nhà (chỉ BA-Shapes).
- Đồ thị vô hướng: cạnh gồm hai cột (u,v) và (v,u). `pair_id[j]` là số nguyên định danh cặp (min,max) – hai chiều cùng `pair_id`. Xóa/giữ cạnh phải xử lý cả hai cột cùng `pair_id`.
- **Lớp cần giải thích = `pred[node]`** (không phải `y`). Khi xóa/giữ cạnh vẫn đo xác suất của lớp này trên đồ thị gốc.
- `val_nodes.json`: 100 node validation phân tầng theo nhãn (`nodes`) và tập con 30 node (`multiseed_subset`). Không dùng node test để chỉnh prototype.
- Predictor luôn `model.eval()`, không cập nhật trọng số.
- Seed: split 0; huấn luyện GCN seed 1000; chi tiết trong `config.json`.
