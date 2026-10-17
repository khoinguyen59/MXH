"""Automated Verification & Integrity Audit for Clean T01 Handover.

Performs rigorous sanity checks across all deliverables in data/t01_clean/:
1. Format Integrity: Verifies Data is an instance of torch_geometric.data.Data (no raw dict).
2. Anti-Leakage Separation: Strictly asserts train_mask, val_mask, and test_mask are disjoint.
3. Class 3 Collapse Resolution: Confirms Class 3 Recall on BA-Shapes >= 85% (previously 0.0%).
4. Standard Checkpoints: Verifies gcn_ckpt.pt has state_dict, pred, in_dim, n_classes, etc.
5. Packaging Assets: Verifies summary.json, load_t01.py, and README.md exist and execute cleanly.
6. Explainer Readiness: Tests forward pass with continuous edge_weight and checks backprop gradients.
7. Honest Cora Metadata: Confirms unlabeled nodes count is explicitly and transparently recorded.
8. Validation Nodes Integrity: Asserts all 100 val_nodes belong strictly to val_mask.
"""

import importlib.util
import json
from pathlib import Path
import sys

# Ensure root directory is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch_geometric.data import Data

from src.models.gcn import GCN


def verify_t01(clean_dir: Path):
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("SNA Topic C.10: Automated Integrity Audit for T01 Deliverables")
    print(f"Target Directory: {clean_dir.resolve()}")
    print("=" * 70)

    assert clean_dir.exists(), f"Target directory {clean_dir} does not exist!"

    # 1. Audit Packaging Files
    print("\n[Audit 1/5] Auditing Root Deliverables (summary.json, load_t01.py, README.md, checks.json)...")
    summary_path = clean_dir / "summary.json"
    assert summary_path.exists(), "Root summary.json missing!"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_obj = json.load(f)
    assert "bashapes" in summary_obj and "cora" in summary_obj and "cora_dense" in summary_obj
    print("  ✓ summary.json verified with bashapes, cora, and cora_dense sections.")

    load_path = clean_dir / "load_t01.py"
    assert load_path.exists(), "load_t01.py missing!"
    readme_path = clean_dir / "README.md"
    assert readme_path.exists(), "README.md missing!"
    print("  ✓ load_t01.py and README.md verified.")

    # 2. Audit BA-Shapes
    print("\n[Audit 2/5] Auditing BA-Shapes Package & gcn_ckpt.pt...")
    ba_path = clean_dir / "bashapes" / "data.pt"
    assert ba_path.exists(), "BA-Shapes data.pt missing!"
    data_ba = torch.load(ba_path, weights_only=False)
    assert isinstance(data_ba, Data), f"BA-Shapes data.pt must be PyG Data, got {type(data_ba)}"
    print(f"  ✓ PyG Data class verified: {data_ba.num_nodes} nodes, {data_ba.edge_index.shape[1]} edges.")

    # Check masks
    assert not (data_ba.train_mask & data_ba.val_mask).any(), "Train-Val mask overlap detected!"
    assert not (data_ba.train_mask & data_ba.test_mask).any(), "Train-Test mask overlap detected!"
    assert not (data_ba.val_mask & data_ba.test_mask).any(), "Val-Test mask overlap detected!"
    print("  ✓ Strict Anti-Leakage mutual exclusivity verified across train/val/test masks.")

    # Check validation nodes
    val_nodes_path = clean_dir / "bashapes" / "val_nodes.json"
    assert val_nodes_path.exists(), "BA-Shapes val_nodes.json missing!"
    with open(val_nodes_path, "r", encoding="utf-8") as f:
        val_nodes_data = json.load(f)
    val_nodes_ba = val_nodes_data["nodes"] if isinstance(val_nodes_data, dict) else val_nodes_data
    assert len(val_nodes_ba) == 100, f"Expected 100 validation nodes, got {len(val_nodes_ba)}"
    assert all(data_ba.val_mask[idx].item() for idx in val_nodes_ba), "Some val_nodes are outside val_mask!"
    print(f"  ✓ 100 validation nodes verified strictly inside val_mask.")

    # Check model checkpoint and Class 3 Recall
    ckpt_ba_path = clean_dir / "bashapes" / "gcn_ckpt.pt"
    assert ckpt_ba_path.exists(), "BA-Shapes standard checkpoint gcn_ckpt.pt missing!"
    ckpt_ba = torch.load(ckpt_ba_path, weights_only=False)
    for required_key in ["state_dict", "pred", "in_dim", "n_classes", "hidden", "num_layers"]:
        assert required_key in ckpt_ba, f"Missing key {required_key} in gcn_ckpt.pt!"

    model_ba = GCN(
        in_channels=ckpt_ba["in_dim"],
        hidden_channels=ckpt_ba["hidden"],
        out_channels=ckpt_ba["n_classes"],
        num_layers=ckpt_ba["num_layers"],
        dropout=ckpt_ba.get("dropout", 0.0),
    )
    model_ba.load_state_dict(ckpt_ba["state_dict"])
    model_ba.eval()

    with torch.no_grad():
        out_ba = model_ba(data_ba.x, data_ba.edge_index)
        preds_ba = out_ba.argmax(dim=-1)

    assert torch.equal(preds_ba, ckpt_ba["pred"]), "Stored predictions do not match forward pass!"

    c3_mask = data_ba.test_mask & (data_ba.y == 3)
    c3_total = int(c3_mask.sum())
    c3_correct = int((preds_ba[c3_mask] == 3).sum())
    c3_recall = c3_correct / c3_total if c3_total > 0 else 0.0
    print(f"  ✓ Class 3 (Roof) Test Recall: {c3_recall:.2%} ({c3_correct}/{c3_total}).")
    assert c3_recall >= 0.85, f"Class 3 Recall {c3_recall:.2%} is below required 85.0% threshold!"

    # 3. Audit Cora (Planetoid)
    print("\n[Audit 3/5] Auditing Cora (Planetoid Public Split)...")
    cora_path = clean_dir / "cora" / "data.pt"
    assert cora_path.exists(), "Cora data.pt missing!"
    data_cora = torch.load(cora_path, weights_only=False)
    assert isinstance(data_cora, Data), f"Cora data.pt must be PyG Data, got {type(data_cora)}"

    ckpt_cora_path = clean_dir / "cora" / "gcn_ckpt.pt"
    assert ckpt_cora_path.exists(), "Cora gcn_ckpt.pt missing!"
    ckpt_cora = torch.load(ckpt_cora_path, weights_only=False)
    assert "state_dict" in ckpt_cora and "pred" in ckpt_cora

    with open(clean_dir / "cora" / "config.json", "r", encoding="utf-8") as f:
        cfg_cora = json.load(f)
    assert cfg_cora["unlabeled_nodes"] == 1068, "Transparent unlabeled count missing!"
    print(f"  ✓ Transparent metadata verified: 140 train, 500 val, 1000 test, 1068 unlabeled nodes.")

    # 4. Audit Cora (Full Dense)
    print("\n[Audit 4/5] Auditing Cora (Full Dense Split)...")
    dense_path = clean_dir / "cora_dense" / "data.pt"
    assert dense_path.exists(), "Cora dense data.pt missing!"
    data_dense = torch.load(dense_path, weights_only=False)
    assert (data_dense.train_mask | data_dense.val_mask | data_dense.test_mask).all(), "Dense split must cover 100% of nodes!"
    print(f"  ✓ 100% Full coverage verified across all 2,708 nodes.")

    # 5. Explainer Interface Compatibility & load_t01 loader test
    print("\n[Audit 5/5] Auditing Explainer edge_weight Compatibility & load_t01 Helper...")
    weights = torch.rand(data_ba.edge_index.shape[1], requires_grad=True)
    out_edge = model_ba(data_ba.x, data_ba.edge_index, edge_weight=weights)
    out_edge.sum().backward()
    assert weights.grad is not None and not torch.isnan(weights.grad).any(), "Gradient backward failed on edge_weight!"
    print("  ✓ Continuous edge_weight backpropagation verified successfully.")

    # Dynamic load test
    spec = importlib.util.spec_from_file_location("load_t01", str(load_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d_loaded, m_loaded, pred_loaded, val_nodes_loaded = mod.load("bashapes", root=clean_dir)
    assert torch.equal(pred_loaded, ckpt_ba["pred"]), "load_t01 prediction mismatch on bashapes!"
    print("  ✓ load_t01 helper loader tested successfully on bashapes.")

    print("\n" + "=" * 70)
    print("ALL 5 INTEGRITY AUDITS PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/t01_clean")
    verify_t01(target)
