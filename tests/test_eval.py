"""Tests for the extra evaluation: occluded recall, GT alignment, k-fold CV and U-Net cost."""

import csv

import numpy as np
import pytest
from PIL import Image

from alignment import estimate_offset, shift_mask
from metrics import occluded_recall


def test_occluded_recall_counts_only_hidden_pixels():
    gt = np.zeros((10, 10), bool); gt[:, 4:6] = True        # 20 branch pixels
    visible = np.zeros_like(gt); visible[:5] = gt[:5]       # top half visible, bottom 10 hidden
    pred = np.zeros_like(gt); pred[5:8, 4:6] = True         # finds 6 of the 10 hidden pixels
    assert occluded_recall(pred, gt, visible) == pytest.approx(0.6)
    assert occluded_recall(visible, gt, visible) == 0.0     # a visible-only detector scores 0
    assert np.isnan(occluded_recall(pred, gt, gt))          # nothing hidden: undefined


def test_occluded_recall_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        occluded_recall(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((5, 5)))


def test_shift_mask_moves_and_fills():
    m = np.zeros((5, 5), bool); m[0, 0] = True
    assert shift_mask(m, 2, 3)[2, 3] and shift_mask(m, 2, 3).sum() == 1
    assert shift_mask(m, -1, 0).sum() == 0                  # shifted out of the frame


def test_estimate_offset_recovers_known_shift():
    gt = np.zeros((128, 128), bool); gt[:, 60:66] = True; gt[30:34, 40:90] = True
    ref = shift_mask(gt, 7, -12)
    dy, dx, overlap = estimate_offset(gt, ref, max_shift=20)
    assert (dy, dx) == (7, -12) and overlap == ref.sum()
    assert estimate_offset(gt, np.zeros_like(gt)) == (0, 0, 0)


def test_kfold_by_tree_has_no_leakage():
    from crossval import kfold_by_tree
    pairs = [(f"R01N{i % 12:02d}_Summer_RGB-{i}.png", None, None) for i in range(36)]
    folds = kfold_by_tree(pairs, k=4, seed=0)
    assert len(folds) == 4
    val_trees = []
    for train, val in folds:
        tr = {p[0].split("_")[0] for p in train}
        va = {p[0].split("_")[0] for p in val}
        assert not tr & va and len(train) + len(val) == 36
        val_trees += sorted(va)
    assert sorted(val_trees) == sorted({p[0].split("_")[0] for p in pairs})  # each tree validated once
    with pytest.raises(ValueError):
        kfold_by_tree(pairs[:2], k=4)


def _synthetic_dataset(folder, n_trees=10):
    rng = np.random.default_rng(0)
    for i in range(n_trees):
        col = 300 + int(rng.integers(-20, 20))
        rgb = np.zeros((480, 640, 3), np.uint8); rgb[...] = (30, 160, 30); rgb[:, col:col + 20] = (120, 80, 40)
        depth = np.full((480, 640), 100 + i, np.uint8)   # distinct per tree, or the duplicate-depth check drops them
        gt = np.zeros((256, 256), np.uint8)
        c = (col - 80) * 256 // 480
        gt[:, c:c + 11] = 255
        name = f"R01N{i:02d}_Summer_{{}}-1.png"
        Image.fromarray(rgb).save(folder / name.format("RGB"))
        Image.fromarray(depth).save(folder / name.format("D"))
        Image.fromarray(gt).save(folder / name.format("GT"))


def test_train_and_crossval_run_end_to_end(tmp_path):
    from crossval import main as cv_main
    from train import main as train_main
    data, res = tmp_path / "data", tmp_path / "res"
    data.mkdir()
    _synthetic_dataset(data)
    cfg = tmp_path / "c.toml"
    cfg.write_text('data_root = "x"\nseason = "Summer"\n[baseline]\n[ml]\nwork_size = 32\npixels_per_class = 50\n'
                   '[eval]\ncv_folds = 3\nalign_max_shift = 10\n')
    common = ["--config", str(cfg), "--data-root", str(data), "--res", str(res)]
    assert train_main(common + ["--models", "tree", "--out", str(tmp_path / "out")]) == 0
    with open(res / "ml_results.csv") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}
    assert "occ_recall" in rows["tree"] and float(rows["baseline"]["occ_recall"]) == 0.0
    assert (res / "gt_offsets.csv").exists() and (res / "ml_results_aligned.csv").exists()
    assert cv_main(common + ["--models", "prior", "tree", "linear"]) == 0
    with open(res / "cv_summary.csv") as f:
        summary = list(csv.DictReader(f))
    assert [r["model"] for r in summary] == ["prior", "tree", "linear"] and all(r["folds"] == "3" for r in summary)


def test_unet_flops_match_hand_count():
    torch = pytest.importorskip("torch")
    from unet import count_flops
    conv = torch.nn.Conv2d(4, 8, 3, padding=1)
    flops, acts = count_flops(torch.nn.Sequential(conv), (1, 4, 16, 16))
    assert flops == 2 * (16 * 16 * 8) * (4 * 3 * 3) and acts == 8 * 16 * 16
    up = torch.nn.ConvTranspose2d(8, 4, 2, stride=2)
    flops, _ = count_flops(torch.nn.Sequential(up), (1, 8, 16, 16))
    assert flops == 2 * (8 * 16 * 16) * 4 * 2 * 2


def test_gt_offset_manual_summary(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from gt_offset import summarise_manual
    f = tmp_path / "m.csv"
    f.write_text("image,visible_trunk_col,gt_trunk_col,dx,note\na,100,120,20,\nb,100,90,-10,\nc,,,,hidden\n")
    s = summarise_manual(f)
    assert s["images"] == 3 and s["readable"] == 2
    assert s["median_abs_dx"] == 15 and s["share_abs_ge_15"] == 0.5 and (s["min_dx"], s["max_dx"]) == (-10, 20)
    f.write_text("image,visible_trunk_col,gt_trunk_col,dx,note\na,,,,\n")
    assert summarise_manual(f) is None


def test_baseline_sensitivity_evaluate_scores_known_case():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from baseline_sensitivity import evaluate
    rgb = np.zeros((480, 640, 3), np.uint8); rgb[...] = (30, 160, 30); rgb[:, 310:330] = (120, 80, 40)
    gt = np.zeros((256, 256), bool); gt[:, 123:134] = True
    r = evaluate([(rgb, None, gt)], dict(exg_threshold=0.05, depth_min=1, depth_max=255, closing_radius=0))
    assert r["recall"] > 0.9 and r["precision"] > 0.9 and r["dice"] > 0.9
