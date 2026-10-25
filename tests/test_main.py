"""Tests for the baseline and the evaluation entry point."""

import numpy as np

from baseline import centre_crop_square, excess_green, predict_mask
from main import evaluate_pair, main


def test_excess_green_sign():
    green = np.zeros((1, 1, 3), np.uint8); green[..., 1] = 200
    brown = np.array([[[140, 70, 40]]], np.uint8)
    assert excess_green(green)[0, 0] > 0
    assert excess_green(brown)[0, 0] < 0


def test_centre_crop_shape():
    assert centre_crop_square(np.zeros((480, 640))).shape == (480, 480)


def test_predict_mask_finds_brown_trunk():
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[...] = (30, 160, 30)                  # leaves everywhere
    rgb[:, 310:330] = (120, 80, 40)           # brown trunk in the centre
    mask = predict_mask(rgb, closing_radius=0)
    assert mask.shape == (256, 256)
    assert mask[:, 128].all()                 # trunk column detected
    assert not mask[:, 10].any()              # leaves rejected


def test_evaluate_pair_returns_metrics():
    rgb = np.zeros((480, 640, 3), np.uint8); rgb[:, 310:330] = (120, 80, 40)
    gt = np.zeros((256, 256), bool); gt[:, 123:134] = True
    params = dict(exg_threshold=0.05, depth_min=1, depth_max=255, closing_radius=0)
    out = evaluate_pair(rgb, None, gt, params)
    assert set(out) == {"iou", "dice", "precision", "recall", "chamfer_px"}
    assert out["recall"] > 0.9


def test_main_missing_data_returns_error(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('data_root = "nope"\nseason = "Summer"\n[baseline]\n')
    assert main(["--config", str(cfg), "--data-root", str(tmp_path / "missing")]) == 1


def test_main_runs_on_treeid_style_files(tmp_path, capsys, monkeypatch):
    from PIL import Image
    import main as main_module
    stamp = "12-40-50"
    rgb = np.zeros((480, 640, 3), np.uint8); rgb[...] = (30, 160, 30); rgb[:, 310:330] = (120, 80, 40)
    gt = np.zeros((256, 256), np.uint8); gt[:, 123:134] = 255
    Image.fromarray(rgb).save(tmp_path / f"R01N01_Summer_RGB-{stamp}.png")
    Image.fromarray(np.full((480, 640, 3), 100, np.uint8)).save(tmp_path / f"R01N01_Summer_D-{stamp}.png")  # 3-channel depth
    Image.fromarray(gt).save(tmp_path / f"R01N01_Summer_GT-{stamp}.png")
    cfg = tmp_path / "c.toml"
    cfg.write_text('data_root = "x"\nseason = "Summer"\n[baseline]\nclosing_radius = 0\n')
    seen = []
    monkeypatch.setattr(main_module, "evaluate_pair", lambda rgb, depth, gt, params: seen.append(depth) or evaluate_pair(rgb, depth, gt, params))
    assert main(["--config", str(cfg), "--data-root", str(tmp_path)]) == 0
    assert "Evaluated 1 images" in capsys.readouterr().out
    assert seen[0] is not None and seen[0].shape == (480, 640)   # depth file found and reduced to one channel
