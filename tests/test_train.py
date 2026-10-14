"""Tests for data loading, features, the regression models and the U-Net (skipped without PyTorch)."""

import numpy as np
import pytest
from PIL import Image

from dataset import find_pairs, load_sample, split_by_tree, tree_id


def _write_sample(folder, tree):
    stamp = "12-40-50"
    Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(folder / f"{tree}_Summer_RGB-{stamp}.png")
    Image.fromarray(np.full((480, 640, 3), 100, np.uint8)).save(folder / f"{tree}_Summer_D-{stamp}.png")
    Image.fromarray(np.zeros((256, 256), np.uint8)).save(folder / f"{tree}_Summer_GT-{stamp}.png")


def test_find_pairs_and_load(tmp_path):
    _write_sample(tmp_path, "R01N01")
    _write_sample(tmp_path, "R04N02")
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / "R01N09_Summer_RGB-1-2-3.png")  # no GT
    pairs = find_pairs(tmp_path, "Summer", exclude_rows=["R04"])
    assert [tree_id(p[0]) for p in pairs] == ["R01N01"]
    x, y = load_sample(*pairs[0])
    assert x.shape == (4, 256, 256) and y.shape == (256, 256)
    assert x[3].mean() == pytest.approx(100 / 255, abs=1e-3)


def test_split_by_tree_has_no_overlap():
    pairs = [(f"R01N{i:02d}_Summer_RGB-1.png", None, None) for i in range(40)]
    split = split_by_tree(pairs, 0.2, 0.2, seed=1)
    trees = {k: {tree_id(p[0]) for p in v} for k, v in split.items()}
    assert not (trees["train"] & trees["test"]) and not (trees["val"] & trees["test"])
    assert sum(len(v) for v in split.values()) == 40


def test_unet_output_shape():
    torch = pytest.importorskip("torch")
    from unet import UNet
    out = UNet(in_channels=4, base=4)(torch.zeros(2, 4, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_pixel_features_shape_and_position():
    from features import feature_names, pixel_features
    x = np.random.default_rng(0).random((4, 32, 32)).astype(np.float32)
    f = pixel_features(x)
    assert f.shape == (32 * 32, len(feature_names()))
    assert f[0, -2:].tolist() == [0.0, 0.0] and f[-1, -2:].tolist() == [1.0, 1.0]


def test_sample_pixels_is_balanced():
    from features import sample_pixels
    y = np.zeros((20, 20), bool)
    y[:, :2] = True                                  # 40 branch pixels, 360 background
    X, t = sample_pixels(np.arange(400.0)[:, None], y, 30, np.random.default_rng(0))
    assert t.sum() == 30 and len(t) == 60


def test_best_threshold_and_tree_learns_simple_mask():
    from features import pixel_features
    from train import best_threshold, build_model
    x = np.zeros((4, 32, 32), np.float32)
    x[:3, :, 14:18] = 0.5                            # grey-brown trunk on black background
    y = np.zeros((32, 32), bool)
    y[:, 14:18] = True
    model = build_model("tree", {}, seed=0).fit(pixel_features(x), y.ravel().astype(float))
    prob = model.predict(pixel_features(x)).reshape(32, 32)
    thr, d = best_threshold([prob], [y])
    assert d > 0.95 and 0.1 <= thr <= 0.9
