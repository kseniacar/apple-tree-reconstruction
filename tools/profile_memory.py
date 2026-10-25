"""Measure peak memory of the baseline and of the per-pixel feature pipeline (Brief §6).

Uses ``tracemalloc``, which tracks Python and NumPy allocations, on synthetic
640 x 480 RGB-D images, so no dataset is needed. Also estimates the size of the
training feature matrix that ``src/train.py`` builds.

Usage: python tools/profile_memory.py
"""

import sys
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import mask_to_skeleton, predict_mask  # noqa: E402
from features import downsample, feature_names, pixel_features  # noqa: E402


def peak_mb(fn):
    """Run ``fn`` and return the peak memory it allocated, in MB."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


rng = np.random.default_rng(0)
rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
depth = rng.integers(1, 255, (480, 640), dtype=np.uint8)
x = rng.random((4, 256, 256)).astype(np.float32)
y = np.zeros((256, 256), bool)

print("Peak memory per image (tracemalloc):")
print(f"  baseline predict_mask + skeleton   {peak_mb(lambda: mask_to_skeleton(predict_mask(rgb, depth))):7.1f} MB")
for size in (64, 128, 256):
    mb = peak_mb(lambda: pixel_features(downsample(x, y, size)[0]))
    print(f"  pixel features at {size:>3} x {size:<3}         {mb:7.1f} MB")

n_feat = len(feature_names())
print("\nFeature matrices kept in memory by src/train.py (float32):")
for size in (64, 128, 256):
    per_image = size * size * n_feat * 4 / 1e6
    print(f"  {size:>3} x {size:<3}: {per_image:6.2f} MB per image, {per_image * 541 / 1e3:5.2f} GB for all 541 images")
x_mb = 4 * 256 * 256 * 4 / 1e6
print(f"  plus the (4, 256, 256) input kept per image: {x_mb:.2f} MB, {x_mb * 541 / 1e3:.2f} GB for all 541 images")
train_px = 379 * 1000
print(f"  training set (379 images x 1,000 sampled pixels x {n_feat} features): {train_px * n_feat * 4 / 1e6:.1f} MB")
