"""Profile the baseline on a synthetic image to find bottlenecks (Brief §6).

Usage: python tools/profile_baseline.py
"""

import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import mask_to_skeleton, predict_mask  # noqa: E402

rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
with cProfile.Profile() as prof:
    for _ in range(20):
        mask_to_skeleton(predict_mask(rgb))
pstats.Stats(prof).sort_stats("cumulative").print_stats(10)
