"""Entry point: run the baseline over the dataset and report metrics.

Usage:
    python src/main.py --config config/default.toml [--data-root PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from baseline import mask_to_skeleton, predict_mask
from dataset import find_pairs
from metrics import chamfer_distance, dice, iou, precision_recall
from utils import load_config, read_depth, read_image, read_mask


def evaluate_pair(rgb, depth, gt, params):
    """Run the baseline on one image and score it against the ground truth.

    Args:
        rgb: (H, W, 3) uint8 image.
        depth: (H, W) uint8 depth image, or None for RGB only.
        gt: (256, 256) boolean ground-truth mask.
        params: keyword arguments for ``baseline.predict_mask`` (the
            ``[baseline]`` section of the config).

    Returns:
        dict with keys "iou", "dice", "precision", "recall", "chamfer_px".

    Raises:
        ValueError: If the predicted mask and gt have different shapes.
    """
    pred = predict_mask(rgb, depth, **params)
    p, r = precision_recall(pred, gt)
    return {
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "precision": p,
        "recall": r,
        "chamfer_px": chamfer_distance(mask_to_skeleton(pred), mask_to_skeleton(gt)),
    }


def main(argv=None):
    """Evaluate the baseline on every image with ground truth and print mean metrics.

    Args:
        argv: Command-line arguments (None reads ``sys.argv``).

    Returns:
        0 on success, 1 if the data folder does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1

    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))
    print(f"Found {len(pairs)} {cfg['season']} images with ground truth under {root}")
    start = time.perf_counter()
    results = []
    for rgb_p, depth_p, gt_p in pairs:
        depth = read_depth(depth_p) if depth_p is not None else None
        results.append(evaluate_pair(read_image(rgb_p), depth, read_mask(gt_p), cfg["baseline"]))
    elapsed = time.perf_counter() - start

    if results:
        for key in results[0]:
            print(f"{key:>10}: {np.mean([r[key] for r in results]):.3f}")
    print(f"Evaluated {len(results)} images in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
