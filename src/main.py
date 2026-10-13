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
from metrics import chamfer_distance, dice, iou, precision_recall
from utils import find_samples, load_config, read_image, read_mask


def evaluate_pair(rgb, depth, gt, params):
    """Run the baseline on one image and return a dict of metrics."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1

    files = find_samples(root, cfg["season"], cfg.get("exclude_rows", []))
    # TODO: pair RGB / depth / GT files once naming is confirmed (doc/problem_statement.md §7).
    print(f"Found {len(files)} {cfg['season']} images under {root}")
    start = time.perf_counter()
    results = []
    for f in files:
        name = f.name
        if "RGB" not in name or "GT" in name:
            continue
        gt_path = f.with_name(name.replace("RGB", "GT"))
        depth_path = f.with_name(name.replace("RGB", "Depth"))
        if not gt_path.exists():
            continue
        depth = read_image(depth_path) if depth_path.exists() else None
        results.append(evaluate_pair(read_image(f), depth, read_mask(gt_path), cfg["baseline"]))
    elapsed = time.perf_counter() - start

    if results:
        for key in results[0]:
            print(f"{key:>10}: {np.mean([r[key] for r in results]):.3f}")
    print(f"Evaluated {len(results)} images in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
