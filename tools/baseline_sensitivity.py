"""Sensitivity of the baseline to its two thresholds (Brief §6).

Sweeps the excess-green threshold and the lower edge of the depth band on the
validation trees (same split and seed as src/train.py), picks the setting with
the best mean Dice there, and reports it on the test trees next to the default
setting from config/default.toml.

Usage: python tools/baseline_sensitivity.py [--limit N]

Output: res/baseline_sensitivity.csv (validation metrics for every setting).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import predict_mask  # noqa: E402
from dataset import find_pairs, split_by_tree  # noqa: E402
from metrics import dice, precision_recall  # noqa: E402
from utils import load_config, read_depth, read_image, read_mask  # noqa: E402

EXG_THRESHOLDS = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
DEPTH_MINS = [1, 100, 120, 140]


def evaluate(images, params):
    """Mean Dice, precision and recall of the baseline over loaded images.

    Args:
        images: List of ``(rgb, depth, gt)`` arrays.
        params: Keyword arguments for ``baseline.predict_mask``.

    Returns:
        dict with keys "dice", "precision", "recall".
    """
    rows = []
    for rgb, depth, gt in images:
        pred = predict_mask(rgb, depth, **params)
        p, r = precision_recall(pred, gt)
        rows.append((dice(pred, gt), p, r))
    d, p, r = np.mean(rows, axis=0)
    return {"dice": float(d), "precision": float(p), "recall": float(r)}


def main(argv=None):
    """Run the sweep and print the default and best settings on the test trees.

    Args:
        argv: Command-line arguments (None reads ``sys.argv``).

    Returns:
        0 on success, 1 if the data folder does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--res", default="res")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg = cfg.get("ml", {})
    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))[:args.limit]
    split = split_by_tree(pairs, mcfg.get("val_frac", 0.15), mcfg.get("test_frac", 0.15), mcfg.get("seed", 0))

    def load(ps):
        """Read (rgb, depth, gt) arrays for a list of pairs."""
        return [(read_image(r), read_depth(d) if d else None, read_mask(g)) for r, d, g in ps]

    val, test = load(split["val"]), load(split["test"])
    default = dict(cfg["baseline"])
    rows = []
    for exg in EXG_THRESHOLDS:
        for dmin in DEPTH_MINS:
            params = {**default, "exg_threshold": exg, "depth_min": dmin}
            rows.append({"exg_threshold": exg, "depth_min": dmin, **evaluate(val, params)})
            print(f"exg {exg:+.2f}  depth_min {dmin:>3}: val Dice {rows[-1]['dice']:.3f}", flush=True)
    res = Path(args.res)
    res.mkdir(exist_ok=True)
    with open(res / "baseline_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    best = max(rows, key=lambda r: r["dice"])
    best_params = {**default, "exg_threshold": best["exg_threshold"], "depth_min": best["depth_min"]}
    for label, params in (("default", default), ("best on validation", best_params)):
        r = evaluate(test, params)
        print(f"Test, {label} (exg {params['exg_threshold']}, depth_min {params['depth_min']}): "
              f"Dice {r['dice']:.3f}, precision {r['precision']:.3f}, recall {r['recall']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
