"""Manual check of the GT offset, used to verify the automatic alignment check (Brief §5).

``train.py`` estimates each GT mask's offset automatically
(``alignment.estimate_offset`` against the baseline mask). The baseline marks
about 70% of each image as wood, so that estimate needs checking against
manual inspection with a defined criterion. This tool draws overlay sheets of the bottom half of each image (GT outline in red,
column ticks every 8 px, reading row in yellow) for a fixed random sample of
images. The offset is read at row 230 of 256 as
(GT trunk centre column) - (visible trunk centre column); positive means the GT
is to the right of the trunk. Images where the trunk is hidden are left blank.
Readings go in res/gt_offset_manual.csv, which this tool summarises.

Usage:
    python tools/gt_offset.py [--n 24] [--seed 0]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alignment import estimate_offset  # noqa: E402
from baseline import predict_mask  # noqa: E402
from dataset import find_pairs, load_sample, tree_id  # noqa: E402
from utils import load_config, read_depth, read_image, read_mask  # noqa: E402

READ_ROW = 230  # row (of 256) where the manual offset is read


def summarise_manual(path):
    """Summary statistics of the filled-in manual offset readings.

    Args:
        path: CSV with a "dx" column; blank means the trunk was not visible.

    Returns:
        dict with the number of images, readable images, median and mean
        absolute offset, share with |dx| >= 15 px, and min / max dx; or None
        if nothing has been read yet.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    dx = np.array([float(r["dx"]) for r in rows if r["dx"].strip()])
    if len(dx) == 0:
        return None
    return {"images": len(rows), "readable": len(dx), "median_abs_dx": float(np.median(np.abs(dx))),
            "mean_abs_dx": float(np.mean(np.abs(dx))), "share_abs_ge_15": float(np.mean(np.abs(dx) >= 15)),
            "min_dx": float(dx.min()), "max_dx": float(dx.max())}


def main(argv=None):
    """Write the manual-inspection sheets and summarise the readings.

    Args:
        argv: Command-line arguments (None reads ``sys.argv``).

    Returns:
        0 on success, 1 if the data folder does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n", type=int, default=24, help="images in the manual-inspection sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--res", default="res")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))
    res = Path(args.res)
    res.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    sample = [pairs[i] for i in sorted(rng.choice(len(pairs), min(args.n, len(pairs)), replace=False))]
    per_sheet = 6
    for s in range(0, len(sample), per_sheet):
        chunk = sample[s:s + per_sheet]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(10, 5.4 * len(chunk)), squeeze=False)
        for ax, p in zip(axes[:, 0], chunk):
            x, y = load_sample(*p)
            ax.imshow(x[:3].transpose(1, 2, 0)[128:], extent=(0, 256, 256, 128))
            ax.contour(np.arange(256), np.arange(128, 256), y[128:], [0.5], colors="r", linewidths=1)
            ax.axhline(READ_ROW, color="yellow", lw=0.8)
            ax.set_xticks(range(0, 257, 8))
            ax.tick_params(labelsize=7)
            ax.grid(axis="x", color="white", lw=0.4, alpha=0.6)
            ax.set_title(f"{p[0].name} ({tree_id(p[0])})", fontsize=9)
        fig.tight_layout()
        out = res / f"gt_offset_sheet_{s // per_sheet + 1}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(f"Wrote {out}")
    template = res / "gt_offset_manual.csv"
    if not template.exists():
        with open(template, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "visible_trunk_col", "gt_trunk_col", "dx", "note"])
            for p in sample:
                w.writerow([p[0].name, "", "", "", ""])
        print(f"Wrote empty template {template}; fill it in from the sheets")
    summary = summarise_manual(template)
    if summary:
        print("Manual:", ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()))
        compare_with_automatic(template, pairs, cfg)
    return 0


def compare_with_automatic(path, pairs, cfg):
    """Print how well the automatic offset agrees with the manual readings.

    Args:
        path: Filled-in manual readings CSV.
        pairs: List of ``(rgb_path, depth_path, gt_path)`` from ``find_pairs``.
        cfg: Loaded config (uses ``[baseline]`` and ``[eval] align_max_shift``).

    Returns:
        (correlation, mean absolute difference in px, share with the same sign).
    """
    by_name = {p[0].name: p for p in pairs}
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dx"].strip()]
    manual, auto = [], []
    for r in rows:
        rgb_p, d_p, gt_p = by_name[r["image"]]
        visible = predict_mask(read_image(rgb_p), read_depth(d_p) if d_p else None, **cfg["baseline"])
        _, dx, _ = estimate_offset(read_mask(gt_p), visible, cfg.get("eval", {}).get("align_max_shift", 50))
        manual.append(float(r["dx"]))
        auto.append(-dx)  # estimate_offset moves the GT onto the trunk; manual dx is GT minus trunk
    manual, auto = np.array(manual), np.array(auto)
    corr = float(np.corrcoef(manual, auto)[0, 1])
    diff = float(np.mean(np.abs(manual - auto)))
    same = float(np.mean(np.sign(manual) == np.sign(auto)))
    print(f"Automatic vs manual on {len(manual)} images: correlation {corr:.2f}, "
          f"mean |difference| {diff:.1f} px, same direction {same:.0%}")
    return corr, diff, same


if __name__ == "__main__":
    sys.exit(main())
