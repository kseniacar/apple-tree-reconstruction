"""Where do the methods fail? Per-image scores on the test trees (Brief §5, failure modes).

Loads the decision tree saved by ``python src/train.py`` (outputs/tree.joblib)
and its validation threshold (res/ml_validation.csv), scores the tree, the
shape prior and the baseline on every test image, and relates the tree's Dice
to the GT offset estimated by train.py (res/gt_offsets.csv) and to how much
of the image the target tree's wood covers.

Usage: python tools/failure_cases.py   (after python src/train.py)

Outputs:
    res/per_image_scores.csv   Dice per test image for tree, prior and baseline
    res/failure_cases.png      the 3 worst and 3 best test images for the tree
"""

import csv
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import predict_mask  # noqa: E402
from dataset import find_pairs, split_by_tree  # noqa: E402
from metrics import dice  # noqa: E402
from train import best_threshold, load_images, predict_prob  # noqa: E402
from utils import load_config, read_depth, read_image  # noqa: E402


def main():
    """Score every test image, write the CSV and the figure, and print a summary.

    Returns:
        0 on success, 1 if the trained model or the data folder is missing.
    """
    cfg = load_config("config/default.toml")
    mcfg = cfg["ml"]
    model_path, root = Path("outputs/tree.joblib"), Path(cfg["data_root"])
    if not model_path.exists() or not root.exists():
        print("Needs outputs/tree.joblib (run python src/train.py first) and the dataset.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))
    split = split_by_tree(pairs, mcfg["val_frac"], mcfg["test_frac"], mcfg["seed"])
    work = mcfg["work_size"]
    with open("res/ml_validation.csv") as f:
        thr = {r["model"]: float(r["threshold"]) for r in csv.DictReader(f)}
    with open("res/gt_offsets.csv") as f:
        offset = {r["image"]: abs(int(r["dx"])) for r in csv.DictReader(f)}

    model = joblib.load(model_path)
    train_masks = [img["y"] for img in load_images(split["train"], work, use_depth=True)]
    test = load_images(split["test"], work, use_depth=True)
    # Same prior as train.py: mean training mask, thresholded at the validation-chosen value.
    prior = np.mean(train_masks, axis=0) > thr["prior"]

    rows = []
    for (rgb_p, d_p, _), img in zip(split["test"], test):
        tree_mask = predict_prob(model, img, work) > thr["tree"]
        base = predict_mask(read_image(rgb_p), read_depth(d_p) if d_p else None, **cfg["baseline"])
        rows.append({"image": rgb_p.name, "dice_tree": dice(tree_mask, img["y"]), "dice_prior": dice(prior, img["y"]),
                     "dice_baseline": dice(base, img["y"]), "gt_frac": float(img["y"].mean()),
                     "offset_px": offset.get(rgb_p.name, np.nan), "mask": tree_mask, "img": img})

    with open("res/per_image_scores.csv", "w", newline="") as f:
        keys = ["image", "dice_tree", "dice_prior", "dice_baseline", "gt_frac", "offset_px"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k] for k in keys})

    d = np.array([r["dice_tree"] for r in rows])
    off = np.array([r["offset_px"] for r in rows], float)
    frac = np.array([r["gt_frac"] for r in rows])
    beats_prior = np.mean(d > np.array([r["dice_prior"] for r in rows]))
    print(f"Tree Dice over {len(d)} test images: min {d.min():.3f}, quartiles "
          f"{np.percentile(d, 25):.3f} / {np.median(d):.3f} / {np.percentile(d, 75):.3f}, max {d.max():.3f}")
    print(f"Tree beats the prior on {beats_prior:.0%} of test images")
    print(f"Correlation of tree Dice with |GT offset|: {np.corrcoef(d, off)[0, 1]:.2f}; "
          f"with GT area fraction: {np.corrcoef(d, frac)[0, 1]:.2f}")
    for label, sel in (("|offset| < 10 px", off < 10), ("|offset| >= 20 px", off >= 20)):
        print(f"  mean tree Dice, {label}: {d[sel].mean():.3f} ({sel.sum()} images)")

    order = np.argsort(d)
    picks = list(order[:3]) + list(order[-3:])
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.5 * len(picks)), squeeze=False)
    for ax_row, i in zip(axes, picks):
        r = rows[i]
        ax_row[0].imshow(r["img"]["x"][:3].transpose(1, 2, 0))
        ax_row[0].set_title(f"{r['image'][:6]}  |offset| {r['offset_px']:.0f} px", fontsize=8)
        ax_row[1].imshow(r["img"]["y"], cmap="gray")
        ax_row[1].set_title("ground truth", fontsize=8)
        ax_row[2].imshow(r["mask"], cmap="gray")
        ax_row[2].set_title(f"tree, Dice {r['dice_tree']:.2f}" + (" (worst)" if i in order[:3] else " (best)"), fontsize=8)
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("res/failure_cases.png", dpi=100)
    plt.close(fig)
    print("Wrote res/per_image_scores.csv and res/failure_cases.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
