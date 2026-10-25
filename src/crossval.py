"""k-fold cross-validation by tree, for choosing between the regression models.

``train.py`` uses one fixed train / validation / test split, so each model's
score comes from a single validation set. This script re-uses the same models
and features but rotates the validation trees through k folds of the
non-test trees, so every model gets a mean and a spread. The test trees (same
split and seed as ``train.py``) are never used here, so the final test result
stays independent of model selection.

In each fold the model is trained on the other k-1 folds and its threshold is
chosen on the held-out fold, so the fold Dice is slightly optimistic; it is
meant for ranking the models, not as the final accuracy.

Usage:
    python src/crossval.py                       # all models, k from config/default.toml [eval]
    python src/crossval.py --models tree mlp -k 5
    python src/crossval.py --limit 60            # quick check

Outputs:
    res/cv_folds.csv     validation Dice and threshold per model and fold
    res/cv_summary.csv   mean and standard deviation of the fold Dice per model
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dataset import find_pairs, split_by_tree, tree_id
from features import sample_pixels
from train import ALL_MODELS, best_threshold, build_model, load_images, predict_prob
from utils import load_config


def kfold_by_tree(pairs, k=5, seed=0):
    """Split samples into k folds so that each tree is in exactly one fold.

    Args:
        pairs: List of ``(rgb_path, depth_path_or_None, gt_path)`` tuples.
        k: Number of folds.
        seed: Seed for the random shuffle of tree IDs.

    Returns:
        List of k ``(train_pairs, val_pairs)`` tuples.

    Raises:
        ValueError: If there are fewer distinct trees than folds.
    """
    trees = sorted({tree_id(p[0]) for p in pairs})
    if len(trees) < k:
        raise ValueError(f"{len(trees)} trees cannot be split into {k} folds")
    np.random.default_rng(seed).shuffle(trees)
    fold_of = {t: i % k for i, t in enumerate(trees)}
    folds = []
    for i in range(k):
        train = [p for p in pairs if fold_of[tree_id(p[0])] != i]
        val = [p for p in pairs if fold_of[tree_id(p[0])] == i]
        folds.append((train, val))
    return folds


def main(argv=None):
    """Run k-fold cross-validation of the chosen models on the non-test trees.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Exit code: 0 on success, 1 if the data folder is missing or too small.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("-k", type=int, default=None, help="number of folds")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images (quick test)")
    parser.add_argument("--res", default="res", help="folder for results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg = cfg.get("ml", {})
    k = args.k or cfg.get("eval", {}).get("cv_folds", 5)
    seed = mcfg.get("seed", 0)
    work = mcfg.get("work_size", 128)
    rng = np.random.default_rng(seed)

    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))[:args.limit]
    split = split_by_tree(pairs, mcfg.get("val_frac", 0.15), mcfg.get("test_frac", 0.15), seed)
    pool = split["train"] + split["val"]  # the test trees stay untouched
    try:
        folds = kfold_by_tree(pool, k, seed)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    print(f"{len(pool)} non-test images in {k} folds by tree (test trees excluded)")

    print("Loading images and computing features...")
    data = dict(zip(pool, load_images(pool, work, use_depth=True)))
    n_per_class = mcfg.get("pixels_per_class", 500)

    rows = []
    for i, (train, val) in enumerate(folds):
        parts = [sample_pixels(data[p]["feats"], data[p]["y_small"], n_per_class, rng) for p in train]
        X = np.concatenate([q[0] for q in parts])
        t = np.concatenate([q[1] for q in parts])
        for name in args.models:
            Xf, tf = X, t
            if name == "svr":  # same subset rule as train.py
                keep = rng.choice(len(X), min(mcfg.get("svr_samples", 3000), len(X)), replace=False)
                Xf, tf = X[keep], t[keep]
            model = build_model(name, mcfg, seed).fit(Xf, tf)
            probs = [predict_prob(model, data[p], work) for p in val]
            thr, val_dice = best_threshold(probs, [data[p]["y"] for p in val])
            rows.append({"model": name, "fold": i, "threshold": thr, "val_dice": val_dice})
            print(f"  fold {i + 1}/{k} {name:>6}: Dice {val_dice:.3f} at threshold {thr:.2f}", flush=True)

    res_dir = Path(args.res)
    res_dir.mkdir(exist_ok=True)
    with open(res_dir / "cv_folds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fold", "threshold", "val_dice"])
        w.writeheader()
        w.writerows(rows)
    with open(res_dir / "cv_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_dice", "std_dice", "folds"])
        print(f"\n{k}-fold cross-validation (mean +/- standard deviation of fold Dice):")
        for name in args.models:
            d = [r["val_dice"] for r in rows if r["model"] == name]
            w.writerow([name, f"{np.mean(d):.4f}", f"{np.std(d, ddof=1):.4f}", len(d)])
            print(f"{name:>8}: {np.mean(d):.3f} +/- {np.std(d, ddof=1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
