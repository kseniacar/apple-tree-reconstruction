"""Main method: per-pixel regression models compared with the baseline.

Each pixel gets colour, depth, neighbourhood and position features
(features.py). Linear regression, a decision tree, support vector regression
and an MLP each predict how likely a pixel is to be branch. Images are split by
tree; validation trees set each model's threshold and the held-out test trees
are only used for the final comparison with the baseline and the shape prior
(the mean training mask, which uses no image information).

Usage:
    python src/train.py --limit 40          # quick check
    python src/train.py                     # all models, settings in config/default.toml [ml]
    python src/train.py --work-size 64      # resolution sensitivity study
    python src/train.py --no-depth          # RGB only: does depth help?
    python src/train.py --no-position       # without row/column features

Outputs (tag is "", "_rgb" and/or "_nopos"):
    outputs/<model><tag>.joblib            trained models, not committed
    res/ml_results<tag>.csv                test-set metrics and timings
    res/ml_validation<tag>.csv             chosen thresholds and validation Dice
    res/tree_feature_importance<tag>.csv   decision-tree feature importance
    res/ml_examples<tag>.png               RGB / ground truth / predictions on test images
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from baseline import mask_to_skeleton, predict_mask  # noqa: E402
from dataset import find_pairs, load_sample, split_by_tree  # noqa: E402
from features import downsample, feature_names, pixel_features, sample_pixels, upsample_map  # noqa: E402
from metrics import chamfer_distance, dice, iou, precision_recall  # noqa: E402
from utils import load_config, read_depth, read_image, read_mask  # noqa: E402

ALL_MODELS = ["linear", "tree", "svr", "mlp"]


def build_model(name, cfg, seed):
    """Create an untrained scikit-learn regressor; inputs are standardised first.

    Args:
        name: "linear", "tree", "svr" or "mlp".
        cfg: The ``[ml]`` settings from config/default.toml.
        seed: Random seed for repeatable results.

    Returns:
        A scikit-learn Pipeline (StandardScaler, then the model).

    Raises:
        ValueError: If name is not one of the four models.
    """
    if name == "linear":
        reg = LinearRegression()
    elif name == "tree":
        reg = DecisionTreeRegressor(max_depth=cfg.get("tree_max_depth", 12),
                                    min_samples_leaf=cfg.get("tree_min_samples_leaf", 20), random_state=seed)
    elif name == "svr":
        reg = SVR(kernel="rbf", C=cfg.get("svr_c", 1.0), epsilon=cfg.get("svr_epsilon", 0.1))
    elif name == "mlp":
        reg = MLPRegressor(hidden_layer_sizes=tuple(cfg.get("mlp_hidden", [64, 32])), early_stopping=True,
                           max_iter=cfg.get("mlp_max_iter", 200), random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}, choose from {ALL_MODELS}")
    return make_pipeline(StandardScaler(), reg)


def load_images(pairs, work_size, use_depth, use_position=True):
    """Load samples and compute their pixel features.

    Args:
        pairs: List of ``(rgb_path, depth_path, gt_path)``.
        work_size: Resolution the features are computed at.
        use_depth: If False, the depth channel is set to zero.
        use_position: Include the row/column features.

    Returns:
        list of dicts with keys "x" (input at 256), "feats" (pixel features at
        work size), "y_small" (mask at work size) and "y" (mask at 256).
    """
    out = []
    for i, p in enumerate(pairs, 1):
        x, y = load_sample(*p)
        if not use_depth:
            x[3] = 0
        xs, ys = downsample(x, y > 0.5, work_size)
        out.append({"x": x, "feats": pixel_features(xs, position=use_position), "y_small": ys, "y": y > 0.5})
        if i % 100 == 0:
            print(f"  loaded {i}/{len(pairs)}", flush=True)
    return out


def predict_prob(model, img, work_size, out_size=256):
    """Predict a branch value map for one image.

    Args:
        model: Fitted regressor from ``build_model``.
        img: One dict from ``load_images``.
        work_size: Resolution the features were computed at.
        out_size: Output side length (the ground-truth size).

    Returns:
        (out_size, out_size) float array clipped to [0, 1].
    """
    small = model.predict(img["feats"]).reshape(work_size, work_size)
    return np.clip(upsample_map(small, out_size), 0, 1)


def score(pred, gt):
    """Metrics for one predicted mask against ground truth.

    Args:
        pred: (256, 256) boolean predicted mask.
        gt: (256, 256) boolean ground-truth mask.

    Returns:
        dict with keys "iou", "dice", "precision", "recall", "chamfer_px".
    """
    p, r = precision_recall(pred, gt)
    return {"iou": iou(pred, gt), "dice": dice(pred, gt), "precision": p, "recall": r,
            "chamfer_px": chamfer_distance(mask_to_skeleton(pred), mask_to_skeleton(gt))}


def mean_scores(rows):
    """Average each metric over a list of ``score`` dicts."""
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def best_threshold(probs, gts, candidates=np.linspace(0.1, 0.9, 17)):
    """Threshold that maximises mean Dice on the validation images.

    Args:
        probs: List of (H, W) branch value maps.
        gts: List of (H, W) boolean ground-truth masks, same order.
        candidates: Thresholds to try.

    Returns:
        (threshold, mean validation Dice at that threshold).
    """
    scores = [np.mean([dice(p > t, g) for p, g in zip(probs, gts)]) for t in candidates]
    i = int(np.argmax(scores))
    return float(candidates[i]), float(scores[i])


def main(argv=None):
    """Train the chosen models, compare them on the test trees and write results.

    Args:
        argv: Command-line arguments (None reads ``sys.argv``).

    Returns:
        0 on success, 1 if the data folder is missing or has fewer than 3
        images with ground truth.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--work-size", type=int, default=None, help="working resolution in pixels")
    parser.add_argument("--no-depth", action="store_true", help="set the depth channel to zero")
    parser.add_argument("--no-position", action="store_true", help="leave out the row/column features")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images (quick test)")
    parser.add_argument("--out", default="outputs", help="folder for trained models")
    parser.add_argument("--res", default="res", help="folder for results and figures")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    mcfg = cfg.get("ml", {})
    seed = mcfg.get("seed", 0)
    work = args.work_size or mcfg.get("work_size", 128)
    rng = np.random.default_rng(seed)

    root = Path(args.data_root or cfg["data_root"])
    if not root.exists():
        print(f"Data folder not found: {root}. See README 'Data'.", file=sys.stderr)
        return 1
    pairs = find_pairs(root, cfg["season"], cfg.get("exclude_rows", []))[:args.limit]
    if len(pairs) < 3:
        print(f"Only {len(pairs)} images with ground truth found under {root}.", file=sys.stderr)
        return 1
    split = split_by_tree(pairs, mcfg.get("val_frac", 0.15), mcfg.get("test_frac", 0.15), seed)
    print(f"{len(pairs)} images: {len(split['train'])} train, {len(split['val'])} val, "
          f"{len(split['test'])} test (split by tree), working size {work}x{work}")

    print("Loading images and computing features...")
    data = {k: load_images(v, work, not args.no_depth, not args.no_position) for k, v in split.items()}
    n_per_class = mcfg.get("pixels_per_class", 500)
    parts = [sample_pixels(img["feats"], img["y_small"], n_per_class, rng) for img in data["train"]]
    X = np.concatenate([p[0] for p in parts])
    t = np.concatenate([p[1] for p in parts])
    print(f"Training set: {len(X):,} pixels x {X.shape[1]} features")

    out_dir, res_dir = Path(args.out), Path(args.res)
    out_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    tag = ("_rgb" if args.no_depth else "") + ("_nopos" if args.no_position else "")
    results, val_rows, test_masks = {}, [], {}

    # Baseline on the same test trees, for a like-for-like comparison.
    t0 = time.perf_counter()
    base_rows, base_masks = [], []
    for rgb_p, d_p, gt_p in split["test"]:
        depth = read_depth(d_p) if d_p is not None and not args.no_depth else None
        m = predict_mask(read_image(rgb_p), depth, **cfg["baseline"])
        base_masks.append(m)
        base_rows.append(score(m, read_mask(gt_p)))
    results["baseline"] = mean_scores(base_rows)
    results["baseline"].update(fit_s=0.0, sec_per_image=(time.perf_counter() - t0) / len(base_rows))
    test_masks["baseline"] = base_masks

    # Shape prior: the mean training mask, thresholded on validation, predicted for every test image.
    mean_mask = np.mean([img["y"] for img in data["train"]], axis=0)
    thr, val_dice = best_threshold([mean_mask] * len(data["val"]), [img["y"] for img in data["val"]])
    val_rows.append({"model": "prior", "threshold": thr, "val_dice": val_dice})
    prior = mean_mask > thr
    results["prior"] = mean_scores([score(prior, img["y"]) for img in data["test"]])
    results["prior"].update(fit_s=0.0, sec_per_image=0.0)
    test_masks["prior"] = [prior] * len(data["test"])

    for name in args.models:
        model = build_model(name, mcfg, seed)
        Xf, tf = X, t
        if name == "svr":  # SVR training cost grows ~quadratically with samples, so use a subset
            n = min(mcfg.get("svr_samples", 3000), len(X))
            keep = rng.choice(len(X), n, replace=False)
            Xf, tf = X[keep], t[keep]
        print(f"Training {name} on {len(Xf):,} pixels...", flush=True)
        t0 = time.perf_counter()
        model.fit(Xf, tf)
        fit_s = time.perf_counter() - t0
        joblib.dump(model, out_dir / f"{name}{tag}.joblib")

        val_probs = [predict_prob(model, img, work) for img in data["val"]]
        thr, val_dice = best_threshold(val_probs, [img["y"] for img in data["val"]])
        val_rows.append({"model": name, "threshold": thr, "val_dice": val_dice})

        t0 = time.perf_counter()
        masks = [predict_prob(model, img, work) > thr for img in data["test"]]
        pred_s = (time.perf_counter() - t0) / len(masks)
        results[name] = mean_scores([score(m, img["y"]) for m, img in zip(masks, data["test"])])
        results[name].update(fit_s=fit_s, sec_per_image=pred_s)
        test_masks[name] = masks
        print(f"  fit {fit_s:.1f} s, threshold {thr:.2f} (val Dice {val_dice:.3f}), "
              f"test Dice {results[name]['dice']:.3f}", flush=True)
        if name == "tree":
            imp = model[-1].feature_importances_
            with open(res_dir / f"tree_feature_importance{tag}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["feature", "importance"])
                for fname, v in sorted(zip(feature_names(position=not args.no_position), imp), key=lambda z: -z[1]):
                    w.writerow([fname, f"{v:.4f}"])

    keys = list(results["baseline"])
    print(f"\nTest set: {len(split['test'])} images from trees not used in training")
    print(f"{'':>10}" + "".join(f"{k:>14}" for k in keys))
    for name, r in results.items():
        print(f"{name:>10}" + "".join(f"{r[k]:>14.3f}" for k in keys))
    with open(res_dir / f"ml_results{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for name, r in results.items():
            w.writerow([name] + [f"{r[k]:.4f}" for k in keys])
    with open(res_dir / f"ml_validation{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "threshold", "val_dice"])
        w.writeheader()
        w.writerows(val_rows)

    cols = ["RGB", "ground truth"] + list(test_masks)
    n = min(3, len(data["test"]))
    fig, axes = plt.subplots(n, len(cols), figsize=(2.3 * len(cols), 2.4 * n), squeeze=False)
    for i in range(n):
        img = data["test"][i]
        panels = [img["x"][:3].transpose(1, 2, 0), img["y"]] + [test_masks[k][i] for k in test_masks]
        for ax, panel, title in zip(axes[i], panels, cols):
            ax.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            ax.set_title(title if i == 0 else "")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(res_dir / f"ml_examples{tag}.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved models to {out_dir}/ and results/figures to {res_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
