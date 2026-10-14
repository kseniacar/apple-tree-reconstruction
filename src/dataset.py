"""Load paired RGB / depth / ground-truth TreeID samples and split them by tree."""

from pathlib import Path

import numpy as np
from skimage.transform import resize

from baseline import centre_crop_square
from utils import read_image, read_mask


def find_pairs(data_root, season="Summer", exclude_rows=()):
    """List (rgb, depth, gt) paths for every image that has a ground-truth mask.

    TreeID names look like ``R01N01_Summer_RGB-12-40-50.png`` with matching
    ``_D-`` (depth) and ``_GT-`` files in the same folder.

    Args:
        data_root: Folder containing the dataset (searched recursively).
        season: "Summer" or "Winter".
        exclude_rows: Row prefixes such as ``"R04"`` to skip.

    Returns:
        List of ``(rgb_path, depth_path_or_None, gt_path)`` tuples.
    """
    pairs = []
    for rgb in sorted(Path(data_root).rglob(f"*_{season}_RGB-*.png")):
        if any(rgb.name.startswith(r) for r in exclude_rows):
            continue
        gt = rgb.with_name(rgb.name.replace("_RGB-", "_GT-"))
        if not gt.exists():
            continue
        depth = rgb.with_name(rgb.name.replace("_RGB-", "_D-"))
        pairs.append((rgb, depth if depth.exists() else None, gt))
    return pairs


def tree_id(path):
    """Tree identifier from a file name, e.g. ``R01N01``."""
    return Path(path).name.split("_")[0]


def split_by_tree(pairs, val_frac=0.15, test_frac=0.15, seed=0):
    """Split samples into train / val / test so no tree appears in two sets.

    Returns:
        dict with keys "train", "val", "test", each a list of pairs.
    """
    trees = sorted({tree_id(p[0]) for p in pairs})
    np.random.default_rng(seed).shuffle(trees)
    n_test = round(len(trees) * test_frac)
    n_val = round(len(trees) * val_frac)
    test, val = set(trees[:n_test]), set(trees[n_test:n_test + n_val])
    split = {"train": [], "val": [], "test": []}
    for p in pairs:
        t = tree_id(p[0])
        split["test" if t in test else "val" if t in val else "train"].append(p)
    return split


def load_sample(rgb_path, depth_path, gt_path, size=256):
    """Load one sample as network input and target.

    The RGB-D image is centre-cropped to a square (same assumption as the
    baseline) and resized to ``size``.

    Returns:
        x: (4, size, size) float32 in [0, 1], channels R, G, B, depth.
        y: (size, size) float32 mask, 1 = branch.
    """
    rgb = read_image(rgb_path)[..., :3]
    if depth_path is None:
        depth = np.zeros(rgb.shape[:2], np.uint8)
    else:
        depth = read_image(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]  # greyscale stored as 3 identical channels
    x = centre_crop_square(np.dstack([rgb, depth]))
    x = resize(x, (size, size), order=1, preserve_range=True, anti_aliasing=True)
    y = read_mask(gt_path)
    if y.shape != (size, size):
        y = resize(y.astype(float), (size, size), order=0) > 0.5
    return (x / 255).astype(np.float32).transpose(2, 0, 1), y.astype(np.float32)
