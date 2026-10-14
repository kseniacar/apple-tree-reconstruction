"""Per-pixel features for the regression models: colour, depth, context and position."""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

BASE_CHANNELS = ["r", "g", "b", "depth", "exg"]


def feature_names(scales=(0, 2, 6), position=True):
    """Names of the columns returned by ``pixel_features`` (same order)."""
    names = [f"{c}_s{s}" for s in scales for c in BASE_CHANNELS]
    return names + (["row", "col"] if position else [])


def pixel_features(x, scales=(0, 2, 6), position=True):
    """Build one feature vector per pixel.

    Each pixel gets its own R, G, B, depth and excess-green values, the same
    values Gaussian-smoothed at each scale in ``scales`` (local context, which
    lets a model guess at branches hidden behind leaves), and its normalised
    row/column position.

    Args:
        x: (4, H, W) float array in [0, 1], channels R, G, B, depth.
        scales: Gaussian sigmas in pixels; 0 means the raw value.
        position: include the row/column features.

    Returns:
        (H * W, F) float32 array, F = 5 * len(scales) (+ 2 with position).
    """
    r, g, b, d = x[:4]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    feats = []
    for sigma in scales:
        for ch in (r, g, b, d, exg):
            feats.append(ch if sigma == 0 else gaussian_filter(ch, sigma))
    if position:
        h, w = r.shape
        rows, cols = np.mgrid[0:h, 0:w]
        feats += [rows / (h - 1), cols / (w - 1)]
    return np.stack(feats, axis=-1).reshape(-1, len(feats)).astype(np.float32)


def downsample(x, y, size):
    """Resize a (C, H, W) input (bilinear) and (H, W) mask (nearest) to ``size``."""
    if x.shape[-1] == size:
        return x, y
    xs = resize(x.transpose(1, 2, 0), (size, size), order=1, anti_aliasing=True).transpose(2, 0, 1)
    ys = resize(y.astype(float), (size, size), order=0) > 0.5
    return xs.astype(np.float32), ys


def sample_pixels(feats, y, n_per_class, rng):
    """Pick a class-balanced random subset of pixels for training.

    Branch pixels are rare (a few percent of the image), so without balancing
    a model can score well by predicting "no branch" everywhere.

    Returns:
        (X, t): features and 0/1 targets of the chosen pixels.
    """
    y = y.ravel()
    idx = []
    for cls in (True, False):
        pool = np.flatnonzero(y == cls)
        if len(pool):
            idx.append(rng.choice(pool, min(n_per_class, len(pool)), replace=False))
    idx = np.concatenate(idx)
    return feats[idx], y[idx].astype(np.float32)


def upsample_map(prob, size):
    """Bilinear interpolation of a (h, w) probability map up to (size, size)."""
    if prob.shape[0] == size:
        return prob
    return resize(prob, (size, size), order=1)
