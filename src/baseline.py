"""Classical baseline: colour + depth thresholding, then skeletonisation."""

import numpy as np
from scipy.ndimage import binary_closing
from skimage.morphology import disk, skeletonize
from skimage.transform import resize


def excess_green(rgb):
    """Excess-green index 2g - r - b on chromaticity-normalised RGB.

    Args:
        rgb: (H, W, 3) uint8 or float image.

    Returns:
        (H, W) float array, higher = greener (leaf-like).
    """
    rgb = rgb[..., :3].astype(float)
    total = rgb.sum(axis=2) + 1e-9
    r, g, b = (rgb[..., i] / total for i in range(3))
    return 2 * g - r - b


def centre_crop_square(img):
    """Crop the central square of an image (e.g. 640x480 -> 480x480).

    Assumed mask-to-image transform until the teaching team confirms it.

    Args:
        img: (H, W) or (H, W, C) array.

    Returns:
        (S, S) or (S, S, C) view, S = min(H, W).
    """
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img[top:top + s, left:left + s]


def predict_mask(rgb, depth=None, exg_threshold=0.05, depth_min=1, depth_max=255,
                 closing_radius=3, out_size=256):
    """Predict a binary branch mask from one RGB(-D) image.

    Pixels that are not leaf-green and fall inside the depth band are kept,
    then small gaps are closed. Occluded branches are NOT recovered; that is
    the limitation the learned methods are compared against.

    Args:
        rgb: (H, W, 3) image.
        depth: optional (H, W) 8-bit depth; 0 is treated as "no reading".
        exg_threshold: excess-green above this is classed as leaf.
        depth_min, depth_max: depth band kept as the target tree.
        closing_radius: radius of the morphological closing disk.
        out_size: side length of the output mask (GT masks are 256).

    Returns:
        (out_size, out_size) boolean mask.

    Raises:
        ValueError: If depth is given and is not the same size as rgb.
    """
    if depth is not None and depth.shape != rgb.shape[:2]:
        raise ValueError(f"depth {depth.shape} does not match RGB {rgb.shape[:2]}")
    mask = excess_green(rgb) < exg_threshold
    if depth is not None:
        mask &= (depth >= depth_min) & (depth <= depth_max)
    mask = centre_crop_square(mask)
    mask = resize(mask.astype(float), (out_size, out_size), order=0) > 0.5
    if closing_radius > 0:
        mask = binary_closing(mask, structure=disk(closing_radius))
    return mask


def mask_to_skeleton(mask):
    """Reduce a binary mask to a one-pixel-wide skeleton (Zhang's thinning).

    Args:
        mask: (H, W) boolean array.

    Returns:
        (H, W) boolean array, True on the centreline.
    """
    return skeletonize(mask)
