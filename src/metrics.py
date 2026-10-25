"""Evaluation metrics comparing a predicted branch mask with the ground truth.

Mask metrics (IoU, Dice, precision, recall) count pixel agreements. The
skeleton metric (symmetric Chamfer distance) measures how far apart two
one-pixel-wide centrelines are, in pixels.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def confusion(pred, gt):
    """Count pixel agreements between a predicted and a true mask.

    Args:
        pred: Boolean array, the predicted mask.
        gt: Boolean array of the same shape, the ground-truth mask.

    Returns:
        (tp, fp, fn, tn): true/false positive and negative pixel counts.

    Raises:
        ValueError: If pred and gt have different shapes.
    """
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    return int(tp), int(fp), int(fn), int(tn)


def iou(pred, gt):
    """Intersection over union, tp / (tp + fp + fn).

    Args:
        pred: Boolean array, the predicted mask.
        gt: Boolean array of the same shape, the ground-truth mask.

    Returns:
        float in [0, 1]; 1.0 when both masks are empty.

    Raises:
        ValueError: If pred and gt have different shapes.
    """
    tp, fp, fn, _ = confusion(pred, gt)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def dice(pred, gt):
    """Dice coefficient (equal to pixel F1), 2tp / (2tp + fp + fn).

    Args:
        pred: Boolean array, the predicted mask.
        gt: Boolean array of the same shape, the ground-truth mask.

    Returns:
        float in [0, 1]; 1.0 when both masks are empty.

    Raises:
        ValueError: If pred and gt have different shapes.
    """
    tp, fp, fn, _ = confusion(pred, gt)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def precision_recall(pred, gt):
    """Pixel precision and recall.

    Precision is the share of predicted branch pixels that are branch (low
    precision = false obstacles). Recall is the share of true branch pixels
    that were found (low recall = missed obstacles).

    Args:
        pred: Boolean array, the predicted mask.
        gt: Boolean array of the same shape, the ground-truth mask.

    Returns:
        (precision, recall) as floats in [0, 1]; each is 0.0 when its
        denominator is zero.

    Raises:
        ValueError: If pred and gt have different shapes.
    """
    tp, fp, fn, _ = confusion(pred, gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def chamfer_distance(skel_a, skel_b):
    """Symmetric mean distance (pixels) between two skeletons.

    For every pixel of each skeleton, the distance to the nearest pixel of the
    other is found with a Euclidean distance transform; the two means are
    averaged.

    Args:
        skel_a: Boolean array, first skeleton.
        skel_b: Boolean array of the same shape, second skeleton.

    Returns:
        float distance in pixels; ``inf`` if either skeleton is empty.
    """
    a, b = np.asarray(skel_a, bool), np.asarray(skel_b, bool)
    if not a.any() or not b.any():
        return float("inf")
    dist_to_b = distance_transform_edt(~b)
    dist_to_a = distance_transform_edt(~a)
    return 0.5 * (dist_to_b[a].mean() + dist_to_a[b].mean())
