"""Evaluation metrics comparing predicted and ground-truth branch masks."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def confusion(pred, gt):
    """Return (tp, fp, fn, tn) pixel counts for two boolean masks."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)
    return int(tp), int(fp), int(fn), int(tn)


def iou(pred, gt):
    """Intersection over union. Returns 1.0 when both masks are empty."""
    tp, fp, fn, _ = confusion(pred, gt)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def dice(pred, gt):
    """Dice coefficient (equal to pixel F1). Returns 1.0 when both are empty."""
    tp, fp, fn, _ = confusion(pred, gt)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def precision_recall(pred, gt):
    """Pixel precision and recall as a tuple."""
    tp, fp, fn, _ = confusion(pred, gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def chamfer_distance(skel_a, skel_b):
    """Symmetric mean distance (pixels) between two skeletons.

    Returns ``inf`` if either skeleton is empty.
    """
    a, b = np.asarray(skel_a, bool), np.asarray(skel_b, bool)
    if not a.any() or not b.any():
        return float("inf")
    dist_to_b = distance_transform_edt(~b)
    dist_to_a = distance_transform_edt(~a)
    return 0.5 * (dist_to_b[a].mean() + dist_to_a[b].mean())
