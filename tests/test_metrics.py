"""Tests for metrics.py on small synthetic masks with known answers."""

import numpy as np
import pytest

from metrics import chamfer_distance, dice, iou, precision_recall


def square(size=10, lo=2, hi=6):
    m = np.zeros((size, size), bool)
    m[lo:hi, lo:hi] = True
    return m


def test_identical_masks_score_one():
    m = square()
    assert iou(m, m) == 1.0
    assert dice(m, m) == 1.0
    assert precision_recall(m, m) == (1.0, 1.0)


def test_disjoint_masks_score_zero():
    a, b = square(lo=0, hi=3), square(lo=5, hi=8)
    assert iou(a, b) == 0.0
    assert dice(a, b) == 0.0


def test_half_overlap_known_value():
    a = np.zeros((1, 4), bool); a[0, :2] = True
    b = np.zeros((1, 4), bool); b[0, 1:3] = True
    assert iou(a, b) == pytest.approx(1 / 3)
    assert dice(a, b) == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        iou(np.zeros((2, 2)), np.zeros((3, 3)))


def test_chamfer_shifted_line():
    a = np.zeros((10, 10), bool); a[:, 3] = True
    b = np.zeros((10, 10), bool); b[:, 5] = True
    assert chamfer_distance(a, b) == pytest.approx(2.0)
    assert chamfer_distance(a, np.zeros_like(a)) == float("inf")
