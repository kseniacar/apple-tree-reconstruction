"""Ground-truth alignment check: estimate how far each GT mask is shifted from the image.

Students reported on Ed that summer GT masks look offset by 15-45 px from the
visible trunk. This module measures that offset per image by cross-correlating
the GT mask with a detection of the visible wood (the baseline mask), so results
can be reported both with and without an alignment correction.
"""

import numpy as np


def estimate_offset(gt, reference, max_shift=50):
    """Find the integer shift that best aligns a GT mask with a reference mask.

    The cross-correlation of the two masks is computed with FFTs and its peak is
    searched within ``max_shift`` pixels of zero shift.

    Args:
        gt: Boolean (H, W) ground-truth mask.
        reference: Boolean (H, W) mask of visible wood in the image, for
            example ``baseline.predict_mask`` output.
        max_shift: Largest shift in pixels to consider in each direction.

    Returns:
        Tuple ``(dy, dx, overlap)``: ``shift_mask(gt, dy, dx)`` overlaps
        ``reference`` best, with ``overlap`` pixels in common. ``(0, 0, 0)``
        if either mask is empty.

    Raises:
        ValueError: If the masks have different shapes.
    """
    a, b = np.asarray(gt, float), np.asarray(reference, float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if not a.any() or not b.any():
        return 0, 0, 0
    h, w = a.shape
    # Zero-pad to avoid wrap-around, then corr[dy, dx] = sum(gt shifted by (dy, dx) * reference).
    fa = np.fft.rfft2(a, (2 * h, 2 * w))
    fb = np.fft.rfft2(b, (2 * h, 2 * w))
    corr = np.fft.irfft2(fb * np.conj(fa), (2 * h, 2 * w))
    shifts = np.arange(-max_shift, max_shift + 1)
    window = corr[np.ix_(shifts % (2 * h), shifts % (2 * w))]
    i, j = np.unravel_index(np.argmax(window), window.shape)
    return int(shifts[i]), int(shifts[j]), int(round(window[i, j]))


def shift_mask(mask, dy, dx):
    """Shift a mask by whole pixels, filling uncovered pixels with False.

    Args:
        mask: Boolean (H, W) mask.
        dy: Rows to move down (negative moves up).
        dx: Columns to move right (negative moves left).

    Returns:
        Boolean (H, W) shifted mask.
    """
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out
