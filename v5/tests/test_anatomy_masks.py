"""Test anatomy mask geometry helpers. No model load required."""

from __future__ import annotations

import numpy as np

from v5.data.anatomy_masks import _split_lung_left_right, _split_lung_vertical


def test_vertical_split_contiguous():
    mask = np.zeros((120, 80), dtype=np.uint8)
    mask[30:90, 20:60] = 1  # 60-row lung in middle
    u, m, l = _split_lung_vertical(mask)
    # union should equal original
    assert (u + m + l > 0).sum() == mask.sum()
    # upper rows should be lower-y indices
    if u.sum() > 0 and l.sum() > 0:
        assert np.argwhere(u > 0)[:, 0].mean() < np.argwhere(l > 0)[:, 0].mean()


def test_left_right_split_uses_centroid():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :30] = 1
    mask[:, 70:] = 1
    left, right = _split_lung_left_right(mask)
    # left has higher centroid-x than right (patient-left = observer-right)
    if left.sum() > 0 and right.sum() > 0:
        lx = np.argwhere(left > 0)[:, 1].mean()
        rx = np.argwhere(right > 0)[:, 1].mean()
        assert lx > rx
