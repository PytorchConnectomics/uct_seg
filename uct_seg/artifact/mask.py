"""Crack + bright-aggregate mask fusion and seg cleanup."""
from __future__ import annotations

import numpy as np


def fuse_crack_bv(crack_mask, bright_mask):
    """Fuse 2 binary masks into uint8 with label = {1: crack, 2: bright}.

    bright wins where they overlap.
    """
    out = (crack_mask > 0).astype(np.uint8)
    bright = (bright_mask > 0).astype(np.uint8) * 2
    out[bright > 0] = bright[bright > 0]
    return out


def apply_artifact_mask(seg, *masks):
    """Zero out voxels in `seg` covered by any of the provided binary masks."""
    seg = seg.copy()
    for m in masks:
        seg[m > 0] = 0
    return seg
