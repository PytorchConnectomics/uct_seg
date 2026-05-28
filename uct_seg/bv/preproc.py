"""Blood vessel preprocessing: CLAHE + median blur over re-sliced volumes."""
from __future__ import annotations

import os
import numpy as np

from ..util.imops import histogram_clahe


def preprocess_slice(im, clip_limit: float = 2.0, tile_grid=(8, 8), median_ksize: int = 3):
    """Per-slice CLAHE + median blur. Returns uint8."""
    import cv2
    im = histogram_clahe(im, clip_limit=clip_limit, tile_grid_size=tile_grid)
    if median_ksize > 1:
        im = cv2.medianBlur(im, median_ksize)
    return im


def preprocess_volume(
    src_pattern: str,
    dst_pattern: str,
    z_range: range,
    clip_limit: float = 2.0,
    median_ksize: int = 3,
) -> None:
    """Apply `preprocess_slice` to each z, write to `dst_pattern % z`.

    Ported from T_nag.py opt '0.2' (CLAHE re-slice).
    """
    from imageio import imread, imwrite
    for z in z_range:
        im = imread(src_pattern % z)
        im = preprocess_slice(im, clip_limit=clip_limit, median_ksize=median_ksize)
        imwrite(dst_pattern % z, im)
