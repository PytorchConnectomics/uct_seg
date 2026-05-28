"""Image intensity ops: percentile rescale, CLAHE."""
from __future__ import annotations

import numpy as np


def im_adjust(I, thres=(1, 99, True), autoscale: str | None = None):
    """Clamp + optional rescale.

    thres = (low, high, is_percentile). autoscale: None | 'uint8' (0..255).
    """
    if thres[2]:
        I_low, I_high = np.percentile(I.reshape(-1), thres[:2])
    else:
        I_low, I_high = thres[0], thres[1]
    I = I.copy()
    I[I > I_high] = I_high
    I[I < I_low] = I_low
    if autoscale is not None:
        I = (I.astype(float) - I_low) / (I_high - I_low)
        if autoscale == "uint8":
            I = (I * 255).astype(np.uint8)
    return I


def histogram_clahe(im, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization (cv2)."""
    import cv2
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(im)


imAdjust = im_adjust
HistogramCLAHE = histogram_clahe
