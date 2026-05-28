"""Nuclei post-processing: connected components, dust removal, shape stats.

Optional deps:
  - `cc3d` (connected-components-3d) for 26-connectivity 3D labelling
  - shape stats (`getSphericity`) live in a separate `imu` package upstream;
    we expose only the parts we actually use here. For the full IoU/sphericity
    curation flow see `uct_seg.nuclei.curate.score_predictions`.
"""
from __future__ import annotations

import numpy as np

from ..util.seg_ops import remove_seg, get_seg_dtype


def cc3d_label(seg, connectivity: int = 26):
    """Wrap cc3d.connected_components into an uint16/32 array."""
    import cc3d
    out = cc3d.connected_components(seg.astype(np.uint32), connectivity=connectivity)
    return out.astype(get_seg_dtype(int(out.max())))


def filter_dust(seg, dust_sz: int = 25):
    """Remove components with voxel count <= dust_sz. Returns relabeled seg."""
    ui, uc = np.unique(seg, return_counts=True)
    uc[ui == 0] = 0
    bid = ui[uc <= dust_sz]
    if len(bid) > 0:
        seg = remove_seg(seg, bid)
    return seg


def percentile_outlier_ids(values, ids, lo: float = 5, hi: float = 95):
    """Return ids whose `values` fall outside the [lo, hi] percentile band."""
    values = np.asarray(values)
    ids = np.asarray(ids)
    thres = np.percentile(values[values > 0], [lo, hi])
    bad = (values < thres[0]) | (values > thres[1])
    return ids[bad]
