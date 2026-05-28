"""Nuclei curation: build VAST anchor trees for human review.

Wraps the `uct_seg.vast` writer with the conventional good/bad grouping logic
used by the NucMM mouse review pipeline (originally `T_nag_pf.py` opt 1.25).
"""
from __future__ import annotations

import numpy as np

from ..util.vast import write_vast_anchor_tree_by_id


def split_good_bad_by_shape(seg_ids, sphericity, voxel_count, sphe_band=(5, 95), sz_band=(5, 95)):
    """Bucket seg-ids into (good, bad) based on size + sphericity percentiles."""
    ui = np.asarray(seg_ids)
    sphe = np.asarray(sphericity, dtype=float)
    uc = np.asarray(voxel_count, dtype=float)
    uc[ui == 0] = 0

    bid = []
    thres_sz = np.percentile(uc[uc > 0], list(sz_band))
    bid += list(ui[(uc < thres_sz[0]) | (uc > thres_sz[1])])
    thres_sphe = np.percentile(sphe[sphe > 0], list(sphe_band))
    bid += list(ui[(sphe < thres_sphe[0]) | (sphe > thres_sphe[1])])

    bid = np.unique([b for b in bid if b > 0])
    gid = np.array(sorted(set(ui[ui > 0]) - set(bid)))
    return gid, bid


def write_review_meta(meta_path: str, good_ids, bad_ids, bbs=None, num_max: int = 8000):
    """Write a VAST meta.txt grouping good_ids / bad_ids under named folders."""
    if bbs is None:
        bbs = np.zeros((num_max, 6), int)
    sids = [np.asarray(good_ids), np.asarray(bad_ids)]
    write_vast_anchor_tree_by_id(meta_path, sids, bbs, pref="cb", nn=["good", "bad"])
