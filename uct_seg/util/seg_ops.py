"""Segmentation array ops: dtype packing, relabel, bbox-per-id, IoU.

Ported from `T_util.py`. NumPy + (optional) scikit-image.
"""
from __future__ import annotations

import numpy as np


def get_seg_dtype(mid: int):
    """Smallest unsigned int dtype that fits `mid`."""
    if mid < 2 ** 8:
        return np.uint8
    if mid < 2 ** 16:
        return np.uint16
    if mid < 2 ** 32:
        return np.uint32
    return np.uint64


def cast_to_dtype(seg):
    return seg.astype(get_seg_dtype(int(seg.max())))


def remove_seg(seg, did, invert: bool = False):
    """Zero out (or keep, if invert) the listed seg-ids via a single LUT."""
    did = np.asarray(did)
    sm = int(seg.max())
    did = did[did <= sm]
    if invert:
        rl = np.zeros(sm + 1, dtype=seg.dtype)
        rl[did] = did
    else:
        rl = np.arange(sm + 1, dtype=seg.dtype)
        rl[did] = 0
    return rl[seg]


def remove_small(seg, thres: int = 100, bid=None, invert: bool = False):
    if thres <= 0:
        return seg
    if bid is None:
        uid, uc = np.unique(seg, return_counts=True)
        bid = uid[uc < thres]
    if len(bid) > 0:
        seg = remove_seg(seg, bid, invert)
    return seg


def relabel(seg, uid=None, do_sort: bool = False):
    """Compact seg-ids to 1..N. Background 0 stays 0."""
    if seg is None or seg.max() == 0:
        return seg
    if do_sort:
        uid, _ = seg_to_count(seg, do_sort=True, rm_zero=True)
    elif uid is None:
        uid = np.unique(seg)
    else:
        uid = np.asarray(uid)
    uid = uid[uid > 0]
    sm = int(seg.max())
    rl = np.zeros(sm + 1, dtype=get_seg_dtype(len(uid)))
    rl[uid] = np.arange(1, 1 + len(uid))
    return rl[seg]


def seg_to_count(seg, do_sort: bool = True, rm_zero: bool = False):
    sm = int(seg.max())
    if sm == 0:
        return None, None
    ui, uc = np.unique(seg, return_counts=True)
    if rm_zero:
        uc = uc[ui > 0]
        ui = ui[ui > 0]
    if do_sort:
        order = np.argsort(-uc)
        ui = ui[order]
        uc = uc[order]
    return ui, uc


# ---------- per-id bbox ----------

def get_bb_label2d_v2(seg, do_count: bool = False, uid=None):
    """Per-id bbox in 2D. Returns rows [id, r0, r1, c0, c1, (count)]."""
    sz = seg.shape
    assert len(sz) == 2
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    if len(uid) == 0:
        return np.zeros((1, 5 + int(do_count)), dtype=np.uint32)
    um = int(uid.max())
    out = np.zeros((1 + um, 5 + int(do_count)), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    rids = np.where((seg > 0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[rid])
        sid = sid[(sid > 0) * (sid <= um)]
        out[sid, 1] = np.minimum(out[sid, 1], rid)
        out[sid, 2] = np.maximum(out[sid, 2], rid)
    cids = np.where((seg > 0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, cid])
        sid = sid[(sid > 0) * (sid <= um)]
        out[sid, 3] = np.minimum(out[sid, 3], cid)
        out[sid, 4] = np.maximum(out[sid, 4], cid)
    if do_count:
        ui, uc = np.unique(seg, return_counts=True)
        out[ui, -1] = uc
    return out[uid]


def get_bb_label3d_v2(seg, do_count: bool = False, uid=None):
    """Per-id bbox in 3D. Returns rows [id, z0, z1, y0, y1, x0, x1, (count)]."""
    sz = seg.shape
    assert len(sz) == 3
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid > 0]
    um = int(uid.max())
    out = np.zeros((1 + um, 7 + int(do_count)), dtype=np.uint32)
    out[:, 0] = np.arange(out.shape[0])
    out[:, 1] = sz[0]
    out[:, 3] = sz[1]
    out[:, 5] = sz[2]
    zids = np.where((seg > 0).sum(axis=1).sum(axis=1) > 0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid > 0) * (sid <= um)]
        out[sid, 1] = np.minimum(out[sid, 1], zid)
        out[sid, 2] = np.maximum(out[sid, 2], zid)
    rids = np.where((seg > 0).sum(axis=0).sum(axis=1) > 0)[0]
    for rid in rids:
        sid = np.unique(seg[:, rid])
        sid = sid[(sid > 0) * (sid <= um)]
        out[sid, 3] = np.minimum(out[sid, 3], rid)
        out[sid, 4] = np.maximum(out[sid, 4], rid)
    cids = np.where((seg > 0).sum(axis=0).sum(axis=0) > 0)[0]
    for cid in cids:
        sid = np.unique(seg[:, :, cid])
        sid = sid[(sid > 0) * (sid <= um)]
        out[sid, 5] = np.minimum(out[sid, 5], cid)
        out[sid, 6] = np.maximum(out[sid, 6], cid)
    if do_count:
        ui, uc = np.unique(seg, return_counts=True)
        out[ui[ui <= um], -1] = uc[ui <= um]
    return out[uid]


# ---------- IoU ----------

def seg_iou3d(seg1, seg2, ui0=None):
    """For each id in seg1, find the best-overlap id in seg2.

    Returns Nx5: [seg1_id, seg2_best_id, seg1_count, seg2_best_count, overlap].
    """
    ui, uc = np.unique(seg1, return_counts=True)
    uc = uc[ui > 0]
    ui = ui[ui > 0]
    ui2, uc2 = np.unique(seg2, return_counts=True)
    if ui0 is None:
        ui0 = ui
    out = np.zeros((len(ui0), 5), int)
    bbs = get_bb_label3d_v2(seg1, uid=ui0)[:, 1:]
    out[:, 0] = ui0
    out[:, 2] = uc[np.in1d(ui, ui0)]
    for j, i in enumerate(ui0):
        bb = bbs[j]
        ui3, uc3 = np.unique(
            seg2[bb[0]: bb[1] + 1, bb[2]: bb[3] + 1, bb[4]: bb[5] + 1]
            * (seg1[bb[0]: bb[1] + 1, bb[2]: bb[3] + 1, bb[4]: bb[5] + 1] == i),
            return_counts=True,
        )
        uc3[ui3 == 0] = 0
        out[j, 1] = ui3[np.argmax(uc3)]
        out[j, 3] = uc2[ui2 == out[j, 1]]
        out[j, 4] = uc3.max()
    return out


# Legacy CamelCase aliases used by ported scripts
removeSeg = remove_seg
removeSmall = remove_small
getSegDtype = get_seg_dtype
cast2dtype = cast_to_dtype
seg2Count = seg_to_count
