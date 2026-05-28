"""VAST Lite annotation interop: meta parse, good/bad relabel, anchor tree writer.

VAST `meta.txt` columns (24 ints + name string):
    Nr  flags  red1 green1 blue1 pattern1  red2 green2 blue2 pattern2
    anchorx anchory anchorz  parentnr childnr prevnr nextnr  collapsednr
    bboxx1 bboxy1 bboxz1 bboxx2 bboxy2 bboxz2   "name"
"""
from __future__ import annotations

import numpy as np


# ---------- color palette ----------

def get_spaced_colors(n: int):
    max_value = 16581375  # 255**3
    interval = max(1, max_value // max(1, n))
    colors = [hex(i)[2:].zfill(6) for i in range(0, max_value, interval)]
    return [(int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)) for c in colors]


# ---------- meta parser ----------

def read_vast_meta(fn: str):
    """Return (dd int[N,24], names list[N]). Includes background seg row."""
    a = open(fn).readlines()
    st_id = 0
    while a[st_id][0] in ("%", "\\", "\n"):
        st_id += 1
    st_id -= 1
    out = np.zeros((len(a) - st_id - 1, 24), dtype=int)
    name = [None] * (len(a) - st_id - 1)
    for i in range(st_id + 1, len(a)):
        out[i - st_id - 1] = np.array(
            [int(x) for x in a[i][: a[i].find('"')].split(" ") if len(x) > 0]
        )
        name[i - st_id - 1] = a[i][a[i].find('"') + 1: a[i].rfind('"')]
    return out, name


readVastSeg = read_vast_meta


# ---------- good/bad relabel ----------

def vast_meta_relabel(
    fn: str,
    kw_bad=("bad", "del"),
    kw_nm=("good",),
    do_print: bool = False,
    return_ids: bool = False,
):
    """Build a lookup `rl` so that `seg_relabeled = rl[seg]`.

    - kw_bad-named segs (and their VAST-tree children) get mapped to 0.
    - kw_nm-named segs are kept as-is; their children collapse upward.
    - Unnamed branches collapse to their first ancestor in `kw_nm` ∪ kw_bad.

    If `return_ids`, instead returns (good_ids, bad_ids) tuple over the
    non-empty seg rows (skipping background).
    """
    dd, nn = read_vast_meta(fn)
    rl = np.arange(dd[:, 0].max() + 1, dtype=np.uint16)
    pid = np.unique(dd[:, 13])

    if do_print:
        print(
            ",".join(
                nn[x] for x in np.where(dd[:, 13] == 0)[0] if "Imported Segment" not in nn[x]
            )
        )

    pid_b = []
    bid = np.array([], dtype=int)
    if len(kw_bad) > 0:
        pid_b = [
            i
            for i, x in enumerate(nn)
            if max(y in x.lower() for y in kw_bad)
        ]
        bid = np.where(np.in1d(dd[:, 13], pid_b))[0]
        bid = np.hstack([pid_b, bid]).astype(int)
        if len(bid) > 0:
            rl[bid] = 0
        print("found %d bad" % len(bid))

    if return_ids:
        neid = np.where(dd[:, -1] > -1)[0]
        bid = bid[np.in1d(bid, neid)]
        gid = neid[np.in1d(neid, bid, invert=True)]
        return dd[gid, 0], dd[bid, 0]

    kw_nm = list(kw_nm) + ["background"]
    pid_nm = [
        i
        for i, x in enumerate(nn)
        if max(y.lower() in x.lower() for y in kw_nm)
    ]
    pid_nm = np.hstack([pid_nm, pid_b])
    for p in pid[np.in1d(pid, pid_nm, invert=True)]:
        rl[dd[dd[:, 13] == p, 0]] = p

    # consolidate root labels
    for u in np.unique(rl[np.where(rl[rl] != rl)[0]]):
        u0 = u
        while u0 != rl[u0]:
            rl[rl == u0] = rl[u0]
            u0 = rl[u0]
    return rl


vastMetaRelabel = vast_meta_relabel


# ---------- anchor tree writer ----------

def write_vast_anchor_tree_by_id(
    fn: str,
    sids,
    bbs,
    nn=("good", "bad"),
    pref: str = "seg",
    id_rl=None,
):
    """Write a VAST `meta.txt` grouping seg-ids under named parent folders.

    sids: list of arrays — one per group (e.g. [good_ids, bad_ids]).
    bbs:  Nx6 array of [x0,y0,z0,x1,y1,z1] per seg id (indexed by id).
    nn:   group names; len == len(sids).
    pref: per-seg name prefix → "<pref>%d".
    id_rl: optional remap to express parent/child relations between ids.
    """
    sid_m = max(s.max() for s in sids) + 1

    with open(fn, "w") as oo:
        bg = (
            '0   0   0 0 0 0   0 0 0 0   -1 -1 -1  0 0 0 %d   0   '
            '-1 -1 -1 -1 -1 -1   "Background"\n' % sid_m
        )
        oo.write(bg)

        ccs = np.array(get_spaced_colors(bbs.shape[0]))
        ccs = ccs[np.random.permutation(ccs.shape[0])]
        seg_fmt = (
            '%d 1 %d %d %d 0 255 0 0 0 %d %d %d %d 0 %d %d %d %d '
            '%d %d %d %d %d "%s%d"\n'
        )
        cid = [None] * len(sids)
        cc_id = 0
        out = [""] * bbs.shape[0]
        for ii, sid in enumerate(sids):
            numS = len(sid) if isinstance(sid, list) else sid.size
            if numS == 1:
                sid = [int(sid)]
            for i in range(numS):
                bb = bbs[sid[i]]
                prevn = sid[i - 1] if i != 0 else 0
                nextn = sid[i + 1] if i != numS - 1 else 0
                parent = sid_m + ii
                if id_rl is not None:
                    i0 = i - 1
                    while id_rl[prevn] != prevn:
                        prevn = sid[i0 - 1] if i0 != 0 else 0
                        i0 -= 1
                    i0 = i + 1
                    while id_rl[nextn] != nextn:
                        nextn = sid[i0 + 1] if i0 != numS - 1 else 0
                        i0 += 1
                out[sid[i] - 1] = seg_fmt % (
                    sid[i],
                    ccs[cc_id, 0], ccs[cc_id, 1], ccs[cc_id, 2],
                    (bb[0] + bb[3]) // 2, (bb[1] + bb[4]) // 2, (bb[2] + bb[5]) // 2,
                    parent, prevn, nextn, parent,
                    bb[0], bb[1], bb[2], bb[3], bb[4], bb[5],
                    pref, sid[i],
                )
                if i == 0:
                    cid[ii] = sid[0]
                cc_id += 1

        if id_rl is not None:
            ui, uc = np.unique(id_rl, return_counts=True)
            for ii in ui[uc > 1]:
                jjs = np.where(id_rl == ii)[0]
                jjs = jjs[jjs != ii]
                tmp = out[ii - 1].split(" ")
                tmp[14] = str(jjs[0])
                out[ii - 1] = " ".join(tmp)
                for jid, jj in enumerate(jjs):
                    tmp = out[jj - 1].split(" ")
                    tmp[13] = str(ii)
                    tmp[15] = "0" if jid == 0 else str(jjs[jid - 1])
                    tmp[16] = "0" if jid == len(jjs) - 1 else str(jjs[jid + 1])
                    out[jj - 1] = " ".join(tmp)

        for i in range(bbs.shape[0]):
            oo.write(out[i])

        ccs = get_spaced_colors(len(nn))
        for nid, n in enumerate(nn):
            prevn = sid_m + nid - 1 if nid != 0 else 0
            nextn = sid_m + nid + 1 if nid != len(nn) - 1 else 0
            footer = (
                '%d   1   %d %d %d %d   %d %d %d %d   -1 -1 -1  0 %d %d %d   '
                '%d   -1 -1 -1 -1 -1 -1   "%s"\n'
                % (
                    sid_m + nid,
                    ccs[nid][0], ccs[nid][1], ccs[nid][2], sid_m + nid,
                    ccs[nid][0], ccs[nid][1], ccs[nid][2], sid_m + nid,
                    cid[nid], prevn, nextn, sid_m + nid, n,
                )
            )
            oo.write(footer)


writeVastAnchorTreeById = write_vast_anchor_tree_by_id
