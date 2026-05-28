"""Filesystem + VAST conversion + bbox helpers.

Ported from `T_util.py` (Donglai's lab code). Kept thin: numpy + h5py only at
import time. Heavy deps (cv2, neuroglancer) live in dedicated modules.
"""
from __future__ import annotations

import os
import numpy as np


# ---------- mkdir ----------

def mkdir(fn: str, opt: str = "") -> None:
    """Create dir. opt: '' (mkdir), 'all' (makedirs), 'parent' (mkdir on dirname)."""
    if opt == "parent":
        fn = fn[: fn.rfind("/")]
    if not os.path.exists(fn):
        if opt == "all":
            os.makedirs(fn)
        else:
            os.mkdir(fn)


# ---------- text ----------

def readtxt(filename: str) -> list[str]:
    with open(filename) as f:
        return f.readlines()


def writetxt(filename: str, content) -> None:
    with open(filename, "w") as f:
        if isinstance(content, list):
            for ll in content:
                f.write(ll)
                if "\n" not in ll:
                    f.write("\n")
        else:
            f.write(content)


# ---------- h5 ----------

def readh5(filename: str, datasetname=None):
    import h5py
    fid = h5py.File(filename, "r")
    if datasetname is None:
        datasetname = list(fid)
    if len(datasetname) == 1:
        datasetname = datasetname[0]
    if isinstance(datasetname, list):
        return [np.array(fid[d]) for d in datasetname]
    return np.array(fid[datasetname])


def writeh5(filename: str, dtarray, datasetname="main") -> None:
    import h5py
    with h5py.File(filename, "w") as fid:
        if isinstance(datasetname, list):
            for i, dd in enumerate(datasetname):
                ds = fid.create_dataset(
                    dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype
                )
                ds[:] = dtarray[i]
        else:
            ds = fid.create_dataset(
                datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype
            )
            ds[:] = dtarray


def readh5_b(filename: str, sz, datasetname="main"):
    """Read packed-bit binary h5."""
    import h5py
    tmp = np.unpackbits(np.array(h5py.File(filename, "r")[datasetname]))
    return tmp[: int(np.prod(sz))].reshape(sz)


def writeh5_b(filename: str, dtarray, datasetname="main") -> None:
    """Write packed-bit binary h5."""
    import h5py
    with h5py.File(filename, "w") as fid:
        if isinstance(datasetname, list):
            for i, dd in enumerate(datasetname):
                tmp = np.packbits(dtarray[i].reshape(-1))
                ds = fid.create_dataset(dd, tmp.shape, compression="gzip", dtype=np.uint8)
                ds[:] = tmp
        else:
            tmp = np.packbits(dtarray.reshape(-1))
            ds = fid.create_dataset(datasetname, tmp.shape, compression="gzip", dtype=np.uint8)
            ds[:] = tmp


# ---------- VAST RGB ↔ uint32 ----------

def seg2vast(seg):
    """uint32 seg -> HxWx3 uint8 (24-bit VAST png)."""
    return np.stack([seg // 65536, seg // 256, seg % 256], axis=2).astype(np.uint8)


def vast2seg(seg):
    """HxWx3 uint8 VAST png -> uint32 seg. 2D grayscale passes through."""
    if seg.ndim == 2:
        return seg
    return (
        seg[:, :, 0].astype(np.uint32) * 65536
        + seg[:, :, 1].astype(np.uint32) * 256
        + seg[:, :, 2].astype(np.uint32)
    )


# Legacy CamelCase aliases
seg2Vast = seg2vast
vast2Seg = vast2seg


# ---------- bbox ----------

def get_bb(seg, do_count: bool = False):
    """Tight bbox over seg>0. Returns [z0,z1,y0,y1,...] (+ count optionally)."""
    dim = len(seg.shape)
    a = np.where(seg > 0)
    if len(a[0]) == 0:
        return [-1] * dim * 2
    out = []
    for i in range(dim):
        out += [int(a[i].min()), int(a[i].max())]
    if do_count:
        out += [len(a[0])]
    return out


def get_union(a, b):
    """Per-axis union of two bbox vectors [min,max,min,max,...]."""
    ll = len(a)
    out = [None] * ll
    for i in range(0, ll, 2):
        out[i] = min(a[i], b[i])
    for i in range(1, ll, 2):
        out[i] = max(a[i], b[i])
    return out


def get_intersect(a, b):
    ll = len(a)
    out = [None] * ll
    for i in range(0, ll, 2):
        out[i] = max(a[i], b[i])
    for i in range(1, ll, 2):
        out[i] = min(a[i], b[i])
    return out


# ---------- misc ----------

def arr2str(arr, delim=",") -> str:
    return delim.join(str(x) for x in arr)


def print_arr(arr, num: int = 10) -> None:
    n_row = (len(arr) + num - 1) // num
    for r in range(n_row):
        print(arr2str(arr[r * num : (r + 1) * num]))


arrToStr = arr2str
printArr = print_arr
U_mkdir = mkdir
