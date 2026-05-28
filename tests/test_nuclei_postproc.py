"""Nuclei post-proc: cc3d → dust removal → good/bad shape split."""
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.seg_ops import get_bb_label3d_v2
from uct_seg.nuclei.postproc import cc3d_label, filter_dust
from uct_seg.nuclei.curate import split_good_bad_by_shape


def synth_volume():
    """Build a (32, 64, 64) binary volume with a few spheres + dust."""
    vol = np.zeros((32, 64, 64), np.uint8)
    centers = [(8, 16, 16, 6), (8, 16, 48, 5), (24, 48, 16, 7), (24, 48, 48, 3)]
    zz, yy, xx = np.indices(vol.shape)
    for cz, cy, cx, r in centers:
        vol[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1
    rng = np.random.default_rng(0)
    # dust noise
    for _ in range(20):
        z, y, x = rng.integers(0, [32, 64, 64], 3)
        vol[z, y, x] = 1
    return vol


def main():
    vol = synth_volume()
    print(f"[input] binary voxels = {int(vol.sum())}")

    try:
        labels = cc3d_label(vol, connectivity=26)
    except ImportError:
        print("[cc3d] skipped — install: pip install connected-components-3d")
        return

    n0 = int(labels.max())
    print(f"[cc3d] components = {n0}")

    cleaned = filter_dust(labels, dust_sz=4)
    n1 = int(np.unique(cleaned).size - 1)
    print(f"[dust>4] survivors = {n1}")

    # fake sphericity / size stats (1 row per surviving id)
    ids = np.unique(cleaned)
    ids = ids[ids > 0]
    bbs = get_bb_label3d_v2(cleaned, uid=ids, do_count=True)
    vox = bbs[:, -1]
    # fake sphericity = 1 - relative bbox aspect deviation
    spans = np.stack([bbs[:, 2] - bbs[:, 1], bbs[:, 4] - bbs[:, 3], bbs[:, 6] - bbs[:, 5]], 1)
    sphe = 1.0 - spans.std(1) / (spans.mean(1) + 1e-6)

    good, bad = split_good_bad_by_shape(ids, sphe, vox, sphe_band=(5, 95), sz_band=(5, 95))
    print(f"[split] good={good.tolist()} bad={bad.tolist()}")


if __name__ == "__main__":
    main()
