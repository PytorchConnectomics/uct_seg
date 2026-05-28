"""Segmentation ops: relabel, remove_seg, per-id bbox, IoU."""
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.seg_ops import (
    relabel, remove_seg, remove_small, get_bb_label3d_v2, seg_iou3d
)


def main():
    # synthetic 2-object volume
    seg = np.zeros((10, 20, 20), np.uint16)
    seg[1:5, 2:8, 2:8] = 42       # object 42
    seg[6:9, 10:15, 10:15] = 100  # object 100
    seg[0:2, 18:20, 18:20] = 7    # tiny dust object 7

    print("[orig] ids:", np.unique(seg).tolist())

    # 1. dust removal (< 16 voxels)
    cleaned = remove_small(seg.copy(), thres=16)
    print("[dust>16] surviving ids:", np.unique(cleaned).tolist())

    # 2. compact relabel
    rl = relabel(cleaned)
    print("[relabel] new ids:", np.unique(rl).tolist())

    # 3. per-id bbox
    bbs = get_bb_label3d_v2(seg)
    for row in bbs:
        sid, z0, z1, y0, y1, x0, x1 = row
        print(f"[bbox] id={sid}: z=[{z0},{z1}] y=[{y0},{y1}] x=[{x0},{x1}]")

    # 4. IoU between original and a "predicted" version shifted by 1 voxel
    pred = np.roll(seg, 1, axis=1)
    iou = seg_iou3d(seg, pred)
    print("[iou] cols = [seg1_id, seg2_best_id, seg1_count, seg2_count, overlap]")
    for row in iou:
        print(f"       {row.tolist()}")

    # 5. remove_seg by id list
    no42 = remove_seg(seg, [42])
    assert 42 not in np.unique(no42)
    print("[remove] dropped id 42 — surviving:", np.unique(no42).tolist())


if __name__ == "__main__":
    main()
