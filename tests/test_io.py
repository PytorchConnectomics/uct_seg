"""h5 write/read + VAST png ↔ uint32 seg roundtrip + bbox/union."""
import os
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.io import (
    readh5, writeh5, vast2seg, seg2vast, get_bb, get_union, mkdir
)

OUT = os.path.join(os.path.dirname(__file__), "output")
mkdir(OUT, "all")


def main():
    # 1. h5 roundtrip
    seg = (np.random.rand(8, 16, 16) * 1000).astype(np.uint32)
    writeh5(os.path.join(OUT, "demo.h5"), seg, "main")
    seg2 = readh5(os.path.join(OUT, "demo.h5"))
    assert (seg == seg2).all()
    print(f"[h5] wrote+read uint32 {seg.shape} — match")

    # 2. VAST 24-bit png roundtrip
    big_ids = np.array([[0, 1, 256, 65536], [70000, 999999, 16777215, 12345]], np.uint32)
    rgb = seg2vast(big_ids)
    back = vast2seg(rgb)
    assert (back == big_ids).all()
    print(f"[vast] roundtrip OK across 24-bit id range; rgb shape={rgb.shape}")

    # 3. bbox + union
    mask = np.zeros((10, 10, 10), np.uint8)
    mask[2:5, 3:7, 4:6] = 1
    bb = get_bb(mask)
    print(f"[bb] mask bbox = {bb}")
    union = get_union(bb, [0, 9, 0, 9, 0, 9])
    print(f"[bb] union vs full vol = {union}")


if __name__ == "__main__":
    main()
