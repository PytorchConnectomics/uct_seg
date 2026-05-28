"""Butterfly v2 manifest JSON writer."""
import json
import os

import _bootstrap  # noqa: F401
from uct_seg.util.io import mkdir
from uct_seg.util.butterfly import write_bfly_v2

OUT = os.path.join(os.path.dirname(__file__), "output")
mkdir(OUT, "all")


def main():
    sz = (700, 996, 968)         # depth, height, width
    numT = (1, 1)                 # 1 row x 1 column tile grid
    fn_pattern = "im_720nm/%04d.png"
    out_path = os.path.join(OUT, "im_720.json")
    write_bfly_v2(
        sz=sz, numT=numT,
        imN=lambda z: fn_pattern % z,
        tsz=4096,
        outName=out_path,
    )
    spec = json.load(open(out_path))
    print(f"[bfly] wrote {out_path}")
    print(f"[bfly] sections={len(spec['image'])} "
          f"depth={spec['depth']} h={spec['height']} w={spec['width']} "
          f"tile_sz={spec['tile_size']}")
    print(f"[bfly] first 3 paths: {spec['image'][:3]}")


if __name__ == "__main__":
    main()
