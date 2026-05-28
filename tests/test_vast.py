"""VAST meta: write tiny meta.txt → parse → relabel good/bad → anchor tree."""
import os
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.io import mkdir
from uct_seg.util.vast import (
    read_vast_meta, vast_meta_relabel, write_vast_anchor_tree_by_id,
)

OUT = os.path.join(os.path.dirname(__file__), "output")
mkdir(OUT, "all")

# Minimal hand-written meta.txt mimicking VAST format.
# Columns (24 ints): Nr flags r1 g1 b1 p1 r2 g2 b2 p2 ax ay az
#                    parent child prev next collapsed bx0 by0 bz0 bx1 by1 bz1
# Tree:
#   1 = "good" folder
#   2 = "bad"  folder
#   3,4,5 belong to good ; 6,7 belong to bad
META = """\
0 0 0 0 0 0 0 0 0 0 -1 -1 -1 0 0 0 8 0 -1 -1 -1 -1 -1 -1 "Background"
1 1 0 255 0 1 0 255 0 1 -1 -1 -1 0 3 0 0 1 -1 -1 -1 -1 -1 -1 "good"
2 1 255 0 0 1 255 0 0 1 -1 -1 -1 0 6 0 0 2 -1 -1 -1 -1 -1 -1 "bad"
3 1 10 10 10 0 10 10 10 0 5 5 5 1 0 0 4 3 0 0 0 9 9 9 "cb3"
4 1 20 20 20 0 20 20 20 0 5 5 5 1 0 3 5 4 0 0 0 9 9 9 "cb4"
5 1 30 30 30 0 30 30 30 0 5 5 5 1 0 4 0 5 0 0 0 9 9 9 "cb5"
6 1 40 40 40 0 40 40 40 0 5 5 5 2 0 0 7 6 0 0 0 9 9 9 "cb6"
7 1 50 50 50 0 50 50 50 0 5 5 5 2 0 6 0 7 0 0 0 9 9 9 "cb7"
"""


def main():
    meta_in = os.path.join(OUT, "meta_in.txt")
    with open(meta_in, "w") as f:
        f.write(META)

    # 1. parse
    dd, names = read_vast_meta(meta_in)
    print(f"[parse] {len(names)} entries: {names}")

    # 2. build lookup that maps bad-tree (ids 2, 6, 7) → 0, good (ids 1, 3, 4, 5) kept
    rl = vast_meta_relabel(meta_in, kw_bad=["bad"], kw_nm=["good"], do_print=True)
    print(f"[relabel] lookup = {rl.tolist()}")
    seg_in = np.array([0, 3, 4, 5, 6, 7], np.uint16)
    print(f"[relabel] seg {seg_in.tolist()} -> {rl[seg_in].tolist()}")

    # 3. write anchor tree grouping ids into good/bad folders
    good_ids = np.array([3, 4, 5])
    bad_ids = np.array([6, 7])
    bbs = np.zeros((10, 6), int)
    for i in (3, 4, 5, 6, 7):
        bbs[i] = [i, i, i, i + 9, i + 9, i + 9]
    out_meta = os.path.join(OUT, "meta_out.txt")
    write_vast_anchor_tree_by_id(
        out_meta, [good_ids, bad_ids], bbs, nn=["good", "bad"], pref="cb"
    )
    print(f"[anchor] wrote {out_meta}")
    print("------- contents -------")
    print(open(out_meta).read())


if __name__ == "__main__":
    main()
