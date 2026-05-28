"""Neuroglancer viewer for ported CT volumes.

Ported from `ct/ng.py`. Usage:

    python ng_view.py <opt>

opt:
  0     vol1 image + seg (720nm res)
  0.1   pyramidal cell gt vol0 / vol1
  0.2   vol1 im + multiple seg variants
  0.3   zergham annotation crop (500^3)

Requires neuroglancer. Prints viewer URL on stdout.
"""
import sys
import numpy as np

from uct_seg.util.io import readh5
from uct_seg.util.viz import launch_viewer, ng_layer, ng_layer3
from _paths import Dv

Dc = Dv + "CT/Zeiss_Nag/"


def main():
    opt = sys.argv[1]
    viewer = launch_viewer(ip="localhost", port=9092)
    res = np.array([720, 720, 720])

    if opt == "0":
        im = readh5(Dc + "vol1/vol1_im-yz.h5")
        seg = readh5(Dc + "vol1/vol1_seg_v1.h5")
        with viewer.txn() as s:
            s.layers.append(name="im", layer=ng_layer(im, res, kind="image"))
            s.layers.append(name="seg", layer=ng_layer(seg, res))

    elif opt == "0.1":
        res = [700, 700, 700]
        s1 = readh5(Dc + "vol_pyr/gt_vol0.h5")
        s2 = readh5(Dc + "vol_pyr/gt_vol1.h5")
        with viewer.txn() as s:
            s.layers.append(name="s1", layer=ng_layer(s1, res))
            s.layers.append(name="s2", layer=ng_layer(s2, res))

    elif opt == "0.2":
        s1 = readh5(Dc + "vol1/vol1_im.h5")
        s2 = readh5(Dc + "vol1/vol1_seg.h5")
        s3 = readh5(Dc + "vol1/vol1_seg_v1.h5")
        s4 = readh5(Dc + "vol1/vol1_seg_pf_v1.h5")
        with viewer.txn() as s:
            s.layers.append(name="im", layer=ng_layer(s1, res, kind="image"))
            s.layers.append(name="s2", layer=ng_layer(s2[0], res * 2))
            s.layers.append(name="s3", layer=ng_layer(s3, res))
            s.layers.append(name="s4", layer=ng_layer(s4, res))

    elif opt == "0.3":
        im = readh5(Dc + "train_cb/im_500.h5")
        cb = readh5(Dc + "train_cb/seg_cb_500.h5")
        with viewer.txn() as s:
            s.layers.append(name="im", layer=ng_layer3(im, res, kind="image"))
            s.layers.append(name="seg", layer=ng_layer3(cb, res))

    else:
        raise SystemExit("unknown opt: %r" % opt)

    print(viewer)


if __name__ == "__main__":
    main()
