"""Neuroglancer LocalVolume layer demo — prints viewer URL.

Needs `pip install -e ".[viz]"`. Keeps the process alive so the URL stays open.
"""
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.viz import launch_viewer, ng_layer3


def main():
    try:
        viewer = launch_viewer(ip="localhost", port=9092)
    except ImportError:
        print("[viz] skipped — install: pip install neuroglancer")
        return

    # synthetic im + seg (zyx order)
    rng = np.random.default_rng(0)
    im = (rng.random((64, 128, 128)) * 255).astype(np.uint8)
    seg = np.zeros((64, 128, 128), np.uint16)
    seg[20:40, 40:80, 40:80] = 1
    seg[10:25, 90:120, 10:40] = 2

    res = [720, 720, 720]
    with viewer.txn() as s:
        s.layers.append(name="im", layer=ng_layer3(im, res, kind="image"))
        s.layers.append(name="seg", layer=ng_layer3(seg, res))

    print("[viz] open this URL in a browser:")
    print(viewer)
    print("[viz] Ctrl-C to exit.")
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
