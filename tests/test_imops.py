"""Image intensity ops: percentile clip + CLAHE."""
import os
import numpy as np

import _bootstrap  # noqa: F401
from uct_seg.util.io import mkdir
from uct_seg.util.imops import im_adjust

OUT = os.path.join(os.path.dirname(__file__), "output")
mkdir(OUT, "all")


def main():
    rng = np.random.default_rng(0)
    # synthetic float image with outliers
    im = rng.normal(0.5, 0.1, size=(128, 128)).astype(np.float32)
    im[0, 0] = 1e6   # bright outlier
    im[-1, -1] = -1e6

    # 1. percentile clip → uint8
    im8 = im_adjust(im.copy(), thres=(1, 99, True), autoscale="uint8")
    print(f"[clip] dtype={im8.dtype} range=[{im8.min()},{im8.max()}] "
          f"(outliers gone, rescaled to 0..255)")

    # 2. CLAHE (optional cv2 dep)
    try:
        from uct_seg.util.imops import histogram_clahe
        out = histogram_clahe(im8, clip_limit=3.0, tile_grid_size=(8, 8))
        print(f"[clahe] dtype={out.dtype} mean before={im8.mean():.1f} "
              f"after={out.mean():.1f}")
        from imageio import imwrite
        imwrite(os.path.join(OUT, "imadjust.png"), im8)
        imwrite(os.path.join(OUT, "clahe.png"), out)
        print(f"[clahe] saved {OUT}/imadjust.png, {OUT}/clahe.png")
    except ImportError as e:
        print(f"[clahe] skipped (install opencv-python): {e}")


if __name__ == "__main__":
    main()
