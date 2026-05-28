"""Dyer-17 microCT dataset prep (V0..V4).

Ported from `T_dyer.py`. Usage:

    python prep_dyer.py <opt>

opt:
  0      nrrd whole volume → per-slice png
  0.1    tiff stack (proj4_masked_390_2014) → per-slice png
  1      per-volume nii/nrrd → h5 (V0..V4)
  1.1    fuse bv+cell semantic labels for V2
  1.2    semantic → instance (cc per class, shifted ids)
  1.3    per-slice bv mask png
  1.4    per-slice raw image png
"""
import sys
import glob
import numpy as np
from imageio import imread, imsave

from uct_seg.util.io import readh5, writeh5
from _paths import Dl, Dv

# Dyer dataset roots (under VCG connectomics share)
DYER_RAW = Dv + "Dyer17/data/"
TRAIN = DYER_RAW + "train/"


def opt_0(opt):
    if opt == "0":
        import nrrd
        fn = DYER_RAW + "original_img_data_0pt65microniso.nrrd"
        data, _ = nrrd.read(fn)
        Do = Dl + "dyer_yuelong/whole/"
        for i in range(data.shape[2]):
            imsave(Do + "%04d.png" % i, data[:, :, i])

    elif opt == "0.1":
        from tifffile import imread as tifread
        D0 = DYER_RAW + "proj4_masked_390_2014/"
        Do = Dl + "dyer_yuelong/whole_v2/"
        for i, p in enumerate(sorted(glob.glob(D0 + "*.tif"))):
            imsave(Do + "%04d.png" % i, tifread(p))


def opt_1(opt):
    if opt == "1":
        import nrrd
        import nibabel
        for i in range(1):
            f1 = TRAIN + "V%d/" % i
            for nn in ("nii", "nrrd"):
                for fn in glob.glob(f1 + "*." + nn):
                    if nn == "nii":
                        data = nibabel.load(fn).get_fdata()
                    else:
                        data, _ = nrrd.read(fn)
                    writeh5(fn[: fn.rfind(".")] + ".h5", data)

    elif opt == "1.1":
        bv = readh5(TRAIN + "V2/V2_anno_dense_bv.h5")
        cb = readh5(TRAIN + "V2/V2_anno_dense_cell.h5")
        # bv=1, cell=2; bv wins per legacy convention
        cb[bv > 0] = 2
        writeh5(TRAIN + "V2/V2_anno_dense_bv_cell.h5", cb)

    elif opt == "1.2":
        from skimage.measure import label
        for i in range(5):
            seg = readh5(TRAIN + "V%d/V%d_anno_dense_bv_cell.h5" % (i, i))
            bv = label(seg == 1)
            cb = label(seg == 2)
            bm = bv.max()
            bv[cb > 0] = cb[cb > 0] + bm
            writeh5(TRAIN + "V%d/V%d_anno_dense_bv_cell.h5" % (i, i), bv)

    elif opt == "1.3":
        for i in range(1, 3):
            bv = readh5(TRAIN + "V%d/V%d_anno_dense_bv_cell_corrected.h5" % (i, i))
            fn = Dl + "dyer_yuelong/V%d/label/" % i + "%04d.png"
            for z in range(bv.shape[0]):
                imsave(fn % z, (bv[z] == 1).astype(np.uint8))

    elif opt == "1.4":
        for i in range(1, 3):
            im = readh5(TRAIN + "V%d/V%d_img.h5" % (i, i))
            fn = Dl + "dyer_yuelong/V%d/img/" % i + "%04d.png"
            for z in range(im.shape[0]):
                imsave(fn % z, im[z])


if __name__ == "__main__":
    opt = sys.argv[1]
    if opt[0] == "0":
        opt_0(opt)
    elif opt[0] == "1":
        opt_1(opt)
    else:
        raise SystemExit("unknown opt: %r" % opt)
