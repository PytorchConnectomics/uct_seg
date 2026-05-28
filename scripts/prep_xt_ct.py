"""Xiaotang synchrotron CT prep + neuroglancer precomputed tile pyramid.

Ported from `eng/T_xt_ct.py`. Usage:

    python prep_xt_ct.py <opt> [job_id job_num]

opt:
  0      x16-downsample whole volume to h5 (low-res preview)
  0.1    quantize raw float32 tiff → 8-bit png (per-slice, parallelizable)
  1.0    neuroglancer precomputed image pyramid
  1.1    neuroglancer precomputed seg
  1.2    neuroglancer mesh (igneous env)
"""
import os
import sys
import numpy as np
from imageio import imread, imwrite

from uct_seg.util.io import writeh5, mkdir
from uct_seg.util.imops import im_adjust
from _paths import Dl2, Dw

D0 = Dl2 + "../Lab/Xiaotang_synchrotron/"
D1 = Dl2 + "CT/xt_202305/"
DB = "db/xt_ct/"
RES = np.array([1380, 1380, 1380])

# default dataset
NN = "mouse_5mo_rec"
SZ = [11088, 6464, 6464]
NN2 = NN + "_80_995/"

Di = D0 + NN + "/recon_%05d.tiff"   # raw float tiff
Di2 = D1 + NN2 + "%05d.png"         # 8-bit png


def parse_argv():
    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id = int(sys.argv[2])
        job_num = int(sys.argv[3])
    return opt, job_id, job_num


def opt_0(opt, job_id, job_num):
    if opt == "0":
        ratio = 16
        mkdir(DB, "all")
        out = np.zeros((np.array(SZ) // ratio).astype(int))
        for z in range(out.shape[0]):
            out[z] = imread(Di % (z * ratio))[::ratio, ::ratio]
        writeh5(DB + "im_x16.h5", out)

    elif opt == "0.1":
        # Pre-computed percentile clip from a previous pass over the whole volume.
        # If running on a different dataset, recompute via im_adjust(..., thres=[1,99,True])
        ran = [8.84428027e-06, 6.62667452e-04, False]
        for z in range(job_id, SZ[0], job_num):
            im = im_adjust(imread(Di % z), ran, "uint8")
            imwrite(Di2 % z, im)


def opt_1(opt, job_id, job_num):
    from imu.ng import NgDataset
    from imu.io import read_tile_volume

    Do0 = "file://" + Dw + "ng/xt_ct/"
    Do = Do0 + NN + "/"
    mkdir(Do[7:], "all")

    chunk = [64, 64, 64]
    mip_im = [
        [1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8],
        [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128],
    ]
    nt = 1

    output_im = Do + "im"
    output_seg = Do + "seg"
    mkdir(output_im[7:], "all")
    mkdir(output_seg[7:], "all")

    if opt == "1.0":
        dst = NgDataset(volume_size=SZ[::-1], resolution=RES,
                        chunk_size=chunk, mip_ratio=mip_im)
        # dst.createInfo(output_im, "im")  # uncomment on first run
        fns = [Di2 % z for z in range(SZ[0])]
        tile_sz = SZ[1:]

        def get_im(z0, z1, y0, y1, x0, x1, ratio):
            return read_tile_volume(
                fns, z0 * ratio[2], z1 * ratio[2], y0, y1, x0, x1,
                tile_sz=tile_sz, tile_st=[1, 1], zstep=ratio[2],
                tile_ratio=[1.0 / ratio[0], 1.0 / ratio[1]],
            )
        ran = range(2, 4)  # adjust per session
        dst.create_tile_parallel(
            get_im, output_im, "im", ran, do_subdir=True,
            num_thread=nt, start_chunk=job_id, step_chunk=job_num,
        )

    # opt 1.1 (seg tile) / 1.2 (mesh) — wire in once we have a seg volume to ship
    # leaving stubs to keep the dispatch table honest
    elif opt in ("1.1", "1.2"):
        raise NotImplementedError(
            "seg/mesh upload requires a finalized seg.h5; populate `seg` then enable."
        )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    opt, job_id, job_num = parse_argv()
    if opt[0] == "0":
        opt_0(opt, job_id, job_num)
    elif opt[0] == "1":
        opt_1(opt, job_id, job_num)
    else:
        raise SystemExit("unknown opt: %r" % opt)
