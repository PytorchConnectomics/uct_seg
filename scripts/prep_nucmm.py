"""NucMM (Zeiss-Nag) mouse cortex prep + neuroglancer publish.

Ported from `T_nag.py`. Usage:

    python prep_nucmm.py <opt> [job_id job_num]

opt:
  8       quantize+resize 8bit_normalized → 720nm png
  8.1     zergham annotation crop (500^3)
  8.11    full 700^3 h5
  8.2     zergham anno seg h5
  8.3     write butterfly JSON
  9       neuroglancer precomputed image (cloud-volume env)
  9.1     neuroglancer precomputed seg (igneous env)
  9.2     neuroglancer mesh (igneous env)
  0       re-slice xy → yz
  0.1     re-slice initial seg
  0.11    yz → 2D labels (per slice)
  0.2     CLAHE on yz slices (use for annotation)
  1       pyramidal cell train data check (gt presence)
  1.01    union bbox over labeled slices
  1.02    crop + write h5 GT
  1.1     fuse YL cb+dendrites into pf gt
  2       crack dataset stack
"""
import os
import sys
import numpy as np
from imageio import imread, imsave

from uct_seg.util.io import readh5, writeh5, vast2seg, seg2vast, get_bb, get_union, mkdir
from uct_seg.util.imops import histogram_clahe
from _paths import Dl, Dv

# Legacy directory aliases
Dl_nag = Dl + "Zeiss_Nag/"
Dc = Dv + "CT/Zeiss_Nag/"

SZ = (1942, 1992, 1936)
SZ8 = (971, 996, 968)
SZ_NUC = (700, 996, 968)


def parse_argv():
    opt = sys.argv[1]
    job_id, job_num = 0, 1
    if len(sys.argv) > 3:
        job_id = int(sys.argv[2])
        job_num = int(sys.argv[3])
    return opt, job_id, job_num


def opt_8(opt):
    # 8bit png ops on Lichtman 8bit_normalized stack
    import cv2

    if opt == "8":
        Di = Dl_nag + "8bit_normalized/Mouse-Cortex_ZeissCT%04d.png"
        Do = Dc + "im_720nm/%04d.png"
        for i in range(SZ_NUC[0]):
            im = imread(Di % (i * 2))
            im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            imsave(Do % i, im)

    elif opt == "8.1":
        Do = Dc + "im_720nm/%04d.png"
        Do2 = Dc + "train_cb/"
        im = np.zeros((500, 500, 500), np.uint8)
        for z in range(100, 600):
            im0 = imread(Do % z)
            im[z - 100] = im0[498 - 250:498 + 250, 484 - 250:484 + 250]
        writeh5(Do2 + "im_500.h5", im)

    elif opt == "8.11":
        Do = Dc + "im_720nm/%04d.png"
        im = np.zeros(SZ_NUC, np.uint8)
        for z in range(SZ_NUC[0]):
            im[z] = imread(Do % z)
        writeh5(Dc + "im_700.h5", im)

    elif opt == "8.2":
        seg = readh5(Dc + "yl_pf/cb_yl_500um_v3.h5")
        writeh5(
            Dc + "train_cb/seg_cb_500.h5",
            seg[100:-100, 498 - 250:498 + 250, 484 - 250:484 + 250],
        )

    elif opt == "8.3":
        from uct_seg.util.butterfly import write_bfly_v2
        fn = Dc + "im_720nm/%04d.png"
        write_bfly_v2(
            sz=(700, 996, 968),
            numT=(1, 1),
            imN=lambda x: fn % x,
            outName=Dc + "im_720.json",
            tsz=4096,
        )


def opt_9(opt):
    # neuroglancer precomputed (requires cloud-volume / igneous envs)
    from emu.ng import ngDataset
    from emu.io import readTileVolume

    Do = "file://" + "/n/pfister_lab2/Lab/public/ng/"
    Di = Dc + "im_720nm/%04d.png"
    volume_size = list(SZ_NUC[::-1])
    resolution = [720, 720, 720]
    mip_im = [[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1], [16, 16, 1], [32, 32, 1]]
    mip_seg = [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16], [32, 32, 32]]
    chunk = [64, 64, 64]
    volume_size[2] = ((volume_size[2] + chunk[2] - 1) // chunk[2]) * chunk[2]
    out_im = Do + "nucmm_mouse_im"
    out_seg = Do + "nucmm_mouse_seg"
    nt = 4

    if opt == "9":
        dst = ngDataset(volume_size=volume_size, resolution=resolution,
                        chunk_size=chunk, mip_ratio=mip_im)
        dst.createInfo(out_im, "im")
        fns = [Di % x for x in range(SZ_NUC[0])]
        tile_sz = SZ_NUC[1:]

        def get_im(z0, z1, y0, y1, x0, x1):
            return readTileVolume(fns, z0, z1, y0, y1, x0, x1, tile_sz=tile_sz)
        dst.createTile(get_im, out_im, "im", range(len(mip_im)),
                       do_subdir=True, num_thread=nt)

    elif opt == "9.1":
        dst = ngDataset(volume_size=volume_size, resolution=resolution,
                        chunk_size=chunk, mip_ratio=mip_seg)
        dst.createInfo(out_seg, "seg")
        seg = readh5(Dc + "cb_gt_0219.h5")

        def get_seg(z0, z1, y0, y1, x0, x1):
            out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint16)
            tmp = seg[z0:z1, y0:y1, x0:x1]
            out[: tmp.shape[0]] = tmp
            return out
        dst.createTile(get_seg, out_seg, "seg", range(len(mip_seg)),
                       do_subdir=True, num_thread=nt)

    elif opt == "9.2":
        dst = ngDataset(volume_size=volume_size, resolution=resolution,
                        chunk_size=chunk, mip_ratio=mip_seg)
        dst.createMesh(out_seg, 1, [484, 498, 352], nt,
                       dust_threshold=10, do_subdir=True)


def opt_0(opt, job_id, job_num):
    # re-slice / CLAHE
    import cv2
    fn = "%04d.png"

    if opt == "0":
        D0 = Dl_nag + "8bit_normalized/"
        D1 = Dl_nag + "8bit_normalized_yz/"
        fn = "Mouse-Cortex_ZeissCT%04d.png"
        out = np.zeros((SZ[1], SZ[0]), np.uint8)
        for x in range(job_id, SZ[2], job_num):
            out[:] = 0
            for z in range(SZ[0]):
                out[:, z] = imread(D0 + fn % z)[:, x]
            imsave(D1 + fn % x, out)

    elif opt == "0.1":
        import glob
        for vid in ("vol1", "vol2"):
            D0 = Dl_nag + vid + "/seg/"
            D1 = Dl_nag + vid + "/seg-yz/"
            mkdir(D1)
            numI = len(glob.glob(D0 + "*.png"))
            sz = imread(D0 + fn % 0).shape
            out = np.zeros((sz[0], numI), np.uint8)
            for x in range(sz[1]):
                out[:] = 0
                for z in range(numI):
                    out[:, z] = imread(D0 + fn % z)[:, x]
                imsave(D1 + fn % x, out)

    elif opt == "0.11":
        import glob
        from skimage.measure import label
        for vid in ("vol1", "vol2"):
            D1 = Dl_nag + vid + "/seg-yz/"
            D2 = Dl_nag + vid + "/seg-yz-2d/"
            mkdir(D2)
            numI = len(glob.glob(D1 + "*.png"))
            cc = 0
            for z in range(numI):
                tmp = label(imread(D1 + fn % z))
                tmp[tmp > 0] += cc
                cc = tmp.max()
                imsave(D2 + fn % z, seg2vast(tmp))

    elif opt == "0.2":
        D0 = Dl_nag + "8bit_normalized_yz/"
        D1 = Dl_nag + "8bit_normalized_yz_clahe/"
        fn = "Mouse-Cortex_ZeissCT%04d.png"
        for x in range(job_id, SZ[2], job_num):
            im = imread(D0 + fn % x)
            im = histogram_clahe(im)
            im = cv2.medianBlur(im, 3)
            imsave(D1 + fn % x, im)


def opt_1(opt):
    # pyramidal cell training-data prep
    Di = Dl_nag + "vol_pyr/"
    Do = Dc + "vol_pyr/"

    if opt == "1":
        for fid in range(446, 714):
            if not os.path.exists(Di + "_s%04d.png" % fid):
                print(fid)

    elif opt == "1.01":
        ffs = [[446, 637], [689, 713]]
        for ii, ff in enumerate(ffs):
            bb = [10000, 0, 10000, 0]
            for fid in range(ff[0], ff[1] + 1):
                if os.path.exists(Di + "_s%04d.png" % fid):
                    im = vast2seg(imread(Di + "_s%04d.png" % fid))
                    bb = get_union(get_bb(im), bb)
            np.savetxt(Do + "bb_vol%d.txt" % ii, ff + bb, "%d")

    elif opt == "1.02":
        ffs = [[446, 637], [689, 713]]
        for ii, ff in enumerate(ffs):
            bb = np.loadtxt(Do + "bb_vol%d.txt" % ii).astype(int)
            sz = (bb[1::2] - bb[::2]) + 1
            ff = [bb[0], bb[1]]
            seg = np.zeros(sz, np.uint8)
            for fid in range(ff[0], ff[1] + 1):
                seg[fid - ff[0]] = vast2seg(imread(Di + "_s%04d.png" % fid))[
                    bb[2]:bb[3] + 1, bb[4]:bb[5] + 1
                ]
            writeh5(Do + "gt_vol%d.h5" % ii, seg)

    elif opt == "1.1":
        from scipy.ndimage import zoom
        bb = np.loadtxt(Do + "bb_vol%d.txt" % 0).astype(int)
        # nag annotation only
        seg = (readh5(Do + "gt_vol%d.h5" % 0) > 0).astype(np.uint8)
        # fuse YL cell
        cb = readh5(Dc + "yl/seg_ds2_label_open1_cell.h5").transpose((2, 1, 0))
        cb = zoom(
            cb[bb[0] // 2:(bb[1] + 1) // 2,
               bb[2] // 2:(bb[3] + 1) // 2,
               bb[4] // 2:(bb[5] + 1) // 2],
            2, order=0,
        )
        seg[cb > 0] = 2
        # fuse YL dendrites
        den = readh5(Dc + "yl/seg_ds2_label_open1_dendrites.h5").transpose((2, 1, 0))
        den = zoom(
            den[bb[0] // 2:(bb[1] + 1) // 2,
                bb[2] // 2:(bb[3] + 1) // 2,
                bb[4] // 2:(bb[5] + 1) // 2],
            2, order=0,
        )
        seg[den > 0] = 3
        writeh5(Do + "gt_v0.h5", seg)


def opt_2(opt):
    if opt == "2":
        out = np.zeros(SZ8, np.uint8)
        for z in range(out.shape[0]):
            out[z] = imread(Dl_nag + "cracks/%04d.png" % z)
        writeh5(Dl_nag + "cracks_stack.h5", out)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    opt, job_id, job_num = parse_argv()
    if opt[0] == "8":
        opt_8(opt)
    elif opt[0] == "9":
        opt_9(opt)
    elif opt[0] == "0":
        opt_0(opt, job_id, job_num)
    elif opt[0] == "1":
        opt_1(opt)
    elif opt[0] == "2":
        opt_2(opt)
    else:
        raise SystemExit("unknown opt: %r" % opt)
