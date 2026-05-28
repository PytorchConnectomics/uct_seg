"""NucMM curation: dust removal + good/bad split + VAST anchor tree.

Ported from `T_nag_pf.py`. Stages:

    0      build pf-only annotation overlays (bv/crack/cb pngs)
    1      stat-based filter on YL+Zudi predictions, output good-id list
    1.1    IoU + sphericity stats per-id (writes cb_pred_stat.h5)
    1.2    merge YL gt + curated Zudi preds + crack/bright mask
    1.21   write VAST meta.txt for cb review
    1.25   Krishna v3/v4/v9/mumu meta relabel + cc3d + dust removal + review meta
    1.251  Krishna v7 meta-only re-org
    1.26   export per-slice VAST pngs from h5
    1.3    NucMM v1/v3 fix-up + write new review meta
    2      Zudi BCD prediction → per-slice VAST pngs
"""
import os
import sys
import numpy as np
import h5py
from imageio import imread, imwrite

from uct_seg.util.io import readh5, writeh5, vast2seg, seg2vast, mkdir
from uct_seg.util.vast import vast_meta_relabel
from uct_seg.util.seg_ops import seg_iou3d, remove_seg
from uct_seg.nuclei.postproc import cc3d_label, filter_dust
from uct_seg.nuclei.curate import split_good_bad_by_shape, write_review_meta
from _paths import Dv, Dl, Dl2

SZ_NUC = (700, 996, 968)


def opt_0(opt):
    D0 = Dv + "CT/Zeiss_Nag/yl_pf/"

    if opt == "0":
        # zudi-pred as VAST annotation template
        fn = "/n/pfister_lab2/Lab/zudilin/data/NucMM/Mouse/U3D_BC_0202/uCT_Mouse_Segm.h5"
        with h5py.File(fn, "r") as h:
            fid = h["main"]
            Do = Dl + "Zeiss_Nag/pf_nucleus/%04d.png"
            for z in range(fid.shape[0]):
                imwrite(Do % z, seg2vast(np.array(fid[z])))

    elif opt == "0.1":
        f0 = h5py.File(D0 + "crack_yl_500um_v4.h5", "r")["main"]
        f1 = h5py.File(D0 + "yl_bright_agg_500um_bv_v2.h5", "r")["main"]
        Do = Dl + "Zeiss_Nag/nucmm/pf_bv-crack/%04d.png"
        for z in range(f0.shape[0]):
            s0 = (np.array(f0[z]) > 0).astype(np.uint8)
            s1 = 2 * (np.array(f1[z]) > 0).astype(np.uint8)
            s0[s1 > 0] = s1[s1 > 0]
            imwrite(Do % z, s0)

    elif opt == "0.3":
        import cv2
        Di = Dl + "Zeiss_Nag/8bit_normalized/Mouse-Cortex_ZeissCT%04d.png"
        Do = Dl + "Zeiss_Nag/nucmm/im/%04d.png"
        for z in range(700):
            im = imread(Di % (z * 2))
            im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            imwrite(Do % z, im)


def opt_1(opt):
    D0 = Dv + "CT/Zeiss_Nag/yl_pf/"
    Do = "db/nucmm/"
    mkdir(Do, "all")

    if opt == "1":
        sn = Do + "cb_yl.txt"
        if not os.path.exists(sn):
            s0 = readh5(D0 + "cb_yl_500um_v3.h5")
            ui, uc = np.unique(s0, return_counts=True)
            np.savetxt(sn, np.vstack([ui, uc]).T, "%d")
        stat = np.loadtxt(sn).astype(int)
        ui, uc = stat[:, 0], stat[:, 1]
        uc[ui == 0] = 0
        print(np.percentile(uc, [10, 30, 50, 70, 90]))

    elif opt == "1.1":
        # imu.misc lives outside this repo — surface the dep clearly
        from imu.misc import getSphericity
        sn_iou = Do + "cb_iou.txt"
        if not os.path.exists(sn_iou):
            s0 = readh5(D0 + "cb_yl_500um_v3.h5")
            s1 = readh5(Dv + "../zudilin/data/NucMM/Mouse/U3D_BC_0202/uCT_Mouse_Segm.h5")
            f0 = readh5(D0 + "crack_yl_500um_v4.h5")
            f1 = readh5(D0 + "yl_bright_agg_500um_bv_v2.h5")
            s1[f0 > 0] = 0
            s1[f1 > 0] = 0
            iou = seg_iou3d(s1, s0)
            np.savetxt(sn_iou, iou, "%d")
        else:
            iou = np.loadtxt(sn_iou).astype(int)

        sn_stat = Do + "cb_pred_stat.h5"
        if not os.path.exists(sn_stat):
            s1 = readh5(Dv + "../zudilin/data/NucMM/Mouse/U3D_BC_0202/uCT_Mouse_Segm.h5")
            f0 = readh5(D0 + "crack_yl_500um_v4.h5")
            f1 = readh5(D0 + "yl_bright_agg_500um_bv_v2.h5")
            s1[f0 > 0] = 0
            s1[f1 > 0] = 0
            sid, sphe, vol = getSphericity(s1)
            writeh5(sn_stat, [sid, sphe, vol], ["sid", "sphe", "vol"])
        else:
            sid, sphe, vol = readh5(sn_stat, ["sid", "sphe", "vol"])

        iou_score = iou[:, -1] / iou[:, -3:-1].max(axis=1).astype(float)
        gid = (vol > 900) & (vol < 2300) & (sphe > 0.7) & (np.hstack([[0], iou_score]) < 0.2)
        np.savetxt(Do + "cb_pred_add-id.txt", sid[gid], "%d")

    elif opt == "1.2":
        s0 = h5py.File(D0 + "cb_yl_500um_v3.h5", "r")["main"]
        s1 = h5py.File(Dv + "../zudilin/data/NucMM/Mouse/U3D_BC_0202/uCT_Mouse_Segm.h5", "r")["main"]
        f0 = h5py.File(D0 + "crack_yl_500um_v4.h5", "r")["main"]
        f1 = h5py.File(D0 + "yl_bright_agg_500um_bv_v2.h5", "r")["main"]

        Do_png = Dl + "Zeiss_Nag/nucmm/pf_cb/%04d.png"
        mm = 7000
        gid = np.loadtxt("db/nucmm/cb_pred_add-id.txt").astype(int)
        rl = np.zeros(20000, np.uint16)
        rl[gid] = mm + np.arange(len(gid)).astype(np.uint16)
        for z in range(s0.shape[0]):
            seg = np.array(s0[z])
            seg2 = rl[np.array(s1[z])]
            seg2[np.array(f0[z]) > 0] = 0
            seg2[np.array(f1[z]) > 0] = 0
            seg[seg2 > 0] = seg2[seg2 > 0]
            imwrite(Do_png % z, seg2vast(seg))

    elif opt == "1.21":
        from uct_seg.util.vast import write_vast_anchor_tree_by_id
        fn = Dl + "Zeiss_Nag/nucmm/pf_cb/meta.txt"
        s0 = readh5(D0 + "cb_yl_500um_v3.h5")
        ui = np.unique(s0)
        gid = ui[ui > 0]
        new_ids = 7000 + np.arange(len(np.loadtxt("db/nucmm/cb_pred_add-id.txt"))).astype(np.uint16)
        write_vast_anchor_tree_by_id(
            fn, [gid, new_ids], np.zeros((8000, 6), int), pref="cb", nn=["good", "new"]
        )

    elif opt == "1.25":
        # Krishna review-pass: re-org VAST tree → fused seg.h5 → curation meta
        from imu.misc import getSphericity
        cfgs = [
            dict(fn=Dl + "Zeiss_Nag/nucmm/v1/", pref="", kw_bad=["bad", "not_nuc"],
                 kw_nm=["good", "new"], do_meta=True),
            dict(fn=Dl + "Zeiss_Nag/nucmm/v3/", pref="", kw_bad=["bad"],
                 kw_nm=["good"], do_meta=False),
            dict(fn=Dl + "Zeiss_Nag/nucmm/v4/", pref="", kw_bad=["bad"],
                 kw_nm=["good"], do_meta=False),
            dict(fn=Dl2 + "nucmm/export/", pref="", kw_bad=["bad"],
                 kw_nm=["good"], do_meta=False),
            dict(fn=Dl2 + "nucmm/export_v9/", pref="", kw_bad=["bad"],
                 kw_nm=["good", "check"], do_meta=False),
            dict(fn=Dl2 + "nucmm/mumu50/", pref="", kw_bad=["bad"],
                 kw_nm=["Mumu", "good"], do_meta=False, sz=(50, 996, 968)),
        ]
        # Pick one to run via env, defaulting to v1 review
        chosen = int(os.environ.get("UCT_CURATE_IDX", "0"))
        c = cfgs[chosen]
        sz = c.get("sz", SZ_NUC)
        sn_seg = c["fn"] + "seg.h5"
        seg = np.zeros(sz, np.uint16)
        rl = vast_meta_relabel(c["fn"] + "meta.txt", kw_bad=c["kw_bad"],
                               kw_nm=c["kw_nm"], do_print=True)
        for i in range(sz[0]):
            seg[i] = rl[vast2seg(imread(c["fn"] + c["pref"] + "_s%03d.png" % i))]
        seg_o = cc3d_label(seg).astype(np.uint16)
        seg_o = filter_dust(seg_o, dust_sz=25)
        writeh5(sn_seg, (seg_o > 0) * seg)

        if c["do_meta"]:
            ui, sphe, uc = getSphericity(seg)
            uc[ui == 0] = 0
            gid, bid = split_good_bad_by_shape(ui, sphe, uc)
            print("bad", len(bid))
            write_review_meta(c["fn"] + "../v1_pf/meta.txt", gid, bid)


def opt_2(opt):
    if opt == "2":
        fn = "/n/boslfs02/LABS/lichtman_lab/zudi/nucmm/bcd_segm.h5"
        with h5py.File(fn, "r") as h:
            fid = h["main"]
            Do = Dl2 + "nucmm/zudi_0606/%04d.png"
            for z in range(fid.shape[0]):
                imwrite(Do % z, seg2vast(np.array(fid[z])))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    opt = sys.argv[1]
    if opt[0] == "0":
        opt_0(opt)
    elif opt[0] == "1":
        opt_1(opt)
    elif opt[0] == "2":
        opt_2(opt)
    else:
        raise SystemExit("unknown opt: %r" % opt)
