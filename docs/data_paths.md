# Known dataset paths (Harvard FAS cluster)

Override roots via env vars (`UCT_DL`, `UCT_DL2`, `UCT_DV`, `UCT_DW`) — see
`scripts/_paths.py`.

## NucMM / Zeiss-Nag mouse cortex (720 nm iso)

| Path | Contents |
|---|---|
| `$UCT_DL/Zeiss_Nag/8bit_normalized/` | source 8-bit png stack `Mouse-Cortex_ZeissCT%04d.png`, sz=1942×1992×1936 |
| `$UCT_DL/Zeiss_Nag/8bit_normalized_yz[_clahe]/` | yz re-slice (raw + CLAHE) |
| `$UCT_DV/CT/Zeiss_Nag/im_720nm/%04d.png` | downsampled 720nm png, sz_nuc=700×996×968 |
| `$UCT_DV/CT/Zeiss_Nag/im_700.h5` | same as h5 |
| `$UCT_DV/CT/Zeiss_Nag/yl_pf/cb_yl_500um_v3.h5` | Yulia-Lin nucleus annotation, 500 μm subvol |
| `$UCT_DV/CT/Zeiss_Nag/yl_pf/crack_yl_500um_v4.h5` | YL crack mask |
| `$UCT_DV/CT/Zeiss_Nag/yl_pf/yl_bright_agg_500um_bv_v2.h5` | YL bright-aggregate (BV) mask |
| `$UCT_DV/CT/Zeiss_Nag/cb_gt_0219.h5` | curated nucleus GT (2022-02-19) |
| `$UCT_DV/CT/Zeiss_Nag/train_cb/{im,seg_cb}_500.h5` | Zergham 500³ training crop |
| `$UCT_DV/CT/Zeiss_Nag/vol_pyr/` | pyramidal-cell sub-volumes (bb_vol0/1.txt, gt_vol0/1.h5) |
| `$UCT_DV/CT/Zeiss_Nag/yl/seg_ds2_label_open1_{cell,dendrites}.h5` | YL semantic dendrites |

Krishna review passes (export iterations of VAST annotations):

| Path | Notes |
|---|---|
| `$UCT_DL/Zeiss_Nag/nucmm/v{1,3,4}/` | v1 = `kw_bad=[bad,not_nuc]`, others `[bad]` |
| `$UCT_DL2/nucmm/export[_v7,_v9]/` | v7 uses `[merger,not_sure,bad]`; v9 uses `[bad]` + `[good,check]` |
| `$UCT_DL2/nucmm/mumu50/` | 50-slice mumu subset, `[Mumu,good]` |

## Zudi-Lin predictions (NucMM benchmark)

| Path | Contents |
|---|---|
| `/n/pfister_lab2/Lab/zudilin/data/NucMM/Mouse/U3D_BC_0202/uCT_Mouse_Segm.h5` | Zudi U3D-BC nucleus seg |
| `/n/boslfs02/LABS/lichtman_lab/zudi/nucmm/bcd_segm.h5` | Zudi BCD seg (2022-06-06) |

## Dyer-17 (Argonne synchrotron, 0.65 μm iso)

| Path | Contents |
|---|---|
| `$UCT_DV/Dyer17/data/original_img_data_0pt65microniso.nrrd` | whole-volume raw |
| `$UCT_DV/Dyer17/data/proj4_masked_390_2014/*.tif` | masked sub-volume |
| `$UCT_DV/Dyer17/data/train/V{0..4}/` | per-volume training data (img + bv + cell annos) |

## Xiaotang synchrotron (1.38 μm iso, mouse 5mo recon)

| Path | Contents |
|---|---|
| `$UCT_DL2/../Lab/Xiaotang_synchrotron/mouse_5mo_rec/recon_%05d.tiff` | float32 tiff stack, sz=11088×6464×6464 |
| `$UCT_DL2/CT/xt_202305/mouse_5mo_rec_80_995/%05d.png` | 8-bit quantized output |

Pre-computed intensity range (from previous pass): `[8.84428e-06, 6.62667e-04]`.

## Neuroglancer precomputed (web-served)

| URL prefix | Contents |
|---|---|
| `file:///n/pfister_lab2/Lab/public/ng/nucmm_mouse_im` | NucMM image |
| `file:///n/pfister_lab2/Lab/public/ng/nucmm_mouse_seg` | NucMM curated cb seg |
| `file://$UCT_DW/ng/xt_ct/mouse_5mo_rec/` | Xiaotang CT pyramid |

Demo viewer link template: see `neuroglancer-demo.appspot.com` URL block in
the original `ct/T_nag.py` header.
