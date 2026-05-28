"""Blood vessel segmentation.

Pipeline (current draft):
  1. preproc:  CLAHE + median (see `uct_seg.bv.preproc.preprocess_volume`)
  2. train:    PyTC config `configs/bv.yaml` (3D U-Net, BCE+Dice)
  3. infer:    PyTC CLI → uint8 prob map → threshold + cc3d
  4. postproc: dust removal (`uct_seg.seg_ops.remove_small`), per-id bbox

Reference data layout: see `docs/data_paths.md` (Dyer-17 V0–V4, Nag NucMM).
"""
from . import preproc  # noqa: F401
