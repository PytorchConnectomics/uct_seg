"""Large dendrite segmentation (TODO).

Planned pipeline:
  - YL semantic labels (dendrites class)
  - 2x upsample to native res (see `T_nag.py` opt 1.1 reference)
  - 3D U-Net via `configs/dendrite.yaml`
  - skeletonization for downstream cable-length stats
"""
