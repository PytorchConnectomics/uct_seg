"""Nuclei (cell body, cb) segmentation + curation.

Two stages:
  - postproc: cc3d → dust removal → sphericity/size filter → IoU vs GT
  - curate:   VAST anchor tree generation for human review (good/bad split)

Training itself runs through PyTC (`configs/nuclei.yaml`).
"""
from . import postproc, curate  # noqa: F401
