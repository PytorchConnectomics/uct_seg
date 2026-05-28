"""Artifact masking: cracks + bright aggregates.

Cracks and bright-aggregate masks are typically produced by a small 3D U-Net
(separate model from BV/nuclei) trained on Yulia-Lin (yl) annotations.
This module handles the *use* of those masks downstream, i.e. fusing them
into a single labelmap and removing affected seg-ids.
"""
from . import mask  # noqa: F401
