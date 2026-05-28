"""Shared dataset roots. Override via env vars if your mount points differ.

These mirror the cluster paths in the legacy `T_nag*.py` / `T_xt_ct.py`
scripts. See `docs/data_paths.md` for the curated list.
"""
import os

# Lichtman lab storage
Dl = os.environ.get("UCT_DL", "/n/boslfs/LABS/lichtman_lab/Donglai/")
Dl2 = os.environ.get("UCT_DL2", "/n/boslfs02/LABS/lichtman_lab/donglai/")
# VCG connectomics storage
Dv = os.environ.get(
    "UCT_DV", "/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/vcg_connectomics/"
)
# Public webhost (neuroglancer precomputed)
Dw = os.environ.get("UCT_DW", "/n/holylfs05/LABS/pfister_lab/Everyone/public/")
