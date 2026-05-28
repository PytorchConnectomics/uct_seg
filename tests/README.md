# uct_seg tests

Self-contained runnable smoke tests. Synthetic data only — no cluster paths required.

All outputs land in `tests/output/` (gitignored).

| Test | Shows | Extra deps |
|---|---|---|
| `test_io.py` | h5 read/write, VAST png↔uint32 roundtrip, mkdir, bbox | core |
| `test_seg_ops.py` | relabel, remove_seg, get_bb_label3d_v2, seg_iou3d | core |
| `test_vast.py` | meta parse, good/bad relabel, anchor tree writer | core |
| `test_imops.py` | imAdjust percentile clip, CLAHE | `[cv]` |
| `test_butterfly.py` | write_bfly_v2 manifest JSON | core |
| `test_nuclei_postproc.py` | cc3d label → dust removal → good/bad shape split | `[cc]` |
| `test_slurm.py` | sbatch script generation | core |
| `test_viz.py` | neuroglancer LocalVolume + viewer URL | `[viz]` |

## Run

```bash
cd tests
python test_io.py
python test_seg_ops.py
# ...
```

Or sweep all:

```bash
for f in tests/test_*.py; do echo "=== $f ==="; python "$f" || break; done
```

Or via pytest:

```bash
pytest tests/
```
