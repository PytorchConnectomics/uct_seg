"""Generate sample sbatch scripts via uct_seg.slurm.gen_jobs."""
import os

import _bootstrap  # noqa: F401
from uct_seg.util.io import mkdir
from uct_seg.slurm.gen_jobs import write_jobs

OUT = os.path.join(os.path.dirname(__file__), "output", "slurm_demo")
mkdir(OUT, "all")


def main():
    paths = write_jobs(
        name="prep_nucmm",
        cmd="python scripts/prep_nucmm.py 0.2",   # CLAHE pass
        num=4,
        out_dir=OUT,
        mem=8000,
        do_gpu=False,
        partition="cox",
        time="0-04:00",
        conda_env="imu",
    )
    print(f"[slurm] wrote {len(paths)} scripts to {OUT}")
    print("------- prep_nucmm_0.sh -------")
    print(open(paths[0]).read())
    print(
        f"submit: for i in {{0..{len(paths) - 1}}}; do "
        f"sbatch {OUT}/prep_nucmm_${{i}}.sh && sleep 1; done"
    )


if __name__ == "__main__":
    main()
