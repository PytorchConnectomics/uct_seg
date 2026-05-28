"""Generate SLURM sbatch scripts for parallel CT segmentation jobs.

Ported from `T_slurm.py`. CLI entry point: `uct-slurm-gen`.

Each generated script runs:

    <cmd> <job_id> <num>

so the target script must accept (job_id, job_num) as its last two args.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def sbatch_header(
    mem: int = 10000,
    do_gpu: bool = False,
    partition: str = "cox",
    cores: int = 1,
    time: str = "4-00:00",
    log_dir: str = "logs/slurm",
) -> str:
    h = "#!/bin/bash\n"
    h += "#SBATCH -N 1\n"
    h += f"#SBATCH -p {partition}\n"
    h += f"#SBATCH -n {cores}\n"
    h += f"#SBATCH --mem {mem}\n"
    if do_gpu:
        h += "#SBATCH --gres=gpu:1\n"
    h += f"#SBATCH -t {time}\n"
    h += f"#SBATCH -o {log_dir}/slurm.%N.%j.out\n"
    h += f"#SBATCH -e {log_dir}/slurm.%N.%j.err\n\n"
    return h


def write_jobs(
    name: str,
    cmd: str,
    num: int,
    out_dir: str = "db/slurm",
    mem: int = 10000,
    do_gpu: bool = False,
    partition: str = "cox",
    cores: int = 1,
    time: str = "4-00:00",
    conda_env: str | None = None,
    log_dir: str | None = None,
) -> list[str]:
    """Write `num` sbatch scripts named `<name>_<i>.sh`. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        log_dir = str(out_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    header = sbatch_header(mem, do_gpu, partition, cores, time, log_dir)
    paths = []
    for i in range(num):
        body = ""
        if conda_env:
            body += f"source activate {conda_env}\n"
        body += f"{cmd} {i} {num}\n"
        p = out_dir / f"{name}_{i}.sh"
        p.write_text(header + body)
        paths.append(str(p))
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SLURM array of CT segmentation jobs.")
    p.add_argument("name", help="Job name prefix (output: <name>_<i>.sh)")
    p.add_argument("--cmd", required=True, help='Command to run, e.g. "python scripts/prep_nucmm.py 0"')
    p.add_argument("--num", type=int, required=True, help="Number of parallel jobs")
    p.add_argument("--out-dir", default="db/slurm")
    p.add_argument("--mem", type=int, default=10000)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--partition", default="cox")
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--time", default="4-00:00")
    p.add_argument("--conda-env", default=None)
    args = p.parse_args()

    paths = write_jobs(
        name=args.name,
        cmd=args.cmd,
        num=args.num,
        out_dir=args.out_dir,
        mem=args.mem,
        do_gpu=args.gpu,
        partition=args.partition,
        cores=args.cores,
        time=args.time,
        conda_env=args.conda_env,
    )
    print(f"wrote {len(paths)} scripts to {args.out_dir}")
    print(
        f"submit: for i in {{0..{args.num - 1}}}; do "
        f"sbatch {args.out_dir}/{args.name}_${{i}}.sh && sleep 1; done"
    )


if __name__ == "__main__":
    main()
