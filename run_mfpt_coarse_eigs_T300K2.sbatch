#!/bin/bash
#SBATCH --job-name=mfpt_coarse_eigs_T300K
#SBATCH --output=logs/mfpt_coarse_eigs_T300K_%j.out
#SBATCH --error=logs/mfpt_coarse_eigs_T300K_%j.err
#SBATCH --time=200:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
ROOT=/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped
TAG=T300K

mkdir -p $CODEDIR/logs
cd $CODEDIR || exit 1

echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "Start: $(date)"
echo "CWD: $(pwd)"

python - <<'PY'
from pathlib import Path
import subprocess, sys

root = Path("/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped")
T = 300
tag = f"T{T}K"

# Optional: resume by sequence (parent folder), e.g. "klvffa_nocap"
START_SEQ = "regger_nocap"   # set to None to disable "start from"
started = (START_SEQ is None)

def is_done(dps: Path) -> bool:
    """
    Define 'done' as: AB kinetics + spectrum files exist in the coarse out dir.
    Adjust if you only care about AB_kinetics.
    """
    out_dir = dps / f"markov_{tag}" / f"GT_kept_{tag}"
    return (
        (out_dir / f"AB_kinetics_{tag}.npz").exists()
        and (out_dir / f"eigenvalues_{tag}.npy").exists()
        and (out_dir / f"timescales_{tag}.npy").exists()
    )

# Find DPS dirs (the directory that contains markov_T300K)
markov_dirs = sorted(root.glob(f"*/*/markov_{tag}"))
dps_dirs = sorted({p.parent for p in markov_dirs})

print(f"Found {len(dps_dirs)} DPS dirs with markov_{tag}.")

fail = 0
ran = 0
skipped_done = 0
skipped_before = 0

for dps in dps_dirs:
    seq = dps.parent.name  # e.g. "klvffa_nocap"

    if not started:
        if seq < START_SEQ:
            skipped_before += 1
            continue
        started = True

    if is_done(dps):
        skipped_done += 1
        continue

    cmd = ["python", "mfpt_analysis.py",
           "--data-dir", str(dps),
           "--T", str(T),
           "--coarse",
           "--max-eigs", "5"]
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    ran += 1
    if r.returncode != 0:
        print("[FAIL]", dps, "returncode", r.returncode, file=sys.stderr)
        fail += 1

print(f"DONE. ran={ran} skipped_done={skipped_done} skipped_before={skipped_before} failures={fail}")
sys.exit(1 if fail else 0)
PY

echo "End: $(date)"
