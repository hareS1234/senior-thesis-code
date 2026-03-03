#!/bin/bash
#SBATCH --job-name=mfpt_coarse_eigs_T300K
#SBATCH --output=logs/mfpt_coarse_eigs_T300K_%j.out
#SBATCH --error=logs/mfpt_coarse_eigs_T300K_%j.err
#SBATCH --time=72:00:00
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

markov_dirs = sorted(root.glob("*/*/markov_T300K"))
dps_dirs = sorted({p.parent for p in markov_dirs})

print(f"Found {len(dps_dirs)} DPS dirs with markov_T300K.")
fail = 0

for dps in dps_dirs:
    cmd = ["python","mfpt_analysis.py",
           "--data-dir", str(dps),
           "--T", str(T),
           "--coarse",
           "--max-eigs","5"]
    print("[RUN]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("[FAIL]", dps, "returncode", r.returncode, file=sys.stderr)
        fail += 1

print("DONE. failures =", fail)
sys.exit(1 if fail else 0)
PY

echo "End: $(date)"
