#!/bin/bash
#SBATCH --job-name=mfpt_coarse_all
#SBATCH --output=logs/mfpt_coarse_all_%j.out
#SBATCH --error=logs/mfpt_coarse_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load anaconda3/2024.6
source activate thesis_ktn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/gpfs/JERELLE/harry/thesis_code

python - << 'EOF'
from config import iter_dps_dirs, TEMPERATURES
import subprocess, sys

for dps_dir in iter_dps_dirs():
    for T in TEMPERATURES:
        cmd = [
            "python",
            "mfpt_analysis.py",
            "--data-dir",
            str(dps_dir),
            "--T",
            str(T),
            "--coarse",
            "--max-eigs",
            "5",
        ]
        print("[mfpt_coarse_all] Running:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[mfpt_coarse_all] Command failed with code",
                  result.returncode, file=sys.stderr)
            sys.exit(result.returncode)
EOF

echo "[mfpt_coarse_all] Done."