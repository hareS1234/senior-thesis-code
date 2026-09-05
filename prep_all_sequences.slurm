#!/bin/bash
#SBATCH --job-name=prep_all
#SBATCH --output=logs/prep_all_%j.out
#SBATCH --error=logs/prep_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

module purge
module load anaconda3/2024.6
source activate thesis_ktn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CODEDIR="${THESIS_CODE_DIR:-/scratch/gpfs/JERELLE/harry/thesis_code}"
DATA_ROOT="${THESIS_DATA_ROOT:-/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped}"

mkdir -p "$CODEDIR/logs"
cd "$CODEDIR"

echo "[prep_all] Running run_all_build.py..."
python run_all_build.py --root "$DATA_ROOT" --temperatures 300

echo "[prep_all] Running build_gt_kept_models.py..."
python build_gt_kept_models.py \
  --root "$DATA_ROOT" \
  --T 300 \
  --style hybrid \
  --percent-retained 5 \
  --min-kept 200 \
  --block 50 \
  --cond-thresh 1e13 \
  --screen \
  --report "logs/gt_kept_build_report_T300K_${SLURM_JOB_ID}.txt"

echo "[prep_all] Microscopic and GT-kept models are ready."
