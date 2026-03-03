#!/bin/bash
#SBATCH --job-name=GT_kept_T300K
#SBATCH --output=logs/GT_kept_T300K_%j.out
#SBATCH --error=logs/GT_kept_T300K_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ---------- environment ----------
module purge
module load anaconda3/2024.6
source activate thesis_ktn

# Avoid oversubscription if any BLAS is used
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- paths ----------
CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
ROOT=/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped

mkdir -p $CODEDIR/logs
cd $CODEDIR || exit 1

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "CWD: $(pwd)"
echo "Start: $(date)"
echo "=================="

# ---------- run ----------
python build_gt_kept_models.py \
  --root $ROOT \
  --T 300 \
  --style hybrid \
  --percent-retained 5 \
  --min-kept 200 \
  --block 50 \
  --cond-thresh 1e13 \
  --screen \
  --overwrite \
  --report logs/gt_kept_build_report_T300K_${SLURM_JOB_ID}.txt

echo "End: $(date)"
