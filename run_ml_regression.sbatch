#!/bin/bash
#SBATCH --job-name=ml_regression
#SBATCH --output=logs/ml_regression_%j.out
#SBATCH --error=logs/ml_regression_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ---------- environment ----------
module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- paths ----------
CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p ml_results_strict ml_results_imputed logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

# ---------- run (strict mode: drop rows with NaN features) ----------
echo "--- Strict mode ---"
python ml_regression.py \
  --features-csv graph_features_coarse_T300K.csv \
  --targets-csv  GTcheck_micro_vs_coarse_T300K_full.csv \
  --out-dir      ml_results_strict

# ---------- run (impute mode: median-fill NaN features) ----------
echo "--- Impute mode ---"
python ml_regression.py \
  --features-csv graph_features_coarse_T300K.csv \
  --targets-csv  GTcheck_micro_vs_coarse_T300K_full.csv \
  --out-dir      ml_results_imputed \
  --impute

echo "End: $(date)"
