#!/bin/bash
#SBATCH --job-name=ml_reg_cpu
#SBATCH --output=logs/ml_reg_cpu_%j.out
#SBATCH --error=logs/ml_reg_cpu_%j.err
#SBATCH --time=180:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zs5958@princeton.edu
#SBATCH --nodes=1

module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p ml_results_cpu logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

python ml_regression.py \
  --features-csv graph_features_coarse_T300K_lite.csv \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --out-dir ml_results_cpu \
  --impute \
  --targets log_MFPT_AB log_MFPT_BA log_t1 t1_over_t2 \
  --n-perm-repeats 100

echo "End: $(date)"
