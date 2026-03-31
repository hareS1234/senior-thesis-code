#!/bin/bash
#SBATCH --job-name=commit_base
#SBATCH --output=logs/commit_base_%j.out
#SBATCH --error=logs/commit_base_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
mkdir -p linear_baseline_results logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

python committor_linear_baseline.py \
  --root ktn_pyg_data_v2 \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --task committor \
  --out-dir linear_baseline_results

echo "End: $(date)"
