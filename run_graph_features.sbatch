#!/bin/bash
#SBATCH --job-name=graph_feat_cpu
#SBATCH --output=logs/graph_feat_cpu_%j.out
#SBATCH --error=logs/graph_feat_cpu_%j.err
#SBATCH --time=150:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
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
mkdir -p logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================="

python graph_features.py \
  --T 300 \
  --out graph_features_coarse_T300K_lite.csv \
  --lite \
  --resume \
  --save-every 1

echo "End: $(date)"
