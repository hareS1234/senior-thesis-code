#!/bin/bash
#SBATCH --job-name=gnn_cpu_small
#SBATCH --output=logs/gnn_cpu_small_%j.out
#SBATCH --error=logs/gnn_cpu_small_%j.err
#SBATCH --time=180:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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
mkdir -p gnn_results_cpu logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

python train_gnn.py \
  --mode node \
  --root ktn_pyg_data_cpu \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --task committor \
  --hidden-dim 32 \
  --n-layers 2 \
  --conv-type gcn \
  --batch-size 1 \
  --node-epochs 60 \
  --patience 15 \
  --device cpu \
  --out-dir gnn_results_cpu

echo "End: $(date)"
