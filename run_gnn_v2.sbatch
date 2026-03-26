#!/bin/bash
#SBATCH --job-name=gnn_v2
#SBATCH --output=logs/gnn_v2_%j.out
#SBATCH --error=logs/gnn_v2_%j.err
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
mkdir -p gnn_results_v2 logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

# Run systematic comparison:
#   - Sparsify to top-20 edges per node
#   - Compare GCN (with enriched node features), GAT, and NNConv
#   - All on CPU with small model (hidden=32, layers=2)
python train_gnn_v2.py \
  --root ktn_pyg_data_v2 \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --task committor \
  --top-k 20 \
  --conv-types gcn gat nnconv \
  --hidden-dim 32 \
  --n-layers 2 \
  --epochs 100 \
  --patience 20 \
  --batch-size 1 \
  --device cpu \
  --out-dir gnn_results_v2

echo "End: $(date)"
