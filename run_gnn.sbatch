#!/bin/bash
#SBATCH --job-name=gnn_ktn
#SBATCH --output=logs/gnn_ktn_%j.out
#SBATCH --error=logs/gnn_ktn_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ---------- environment ----------
module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- paths ----------
CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p gnn_results logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "=================="

# ---------- Step 1: Build PyG dataset (CPU, runs once) ----------
echo "--- Building PyG dataset ---"
python ktn_dataset.py --root ktn_pyg_data --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv

# ---------- Step 2: Train all modes ----------
echo "--- Training GNN (all modes) ---"
python train_gnn.py \
  --mode all \
  --root ktn_pyg_data \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --hidden-dim 64 \
  --n-layers 3 \
  --conv-type nnconv \
  --n-seeds 5 \
  --out-dir gnn_results

echo "End: $(date)"
