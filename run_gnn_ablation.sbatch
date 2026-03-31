#!/bin/bash
#SBATCH --job-name=gnn_abl
#SBATCH --output=logs/gnn_abl_%A_%a.out
#SBATCH --error=logs/gnn_abl_%A_%a.err
#SBATCH --array=0-44
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zs5958@princeton.edu
#SBATCH --nodes=1

# ──────────────────────────────────────────────────────────────────────
#  GNN ablation sweep: 45 configs (5 top_k × 3 hidden × 3 depth)
#
#  Submit:   sbatch run_gnn_ablation.sbatch
#  Monitor:  squeue -u $USER -n gnn_abl
#  After:    python gnn_ablation_aggregate.py --results-dir gnn_ablation_results
# ──────────────────────────────────────────────────────────────────────

module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p gnn_ablation_results logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID,  ArrayTaskID: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"
echo "=================="

python gnn_ablation_sweep.py \
  --task-id    $SLURM_ARRAY_TASK_ID \
  --root       ktn_pyg_data_v2 \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --task       committor \
  --epochs     150 \
  --patience   25 \
  --device     cpu \
  --out-dir    gnn_ablation_results

echo "End: $(date)"
