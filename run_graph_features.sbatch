#!/bin/bash
#SBATCH --job-name=graph_features
#SBATCH --output=logs/graph_features_%j.out
#SBATCH --error=logs/graph_features_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zs5958@princeton.edu
#SBATCH --nodes=1

# ---------- environment ----------
module purge
module load anaconda3/2024.6
source activate thesis_ktn

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- paths ----------
CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "NodeList: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================="

# ---------- run ----------
python graph_features.py --T 300 --out graph_features_coarse_T300K.csv

echo "End: $(date)"
