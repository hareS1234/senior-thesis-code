#!/bin/bash
#SBATCH --job-name=summary_all
#SBATCH --output=logs/summary_all_%j.out
#SBATCH --error=logs/summary_all_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module purge
module load anaconda3/2024.6
source activate thesis_ktn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/gpfs/JERELLE/harry/thesis_code

python run_all_sequences.py

echo "[summary_all] Done."
