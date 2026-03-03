#!/bin/bash
#SBATCH --job-name=prep_all
#SBATCH --output=logs/prep_all_%j.out
#SBATCH --error=logs/prep_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load anaconda3/2024.6
source activate thesis_ktn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/gpfs/JERELLE/harry/thesis_code

echo "[prep_all] Running run_all_build.py..."
python run_all_build.py

echo "[prep_all] Running run_all_coarse_grain.py..."
python run_all_coarse_grain.py

echo "[prep_all] Done."