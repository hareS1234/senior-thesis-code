#!/bin/bash
#SBATCH --job-name=mfpt_micro
#SBATCH --output=logs/mfpt_micro_%j.out
#SBATCH --error=logs/mfpt_micro_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load Anaconda
module purge
module load anaconda3/2024.6

# Activate your thesis environment
source activate thesis_ktn

# Use all allocated cores for BLAS/MKL
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# DPS directory and temperature
DPS="/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped/yyggyy_nocap/yyggyy_99idps_nocap"
T=300

# Go to code directory
cd /scratch/gpfs/JERELLE/harry/thesis_code

# Run microscopic MFPT analysis (no --coarse)
python mfpt_analysis.py --data-dir "$DPS" --T "$T" --max-eigs 5
