#!/bin/bash
#SBATCH --job-name=mfpt_coarse
#SBATCH --output=logs/mfpt_coarse_%j.out
#SBATCH --error=logs/mfpt_coarse_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module purge
module load anaconda3/2024.6
source activate thesis_ktn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DPS="/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped/yyggyy_nocap/yyggyy_99idps_nocap"
T=300

cd /scratch/gpfs/JERELLE/harry/thesis_code

python mfpt_analysis.py --data-dir "$DPS" --T "$T" --coarse --max-eigs 5
