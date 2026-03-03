#!/bin/bash
#SBATCH --job-name=GT_validate_T300K
#SBATCH --output=logs/GT_validate_T300K_%j.out
#SBATCH --error=logs/GT_validate_T300K_%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load anaconda3/2024.6
source activate thesis_ktn

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
ROOT=/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped

mkdir -p $CODEDIR/logs
cd $CODEDIR || exit 1

python analyze_micro_vs_coarse_T300K.py \
  --root $ROOT \
  --T 300 \
  --out-dir $CODEDIR \
  --prefix GTcheck \
  --mfpt-rtol 1e-6 \
  --stationarity-rtol 1e-10