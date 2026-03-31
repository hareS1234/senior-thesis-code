#!/bin/bash
#SBATCH --job-name=lclass_test
#SBATCH --output=logs/lclass_test_%j.out
#SBATCH --error=logs/lclass_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zs5958@princeton.edu
#SBATCH --nodes=1

module purge
module load anaconda3/2024.6
source activate thesis_ktn

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p landscape_class_tests logs

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=================="

python landscape_class_tests.py \
  --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
  --out-dir     landscape_class_tests

echo "End: $(date)"
