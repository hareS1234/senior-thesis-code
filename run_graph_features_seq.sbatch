#!/bin/bash
#SBATCH --job-name=gf_%x
#SBATCH --output=logs/gf_%x_%j.out
#SBATCH --error=logs/gf_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zs5958@princeton.edu
#SBATCH --nodes=1

# Usage: sbatch --job-name=SEQ run_graph_features_seq.sbatch SEQ
# Each job writes its own CSV; merge them afterward.

SEQ=${1:?Usage: sbatch --job-name=SEQ run_graph_features_seq.sbatch SEQ}

module purge
module load anaconda3/2024.6
source activate thesis_ktn

CODEDIR=/scratch/gpfs/JERELLE/harry/thesis_code
cd $CODEDIR || exit 1
mkdir -p logs graph_features_parts

echo "==== Job info ===="
echo "JobID: $SLURM_JOB_ID"
echo "Sequence: $SEQ"
echo "Start: $(date)"
echo "=================="

python graph_features.py \
  --T 300 \
  --out graph_features_parts/gf_${SEQ}.csv \
  --lite \
  --save-every 1 \
  --sequences "$SEQ"

echo "End: $(date)"
