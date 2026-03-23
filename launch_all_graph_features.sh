#!/bin/bash
# Submit one graph-features job per sequence.
# Run from the thesis_code directory on Tiger3.
# After all jobs finish, run merge_graph_features.sh to combine CSVs.

SEQUENCES=(
  aaaaaa aaggaa eeeeee ffggff flgglf gaiigl gailss gggggg ggvvia gyviik
  keggek kkkkkk klvffa kyggyk lfggfl llggll mvggvv nfgail regger rrrrrr
  ryggyr snqnnf ssqvtq sstnvg svsssy vqivyk vvvvvv ykggky yrggry yyggyy
)

mkdir -p logs graph_features_parts

for SEQ in "${SEQUENCES[@]}"; do
  # Skip if this sequence's CSV already has data
  OUTFILE="graph_features_parts/gf_${SEQ}.csv"
  if [[ -f "$OUTFILE" ]] && [[ $(wc -l < "$OUTFILE") -gt 1 ]]; then
    echo "Skipping $SEQ — $OUTFILE already has results"
    continue
  fi
  echo "Submitting $SEQ"
  sbatch --job-name="$SEQ" run_graph_features_seq.sbatch "$SEQ"
done

echo "Done. Monitor with: squeue -u \$USER"
