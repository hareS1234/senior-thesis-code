#!/bin/bash
# Merge per-sequence CSVs into one combined file.
# Run after all graph feature jobs have finished.

OUTFILE="graph_features_coarse_T300K_lite.csv"
PARTS_DIR="graph_features_parts"

# Take header from first file
FIRST=$(ls "$PARTS_DIR"/gf_*.csv 2>/dev/null | head -1)
if [[ -z "$FIRST" ]]; then
  echo "No CSV files found in $PARTS_DIR"
  exit 1
fi

head -1 "$FIRST" > "$OUTFILE"

# Append data rows (skip header) from all files
for f in "$PARTS_DIR"/gf_*.csv; do
  tail -n +2 "$f" >> "$OUTFILE"
done

NROWS=$(( $(wc -l < "$OUTFILE") - 1 ))
echo "Merged $NROWS networks into $OUTFILE"
