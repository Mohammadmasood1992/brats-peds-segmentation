#!/usr/bin/env bash
# Run SynthStrip on the t1n of every subject in one or more split folders.
# Writes <case>-t1n_mask.nii.gz and <case>-t1n_brain.nii.gz next to each t1n; subjects that already
# have a mask are skipped.
# Usage: scripts/skullstrip_dataset.sh data/BraTS-PEDs-v1/Training data/BraTS-PEDs-v1/Validation [--cpu]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNTHSTRIP="$SCRIPT_DIR/../docker/synthstrip/run_synthstrip.sh"
dirs=(); extra=()
for arg in "$@"; do
  if [ -d "$arg" ]; then dirs+=("$arg"); else extra+=("$arg"); fi
done
[ ${#dirs[@]} -eq 0 ] && { echo "Usage: $0 SPLIT_DIR [SPLIT_DIR ...] [--cpu]"; exit 1; }

for split_dir in "${dirs[@]}"; do
  for case_dir in "$split_dir"/*/; do
    case_id="$(basename "$case_dir")"
    if [ -f "$case_dir/$case_id-t1n_mask.nii.gz" ]; then
      continue
    fi
    "$SYNTHSTRIP" "${extra[@]}" -in "$case_dir/$case_id-t1n.nii.gz" -out "$case_dir"
  done
done
echo "Brain masks are next to each subject's t1n."
