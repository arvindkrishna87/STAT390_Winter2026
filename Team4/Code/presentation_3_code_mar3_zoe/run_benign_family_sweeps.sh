#!/usr/bin/env bash
set -euo pipefail

BENIGN_JSON="benign_families.json"

for IDX in 0 1 2 3; do
  echo "=== Generating split for benign family idx=${IDX} ==="
  python3 make_splits.py \
    --save_dir "data_splits/family_${IDX}" \
    --exclude_from_test "46" \
    --benign_families_json "${BENIGN_JSON}" \
    --test_benign_family_idx "${IDX}"

  echo "=== Training/evaluating for benign family idx=${IDX} ==="
  python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits "data_splits/family_${IDX}/data_splits.npz"
done