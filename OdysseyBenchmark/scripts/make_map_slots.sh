#!/usr/bin/env bash
# Copy one nuPlan maps v1.0 directory into one slot per concurrent run, and print the MAP_SLOTS list
# for run_evaluation_*.sh: every GPU's first run, then every GPU's second run, and so on (the order
# `--runs-per-gpu` assigns them). Slots that already exist are kept. Each copy is about 1.4 GB.
# Usage: MAP_SLOTS=$(bash OdysseyBenchmark/scripts/make_map_slots.sh /abs/path/nuplan-maps /abs/path/maps 0,1,2,3 2)
set -euo pipefail
SRC=${1:?usage: make_map_slots.sh <nuplan-maps dir> <slots dir> [GPUS=0,1,2,3] [RUNS_PER_GPU=1]}
DEST=${2:?usage: make_map_slots.sh <nuplan-maps dir> <slots dir> [GPUS=0,1,2,3] [RUNS_PER_GPU=1]}
GPUS=${3:-0,1,2,3}
RUNS=${4:-1}
[ -f "$SRC/nuplan-maps-v1.0.json" ] || { echo "$SRC: nuplan-maps-v1.0.json missing" >&2; exit 1; }
mkdir -p "$DEST"
IFS=, read -ra gpus <<< "$GPUS"
slots=()
for ((k = 1; k <= RUNS; k++)); do
  for g in "${gpus[@]}"; do
    name="final_gpu$g"                      # first run: NUPLAN_MAPS_ROOT can name it, the GPU index is its suffix
    [ "$k" -gt 1 ] && name="final_gpu${g}_run$k"
    [ -d "$DEST/$name" ] || { echo "copying $SRC -> $DEST/$name" >&2; cp -r "$SRC" "$DEST/$name"; }
    slots+=("$DEST/$name")
  done
done
(IFS=,; echo "${slots[*]}")
