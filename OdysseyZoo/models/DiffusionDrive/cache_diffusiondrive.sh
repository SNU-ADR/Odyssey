#!/usr/bin/env bash
# Build the DiffusionDrive dataset cache. One cache serves both arms (baseline / sdroute).
#
# Usage:
#   bash cache_diffusiondrive.sh          # navtrain -> exp/cache_diffusiondrive_navtrain
#   bash cache_diffusiondrive.sh debug    # navtrain_debug -> exp/cache_diffusiondrive_navtrain_debug
#   env overrides: WORKERS_OVERRIDE (worker count), PROCESS_POOL (default true; false = threads)
#
# The cache is built with the sdroute agent, whose targets add route_centerline(_mask); the baseline
# ignores them. A cache built with the baseline agent lacks the route, and the sdroute model refuses it.
#
# The cache is gitignored (multi-GB), so a fresh clone must run this before any training.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh

MODE="${1:-full}"
if [ "$MODE" = "debug" ]; then
    SPLIT=navtrain_debug; CACHE="$NAVSIM_EXP_ROOT/cache_diffusiondrive_navtrain_debug"; WORKERS=16
else
    SPLIT=navtrain;       CACHE="$NAVSIM_EXP_ROOT/cache_diffusiondrive_navtrain";       WORKERS=32
fi

# Caching instantiates the full agent (backbone + trajectory head). The trajectory anchors come
# from the vendored assets/kmeans_navsim_traj_20.npy (TransfuserConfig.plan_anchor_path default);
# the ResNet-34 ImageNet weights come from timm's Hugging Face cache (offline and uncached,
# TransfuserBackbone raises with the hub id to pre-fetch). This script takes no Hydra overrides; to
# use local weights, run run_dataset_caching.py directly with agent.config.bkb_path=<file>.

echo "=== caching DiffusionDrive: split=$SPLIT -> $CACHE ==="
# A directory can be large and still be incomplete: a killed run leaves a partial cache that a
# plain `-d` check accepts. `.complete` is written only after a zero exit. Rerunning rebuilds the
# whole cache (force_cache_computation=true); it does not resume.
rm -f "$CACHE/.complete"
$PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
    agent=diffusiondrive_sdroute_agent \
    experiment_name="cache_diffusiondrive_${SPLIT}" \
    train_test_split=$SPLIT \
    cache_path="$CACHE" \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    worker.use_process_pool=${PROCESS_POOL:-true} \
    worker.max_workers=${WORKERS_OVERRIDE:-$WORKERS}

mkdir -p "$CACHE"
date -Iseconds > "$CACHE/.complete"
echo "=== DiffusionDrive cache complete: $CACHE/.complete ==="
