#!/usr/bin/env bash
# Build the PDM metric cache used by eval_ltf.sh.
#
#   bash cache_metric_ltf.sh          # navtest -> exp/metric_cache_navtest
#   env overrides: SPLIT (e.g. navtest_smoke), METRIC_CACHE (output dir), WORKERS (default 48),
#                  PROCESS_POOL (default true)
#
# This is not the training cache (cache_ltf.sh): it holds the ground-truth simulation assets the
# PDM scorer reads. One metric cache serves both arms, and it needs no route.
#
# Build it with this devkit: metric caches written by other navsim versions are not compatible.
# The metric cache is not tracked in git: run this once before any eval.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

SPLIT="${SPLIT:-navtest}"
CACHE="${METRIC_CACHE:-$NAVSIM_DEVKIT_ROOT/exp/metric_cache_${SPLIT}}"
WORKERS="${WORKERS:-48}"

# Use a process pool: the caching code is pure-Python (nuplan/shapely) and holds the GIL, so a
# thread pool runs at about one core. PROCESS_POOL=false falls back to threads.
PROCESS_POOL="${PROCESS_POOL:-true}"

# Not resumable: default_metric_caching.yaml sets force_feature_computation: True, so a re-run
# recomputes every scenario. The metadata csv that eval reads is written at the end.
echo "=== caching METRIC cache: split=$SPLIT -> $CACHE ==="
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=$SPLIT \
    metric_cache_path="$CACHE" \
    worker=single_machine_thread_pool \
    worker.use_process_pool=$PROCESS_POOL \
    worker.max_workers=$WORKERS
