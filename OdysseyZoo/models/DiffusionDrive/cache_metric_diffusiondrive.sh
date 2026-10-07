#!/usr/bin/env bash
# Build the navtest metric cache used by eval_diffusiondrive.sh.
# Mirrors ../navsim/cache_metric_ltf.sh.
#
#   bash cache_metric_diffusiondrive.sh          # navtest -> exp/metric_cache_navtest
#   env overrides: SPLIT, METRIC_CACHE (output dir), WORKERS, PROCESS_POOL
#
# This is not the dataset cache (cache_diffusiondrive.sh):
#   cache_diffusiondrive.sh         features/targets for training -> exp/cache_diffusiondrive_navtrain
#   cache_metric_diffusiondrive.sh  PDM scoring assets for eval   -> exp/metric_cache_navtest
#
# One metric cache serves both arms, camera-only or with LiDAR: it holds the scenario's ground-truth
# simulation assets (tracked objects, drivable area, centerline, human trajectory), which do not
# depend on the model or the route. It can also be shared with LTF's metric_cache_navtest: point
# METRIC_CACHE at the same dir.
#
# The metric cache is gitignored (multi-GB), so a fresh clone must run this before any eval.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh

SPLIT="${SPLIT:-navtest}"
CACHE="${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}}"
WORKERS="${WORKERS:-48}"

# Use a process pool: single_machine_thread_pool defaults to threads (use_process_pool=False), and
# metric caching is pure-Python, GIL-bound code, so threads are several times slower.
# PROCESS_POOL=false falls back to threads.
PROCESS_POOL="${PROCESS_POOL:-true}"

# Re-running resumes: MetricCacheProcessor skips scenarios whose metric_cache.pkl already exists,
# so a rerun after an interrupt fills the gaps and writes the metadata csv that eval reads.
echo "=== caching METRIC cache: split=$SPLIT -> $CACHE ==="
# The metric-caching config takes `cache.cache_path=`, not `metric_cache_path=` (the run_pdm_score
# key; see config/metric_caching/default_metric_caching.yaml). MetricCacheLoader reads
# $CACHE/metadata/*.csv, written after the last scenario.
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=$SPLIT \
    cache.cache_path="$CACHE" \
    worker=single_machine_thread_pool \
    worker.use_process_pool=$PROCESS_POOL \
    worker.max_workers=$WORKERS
