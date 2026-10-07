#!/usr/bin/env bash
# Build the DrivoR dataset cache. One cache serves both arms (baseline / sdroute).
#
#   bash cache_drivor.sh          # navtrain -> exp/cache_drivor_navtrain
#   bash cache_drivor.sh debug    # navtrain_debug -> exp/cache_drivor_navtrain_debug
#
# Environment: DRIVOR_CKPT (see below), WORKERS_OVERRIDE (worker count for the full split,
# default 32), PROCESS_POOL=false (thread pool instead of processes).
#
# Cached with the sdroute agent regardless of which arm you train: its target builder adds
# `route_centerline`, which the baseline model ignores. A baseline-only cache would lack it, and
# training the sdroute arm on it would fail (DrivoRModel raises on a missing route).
#
# DrivoR-specific:
#  * Runs in the DrivoR env (its own torch stack), not the shared navsim env: set PY to that env's
#    python, or activate it (PY defaults to `python` on PATH).
#  * checkpoint_path '' is DrivoRAgent's training mode, which starts Ray and loads the metric cache;
#    run_dataset_caching builds one agent per worker, so that would call ray.init twice. Passing a
#    checkpoint_path (CKPT below) skips it. Caching never calls initialize(), so any existing
#    file works: it is never loaded.
#
# The cache is gitignored (multi-GB), so a fresh clone must run this before any training.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

MODE="${1:-full}"
if [ "$MODE" = "debug" ]; then
    SPLIT=navtrain_debug; CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_drivor_navtrain_debug"; WORKERS=4
else
    # Each process worker holds its own log/map objects, so the count is bounded by RAM, not cores.
    SPLIT=navtrain;       CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_drivor_navtrain";       WORKERS=${WORKERS_OVERRIDE:-32}
fi

CKPT="${DRIVOR_CKPT:-$NAVSIM_DEVKIT_ROOT/ckpts/drivor_baseline.ckpt}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: DrivoR checkpoint not found at $CKPT" >&2
    echo "Caching needs SOME checkpoint_path to skip the agent's Ray/metric-cache init." >&2
    echo "Any existing file works (it is never loaded): set DRIVOR_CKPT, e.g. to a ckpt in ckpts/." >&2
    exit 1
fi
# Every caching worker builds a full DrivoRAgent, DINOv2 included.
require_dinov2_weights drivor_sdroute_agent

# Use processes: `worker=single_machine_thread_pool` defaults to a ThreadPoolExecutor, but the
# builders are pure-Python shapely/nuplan work that holds the GIL, so threads do not scale.
# Set PROCESS_POOL=false to fall back to threads if the pool trips on something unpicklable.
echo "=== caching DrivoR: split=$SPLIT -> $CACHE ==="
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
    agent=drivor_sdroute_agent \
    experiment_name="cache_drivor_${SPLIT}" \
    train_test_split=$SPLIT \
    cache_path="$CACHE" \
    agent.checkpoint_path="$CKPT" \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    worker.use_process_pool=${PROCESS_POOL:-true} \
    worker.max_workers=$WORKERS
