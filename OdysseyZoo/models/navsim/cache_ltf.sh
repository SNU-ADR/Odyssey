#!/usr/bin/env bash
# Build the LTF training cache (features + targets). One cache serves both arms.
#
# Usage:
#   bash cache_ltf.sh          # navtrain       -> exp/cache_ltf_navtrain
#   bash cache_ltf.sh debug    # navtrain_debug -> exp/cache_ltf_navtrain_debug (smoke runs)
#   env overrides: WORKERS_OVERRIDE (default 32), PROCESS_POOL (default true)
#
# It caches with ltf_sdroute_agent, whose target builder adds route_centerline; the baseline arm
# ignores that tensor, so switching arms never needs a re-cache. The cache is not tracked in git:
# run this once before training.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

MODE="${1:-full}"
if [ "$MODE" = "debug" ]; then
    SPLIT=navtrain_debug; CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_ltf_navtrain_debug"
else
    SPLIT=navtrain;       CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_ltf_navtrain"
fi

# Use a process pool: the builders are pure-Python (shapely/nuplan) and hold the GIL, so a thread
# pool runs at about one core. Each process holds its own map objects, so memory grows with
# WORKERS_OVERRIDE. PROCESS_POOL=false falls back to threads.
echo "=== caching LTF: split=$SPLIT -> $CACHE ==="
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
    agent=ltf_sdroute_agent \
    experiment_name="cache_ltf_${SPLIT}" \
    train_test_split=$SPLIT \
    cache_path="$CACHE" \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    worker.use_process_pool=${PROCESS_POOL:-true} \
    worker.max_workers=${WORKERS_OVERRIDE:-32}
