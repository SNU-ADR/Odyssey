#!/bin/bash
# =============================================================================
#  SafeDrive - navtest metric cache, what eval_safedrive.sh scores against
#
#    bash cache_metric_safedrive.sh        -> exp/metric_cache_navtest
#
#  One cache serves every variant: it holds the scenarios' ground truth, which does
#  not depend on the model being scored.
# =============================================================================
set -e

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BASE"   # the python entrypoint below is cwd-relative
DATA_ROOT=${DATA_ROOT:-$BASE/dataset}       # env wins


# ---- environment ------------------------------------------------------------
export PYTHONPATH=$BASE
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_ROOT/maps
export OPENSCENE_DATA_ROOT=$DATA_ROOT
export NAVSIM_EXP_ROOT=$BASE/exp
export NAVSIM_DEVKIT_ROOT=$BASE/navsim
export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# ---- configuration ----------------------------------------------------------
METRIC_WORKERS=${METRIC_WORKERS:-10}        # ray workers (~3.6 GB RSS each)
TEST_METRIC_CACHE=${TEST_METRIC_CACHE:-$BASE/exp/metric_cache_navtest}
# The logs live under the split's data_split, not its config name
# (train_test_split/navtest.yaml: data_split: test).
TEST_LOG_DIR=${TEST_LOG_DIR:-test}

[ -d "$DATA_ROOT/navsim_logs/$TEST_LOG_DIR" ] || {
    echo "ERROR: no $DATA_ROOT/navsim_logs/$TEST_LOG_DIR -- download navtest first." >&2; exit 1; }

python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path=$TEST_METRIC_CACHE \
    worker.threads_per_node=$METRIC_WORKERS

echo "done -> $TEST_METRIC_CACHE"
