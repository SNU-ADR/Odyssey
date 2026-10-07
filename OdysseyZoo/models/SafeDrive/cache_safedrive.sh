#!/bin/bash
# =============================================================================
#  SafeDrive - training caches, what train_safedrive.sh reads
#
#    step 1 : feature cache      navtrain sensor data -> tensors      exp/safedrive_train_cache
#    step 2 : train metric cache navtrain PDM caches  -> rollout GT   exp/train_metric_cache_navtrain
#    step 3 : SD route           step 1's cache       -> route_centerline, for the sdroute variant
#
#    bash cache_safedrive.sh              all three
#    STEPS="2 3" bash cache_safedrive.sh  e.g. once the feature cache exists
#
#  One cache serves every variant of train_safedrive.sh: it is built with the paper's camera +
#  LiDAR phase-3 config, whose targets include agent_token_ids (the rollout needs them) and whose
#  LiDAR entries the camera-only variants simply ignore. A cache built with a *_CamOnly config has
#  no LiDAR and serves only the camera-only variants. The navtest metric cache for evaluation is
#  cache_metric_safedrive.sh.
# =============================================================================
set -e

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BASE"   # the python entrypoints below are cwd-relative
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
CONFIG=SafeDrive_Phase3_Planner_FullTrain   # see the header: one cache for every variant
ROUTE_CONFIG=safedrive_sdroute_agent        # step 3 only turns the route on (the route has no knobs)
TRAIN_SPLIT=navtrain
STEPS=${STEPS:-1 2 3}
WORKERS=${WORKERS:-60}                      # ray workers for the feature cache
CACHE_IMAGES=${CACHE_IMAGES:-false}         # false: cache image paths, decode in the dataloader
                                            #        (~0.7 MB/sample, ~75 GB for navtrain; needs
                                            #        NUM_WORKERS >= 20 in train_safedrive.sh)
                                            # true : cache the uint8 images (~3.5 MB/sample,
                                            #        ~360 GB; training runs fine with 4 workers)
METRIC_WORKERS=${METRIC_WORKERS:-10}        # ray workers for the metric cache (~3.6 GB RSS each;
                                            #  10 took ~6 h for navtrain on 128 cores; use more)

FEATURE_CACHE=${FEATURE_CACHE:-$BASE/exp/safedrive_train_cache}
TRAIN_METRIC_CACHE=${TRAIN_METRIC_CACHE:-$BASE/exp/train_metric_cache_navtrain}

[ -d "$DATA_ROOT/navsim_logs/trainval" ] || {
    echo "ERROR: no $DATA_ROOT/navsim_logs/trainval -- navtrain needs the trainval logs and sensors." >&2; exit 1; }


# ---- step 1 : feature cache (navtrain) --------------------------------------
# Per-worker BLAS threads are pinned to 1: 60 workers x 8 OMP threads would oversubscribe the host.
if [[ " $STEPS " == *" 1 "* ]]; then
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python navsim/planning/script/run_dataset_caching.py \
    agent=$CONFIG \
    experiment_name=caching/safedrive \
    train_test_split=$TRAIN_SPLIT \
    cache_path=$FEATURE_CACHE \
    ++agent.config.cache_camera_images=$CACHE_IMAGES \
    worker.threads_per_node=$WORKERS
fi


# ---- step 2 : train metric cache (navtrain) ---------------------------------
# Phases 2 and 3 score their own trajectories with the PDM simulator during training, which
# reads these caches through agent.config.safety_metric_cache_path.
if [[ " $STEPS " == *" 2 "* ]]; then
python navsim/planning/script/run_train_metric_caching.py \
    train_test_split=$TRAIN_SPLIT \
    cache.cache_path=$TRAIN_METRIC_CACHE \
    worker.threads_per_node=$METRIC_WORKERS
fi


# ---- step 3 : SD route (navtrain) -------------------------------------------
# Step 1 caches with CONFIG, which has the route off, so its targets carry no route_centerline.
# This adds the route to step 1's target pickles (same filename, atomic replace; route-free configs
# ignore the extra keys). Tokens that already have it are skipped, so a re-run is safe.
if [[ " $STEPS " == *" 3 "* ]]; then
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python navsim/planning/script/run_sdroute_backfill.py \
    agent=$ROUTE_CONFIG \
    experiment_name=caching/safedrive_sdroute_backfill \
    train_test_split=$TRAIN_SPLIT \
    cache_path=$FEATURE_CACHE \
    worker.threads_per_node=$WORKERS \
    +apply=true
fi


echo "done"
echo "  feature cache      -> $FEATURE_CACHE"
echo "  train metric cache -> $TRAIN_METRIC_CACHE"
