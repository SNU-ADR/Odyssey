#!/bin/bash
# =============================================================================
#  SafeDrive - Evaluation on navtest
#
#    bash eval_safedrive.sh baseline [ckpt]   ckpt defaults to ckpts/safedrive_baseline.ckpt
#    bash eval_safedrive.sh sdroute  [ckpt]   ckpt defaults to ckpts/safedrive_sdroute.ckpt
#    bash eval_safedrive.sh paper    <ckpt>   the paper's camera + LiDAR model (see README.md)
#    bash eval_safedrive.sh camonly  <ckpt>   camera-only, vehicles
#  (the variants of train_safedrive.sh; baseline / sdroute weights:
#  https://huggingface.co/ADRLAB/odyssey-models)
#
#  Needs, once: bash cache_metric_safedrive.sh (navtest metric cache).
#
#    stage 1 : run_training.py       forward + weighted scoring -> trajectory pkl
#    stage 2 : run_evaluation_gpu.py PDM scoring of that pkl     -> result.csv
#
#  Stage 1 cannot be done by run_evaluation_gpu.py: the code that injects
#  scoring_test / *_test_weight into the model only exists in run_training.py,
#  so weights passed to the evaluation script are silently ignored.
# =============================================================================
set -e

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BASE"   # the python entrypoints below, and the stage-1 trajectory pkl, are cwd-relative
DATA_ROOT=${DATA_ROOT:-$BASE/dataset}   # env wins, e.g. when navtest lives under another root
                                        # than the training data


# ---- environment ------------------------------------------------------------
export PYTHONPATH=$BASE
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$DATA_ROOT/maps
export OPENSCENE_DATA_ROOT=$DATA_ROOT
export NAVSIM_EXP_ROOT=$BASE/exp
export NAVSIM_DEVKIT_ROOT=$BASE/navsim
export HYDRA_FULL_ERROR=1

# build-time settings for mmcv / spconv CUDA ops; point CUDA_HOME at your CUDA install
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}   # env wins
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=${CC:-gcc-11}                               # env wins
export CXX=${CXX:-g++-11}                             # env wins
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}


# ---- configuration ----------------------------------------------------------
# ARM: the variant, paper | camonly | baseline | sdroute
ARM=${1:-}
case "$ARM" in
    paper)            CONFIG=SafeDrive_Phase3_Planner_FullTrain ;;
    camonly)          CONFIG=SafeDrive_Phase3_Planner_FullTrain_CamOnly ;;
    baseline|sdroute) CONFIG=safedrive_${ARM}_agent
                      DEFAULT_CKPT=$BASE/ckpts/safedrive_${ARM}.ckpt ;;
    *) echo "usage: $0 {baseline|sdroute} [ckpt]  |  $0 {paper|camonly} <ckpt>" >&2; exit 2 ;;
esac
CKPT=${2:-${DEFAULT_CKPT:-}}    # must be trained with CONFIG
[ -n "$CKPT" ] || { echo "usage: $0 $ARM <ckpt>  (no $ARM checkpoint ships in ckpts/)" >&2; exit 2; }
SPLIT=navtest
BATCH=${BATCH:-16}              # GPU batch, stage 1
THREADS=${THREADS:-10}          # PDM workers, stage 2

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
METRIC_CACHE=${METRIC_CACHE:-$BASE/exp/metric_cache_navtest}
AGENT_NAME=${AGENT_NAME:-safedrive_eval_$ARM}   # trajectories -> exp/training/$AGENT_NAME
EXP=${EXP:-safedrive/eval_$AGENT_NAME}          # results      -> exp/$EXP


# ---- safety scoring weights: upstream's, used unchanged for every variant ---
WEIGHTS=(
    +imi_test_weight=0.3                        # imitation
    +NC_test_weight=16.0                        # no at-fault collision
    +DAC_test_weight=48.0                       # drivable area compliance
    +EP_test_weight=0.75                        # ego progress
    +TTC_test_weight=15.0                       # time to collision
    +PwNC_test_weight=5.0                       # pair-wise NC
    +TwDAC_test_weight=1.5                      # time-wise DAC
    +W_test_weight=1.0                          # weight of the log(EP, TTC) term
    +pdm_score_test_weight=1.0                  # aggregate PDM head
    +DDC_test_weight=1.0                        # driving direction compliance
    +TLC_test_weight=1.0                        # traffic light compliance
    +LK_test_weight=1.0                         # lane keeping
    +TwDAC_bbox_margin=[1.4,1.6]

    # scoring formula for EP / TTC:
    #   off : score = ... + W * log(EP_w * EP + TTC_w * TTC)
    #   on  : score = ... + EP_w * log(EP) + TTC_w * log(TTC)   <- the weights above
    #                                                              were searched against this form
    +no_EP_TC_sum_scoring=True
)


# ---- stage 1 : GPU forward + weighted scoring -> trajectory pkl -------------
# build_datasets() intersects the scene filter with val_logs, and the default
# split lists trainval logs only, so navtest needs its log list passed in.
VAL_LOGS=$(python3 -c "
import os
logs = sorted(f[:-4] for f in os.listdir('$DATA_ROOT/navsim_logs/test') if f.endswith('.pkl'))
print('[' + ','.join(logs) + ']')")

mkdir -p $NAVSIM_EXP_ROOT/training/$AGENT_NAME   # stage 1 writes the trajectory pkl here

# validate_only breaks this path (token missing from features -> samples skipped),
# debug=false keeps the full token list, precision=32 because bf16 dies in the
# numpy conversion, cache_path=null because navtest runs on the fly.
# ckpt_path is hydra-quoted: lightning names checkpoints "epoch=9-step=6650.ckpt" and the bare
# '=' in the value makes hydra's override parser bail with "mismatched input '=' expecting <EOF>".
python navsim/planning/script/run_training.py \
    agent=$CONFIG \
    train_test_split=$SPLIT \
    split=$SPLIT \
    experiment_name=safedrive/eval_fwd_$AGENT_NAME \
    cache_path=null \
    use_cache_without_dataset=False \
    force_cache_computation=False \
    dataloader.params.batch_size=$BATCH \
    dataloader.params.num_workers=4 \
    trainer.params.max_epochs=1 \
    ++trainer.params.precision=32 \
    ++train_logs=$VAL_LOGS \
    ++val_logs=$VAL_LOGS \
    +debug=false \
    +second_lidar=True \
    +ddp_find_unused_parameters=True \
    "+ckpt_path='$CKPT'" \
    +agent_name=$AGENT_NAME \
    +test_traj_save=True \
    +test_save_name=traj \
    +metric_cache_path=$METRIC_CACHE \
    +scoring_test=True \
    +pair_NC_scoring=true \
    +twdac_scoring=true \
    +TwDAC_bevseg_pred=true \
    "${WEIGHTS[@]}"


# ---- stage 2 : PDM scoring -> result.csv ------------------------------------
# The saved filename is built from the weights plus one suffix per enabled
# option, so take the newest match instead of reconstructing it.
TRAJ_PKL=$(ls -t $NAVSIM_EXP_ROOT/training/$AGENT_NAME/traj_*.pkl | head -1)
echo "scoring $TRAJ_PKL"

python navsim/planning/script/run_evaluation_gpu.py \
    agent=$CONFIG \
    experiment_name=$EXP \
    train_test_split=$SPLIT \
    metric_cache_path=$METRIC_CACHE \
    worker.threads_per_node=$THREADS \
    pred_traj_path=$TRAJ_PKL \
    +second_lidar=True \
    +save_csv=result.csv

echo "done -> $(ls -t $NAVSIM_EXP_ROOT/$EXP/*.csv | head -1)"
