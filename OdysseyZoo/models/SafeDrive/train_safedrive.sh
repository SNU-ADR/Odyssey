#!/bin/bash
# =============================================================================
#  SafeDrive - Training, phase 1 -> 2 -> 3
#
#    bash train_safedrive.sh paper        the paper's model: camera + LiDAR, vehicles (upstream)
#    bash train_safedrive.sh camonly      camera-only, vehicles
#    bash train_safedrive.sh baseline     camera-only, pedestrians too, driving command in (released)
#    bash train_safedrive.sh sdroute      baseline + SD route, driving command dropped    (released)
#
#    phase 1 : perception pretraining, no planning head          90 epochs  (baseline and sdroute share it)
#    phase 2 : perception frozen*, planning + safety heads only   5 epochs
#    phase 3 : full end-to-end fine-tune (the released model)    10 epochs
#    * paper: upstream's phase-2 config freezes only the two BEV-segmentation heads
#
#  Released (not in git): the phase-3 checkpoints of baseline and sdroute, as
#  ckpts/safedrive_<variant>.ckpt, at https://huggingface.co/ADRLAB/odyssey-models. The paper's
#  checkpoints are linked from README.md. This script trains all three phases. Each phase hands its
#  last.ckpt to the next as weights, not as a resume: freezing changes the optimizer's param groups,
#  so optimizer state cannot cross a phase.
#
#  Phase 1 never reads the route, so baseline and sdroute share one phase-1 run, as the released
#  pair did: whichever runs first trains it, the other reuses it. Run the two one after the other
#  (or start the second once phase 1 is done), not both at once.
#
#  Re-running continues: a finished phase leaves a DONE file next to its checkpoints and is
#  skipped. An interrupted phase is never retrained over its own output: continue it with
#  RESUME_P<n>=<its last.ckpt>, or move its directory away to start that phase over.
#
#  Needs, once: bash cache_safedrive.sh (feature cache, train metric cache, SD route).
#
#  The recipe is the one each variant's checkpoint was trained with, bf16-mixed throughout:
#    paper                       global batch  64 (upstream: 2 GPUs x 32), 1330 steps per epoch
#    camonly, baseline, sdroute  global batch 128 (4 GPUs x 32),            665 steps per epoch
#  (steps read off the phase-3 checkpoints; camonly has none and takes the camera-only recipe).
#  Phases 1 and 2 use the same global batch.
# =============================================================================
set -e

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$BASE"   # the python entrypoint below is cwd-relative
DATA_ROOT=${DATA_ROOT:-$BASE/dataset}   # env wins


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
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}   # env wins. The phase 2/3 Ray rollout workers (8 per rank)
                                               # inherit it: 4 GPUs x 8 x 8 = 256 threads. On a host with
                                               # fewer cores, OMP_NUM_THREADS=1 is much faster.


# ---- configuration ----------------------------------------------------------
# ARM: the variant, paper | camonly | baseline | sdroute
ARM=${1:-}
# P1_RUN names the phase-1 run: baseline and sdroute share one (phase 1 has no route, so the
# SD-route variant has no phase-1 config of its own).
case "$ARM" in
    paper)    P1=SafeDrive_Phase1_Perception
              P2=SafeDrive_Phase2_Planner_FreezePerception
              P3=SafeDrive_Phase3_Planner_FullTrain
              P1_RUN=paper;       GLOBAL_BATCH=64;  DEFAULT_DEVICES=2 ;;
    camonly)  P1=SafeDrive_Phase1_Perception_CamOnly
              P2=SafeDrive_Phase2_Planner_FreezePerception_CamOnly
              P3=SafeDrive_Phase3_Planner_FullTrain_CamOnly
              P1_RUN=camonly;     GLOBAL_BATCH=128; DEFAULT_DEVICES=4 ;;
    baseline) P1=SafeDrive_Phase1_Perception_CamOnly_Ped
              P2=SafeDrive_Phase2_Planner_FreezePerception_CamOnly_Ped
              P3=safedrive_baseline_agent
              P1_RUN=camonly_ped; GLOBAL_BATCH=128; DEFAULT_DEVICES=4 ;;
    sdroute)  P1=SafeDrive_Phase1_Perception_CamOnly_Ped
              P2=SafeDrive_Phase2_Planner_FreezePerception_CamOnly_Ped_SDRoute
              P3=safedrive_sdroute_agent
              P1_RUN=camonly_ped; GLOBAL_BATCH=128; DEFAULT_DEVICES=4 ;;
    *) echo "usage: $0 {paper|camonly|baseline|sdroute}" >&2; exit 2 ;;
esac
EXP_P1=safedrive_$P1_RUN/phase1_perception
EXP_P2=safedrive_$ARM/phase2_freeze
EXP_P3=safedrive_$ARM/phase3_e2e

SPLIT=navtrain
DEVICES=${SAFEDRIVE_DEVICES:-$DEFAULT_DEVICES}  # GPUs used (of CUDA_VISIBLE_DEVICES); the per-GPU batch follows
if ! [[ "$DEVICES" =~ ^[1-9][0-9]*$ ]] || (( GLOBAL_BATCH % DEVICES != 0 )); then
    echo "ERROR: SAFEDRIVE_DEVICES=$DEVICES must divide the global batch $GLOBAL_BATCH" >&2; exit 2
fi
BATCH=$((GLOBAL_BATCH / DEVICES))
EPOCHS_P1=90
EPOCHS_P2=5
EPOCHS_P3=10    # the phase-3 lr follows the config's 60-epoch warmup-cosine, not max_epochs

SAVE_BEFORE_VAL=${SAVE_BEFORE_VAL:-True}  # also write checkpoints/pre_validation.ckpt as validation starts (one
                                # rolling 1.1 GB file), so a crash in validation does not lose the epoch
VAL_EVERY=${VAL_EVERY:-1}       # validate every N epochs. With N>1 Lightning also checkpoints only on
                                # those epochs (save_top_k=-1 keeps every one, ~1.1 GB each), so a crash
                                # loses up to N epochs; N=5 keeps phase 1 at ~18 checkpoints.
PREFETCH=${PREFETCH:-2}         # batches each worker keeps ready. Host RAM held by the loaders is
                                # ~ #GPUs x NUM_WORKERS x PREFETCH x per-GPU batch x ~30 MB (decoded
                                # targets); validation spawns the same set again on top.
NUM_WORKERS=${NUM_WORKERS:-24}  # dataloader workers per GPU. A path-only cache (cache_safedrive.sh
                                # CACHE_IMAGES=false) decodes 9 JPEGs per sample in the workers,
                                # ~0.35 s CPU per sample; 4 is enough only when images are cached.

CACHE_PATH=${CACHE_PATH:-$BASE/exp/safedrive_train_cache}             # written by cache_safedrive.sh
METRIC_CACHE=${METRIC_CACHE:-$BASE/exp/train_metric_cache_navtrain}   # written by cache_safedrive.sh

# Full resume (optimizer, scheduler, epoch counter) of an interrupted phase: point RESUME_P<n> at
# that phase's last.ckpt. Different from agent.checkpoint_path below, which loads weights only.
RESUME_P1=${RESUME_P1:-}
RESUME_P2=${RESUME_P2:-}
RESUME_P3=${RESUME_P3:-}
# ckpt_path is hydra-quoted: "epoch=9-step=6650.ckpt" does not parse bare.
resume_args() { [ -n "$1" ] && echo "+resume_from=True +ckpt_path='$1'"; }

COMMON=(
    train_test_split=$SPLIT
    split=$SPLIT
    cache_path=$CACHE_PATH
    use_cache_without_dataset=True
    force_cache_computation=False
    dataloader.params.batch_size=$BATCH
    dataloader.params.num_workers=$NUM_WORKERS
    dataloader.params.prefetch_factor=$PREFETCH
    +trainer.params.devices=$DEVICES
    ++trainer.params.precision=bf16-mixed
    ++trainer.params.check_val_every_n_epoch=$VAL_EVERY
    +save_before_validation=$SAVE_BEFORE_VAL
    +ddp_find_unused_parameters=True
    +second_lidar=True              # selects the custom collate (camera/lidar paths, token lists)
)

# rollout ground truth, phases 2 and 3 only (phase 1 has safety scoring off)
SAFETY_GT=( ++agent.config.safety_metric_cache_path=$METRIC_CACHE )

[ -d "$CACHE_PATH" ] || { echo "ERROR: no feature cache at $CACHE_PATH -- run: bash cache_safedrive.sh" >&2; exit 1; }

ckpt_dir() { echo "$NAVSIM_EXP_ROOT/$1/lightning_logs/checkpoints"; }

# run_phase <n> <agent config> <experiment name> <epochs> [extra hydra args...]
run_phase() {
    local n=$1 config=$2 exp=$3 epochs=$4; shift 4
    local dir resume_var resume
    dir=$(ckpt_dir "$exp"); resume_var=RESUME_P$n; resume=${!resume_var}
    if [ -e "$dir/DONE" ]; then
        echo "=== phase $n: done, reusing $dir/last.ckpt ==="; return
    fi
    if [ -z "$resume" ] && [ -e "$dir/last.ckpt" ]; then
        echo "ERROR: phase $n was interrupted ($dir/last.ckpt, no DONE)." >&2
        echo "       Continue it: RESUME_P$n=$dir/last.ckpt bash $0 $ARM" >&2
        echo "       or move $NAVSIM_EXP_ROOT/$exp away to start phase $n over." >&2
        exit 1
    fi
    echo "=== phase $n: $config -> $NAVSIM_EXP_ROOT/$exp ($DEVICES GPUs x batch $BATCH) ==="
    python navsim/planning/script/run_training.py \
        agent=$config \
        experiment_name=$exp \
        trainer.params.max_epochs=$epochs \
        "${COMMON[@]}" "$@" $(resume_args "$resume")
    [ -s "$dir/last.ckpt" ] || { echo "ERROR: phase $n wrote no $dir/last.ckpt" >&2; exit 1; }
    touch "$dir/DONE"
}


# ---- phase 1 : perception pretraining ---------------------------------------
# Detection and BEV segmentation only: no_planning=True and scene_level_safety=False,
# so there is no planning head and no rollout ground truth to compute.
run_phase 1 "$P1" "$EXP_P1" "$EPOCHS_P1"

# ---- phase 2 : frozen perception --------------------------------------------
run_phase 2 "$P2" "$EXP_P2" "$EPOCHS_P2" \
    "agent.checkpoint_path='$(ckpt_dir "$EXP_P1")/last.ckpt'" "${SAFETY_GT[@]}"

# ---- phase 3 : end-to-end fine-tune -----------------------------------------
run_phase 3 "$P3" "$EXP_P3" "$EPOCHS_P3" \
    "agent.checkpoint_path='$(ckpt_dir "$EXP_P2")/last.ckpt'" "${SAFETY_GT[@]}"

echo "done -> $(ckpt_dir "$EXP_P3")/last.ckpt   (the released phase-3 checkpoints are epoch 9 of this phase)"
