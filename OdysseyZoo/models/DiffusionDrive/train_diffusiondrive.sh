#!/usr/bin/env bash
# Train DiffusionDrive on one of its two arms.
#
#   bash train_diffusiondrive.sh baseline        driving_command in, no route
#   bash train_diffusiondrive.sh sdroute         SD route in, driving_command dropped
#                                                (see diffusiondrive_sdroute_agent.yaml)
#   bash train_diffusiondrive.sh <arm> smoke     fast_dev_run (1 batch, 1 device); run this first
#   Trailing hydra key=value args are forwarded to run_training.py.
#   env overrides: BATCH_SIZE, LR, MAX_EPOCHS, ACCUM, NUM_WORKERS (see the recipe below)
#
# One script for both arms, so lr, batch size, epochs, devices and seed stay identical and only
# `agent=` changes: a baseline-vs-sdroute delta then measures the route instead of the command.
# Do not tune hyperparameters per arm.
#
# Requires the cache: run `bash cache_diffusiondrive.sh` (or `debug` for smoke) once. Both arms
# share it; the baseline ignores the cached route.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh

ARM="${1:-}"
case "$ARM" in
    baseline)                AGENT=diffusiondrive_baseline_agent ;;
    sdroute)                 AGENT=diffusiondrive_sdroute_agent ;;
    *)
        echo "usage: $0 {baseline|sdroute} [smoke|full] [hydra.override=value ...]" >&2
        exit 2 ;;
esac
shift || true

# Remaining args: an optional `smoke`/`full` mode plus trailing Hydra overrides (anything
# containing `=`, e.g. `seed=1` for a repeat run). Use overrides only for settings shared by both arms.
MODE="full"
PASS_HYDRA=()
for arg in "$@"; do
    case "$arg" in
        smoke|full) MODE="$arg" ;;
        *=*)        PASS_HYDRA+=("$arg") ;;
        *) echo "WARNING: ignoring unrecognized argument '$arg' (expected smoke|full or hydra key=value)" >&2 ;;
    esac
done

# Assets: see cache_diffusiondrive.sh (vendored anchors; ResNet-34 weights from timm's HF cache).

if [ "$MODE" = "smoke" ]; then
    CACHE="$NAVSIM_EXP_ROOT/cache_diffusiondrive_navtrain_debug"
    # .complete, not just the directory: a killed caching run leaves a partial cache behind.
    [ -f "$CACHE/.complete" ] || { echo "ERROR: no complete cache at $CACHE -- run: bash cache_diffusiondrive.sh debug" >&2; exit 1; }
    exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
        agent=$AGENT \
        experiment_name="smoke_diffusiondrive_${ARM}" \
        train_test_split=navtrain_debug \
        cache_path="$CACHE" \
        use_cache_without_dataset=true \
        force_cache_computation=false \
        dataloader.params.batch_size=2 \
        dataloader.params.num_workers=2 \
        trainer.params.max_epochs=1 \
        trainer.params.fast_dev_run=true \
        ~trainer.params.strategy \
        +trainer.params.devices=1 \
        "${PASS_HYDRA[@]}"
fi

CACHE="$NAVSIM_EXP_ROOT/cache_diffusiondrive_navtrain"
[ -f "$CACHE/.complete" ] || { echo "ERROR: no complete cache at $CACHE -- run: bash cache_diffusiondrive.sh" >&2; exit 1; }

# No resume: run_training.py calls trainer.fit() without ckpt_path, and agent.checkpoint_path
# loads weights only (TransfuserAgent.init_from_pretrained), so an interrupted run restarts at epoch 0.

# --- Upstream recipe: global batch 512, lr 6e-4, 100 epochs ------------------------------------
# Global batch = BATCH_SIZE * NGPU * ACCUM; NGPU comes from CUDA_VISIBLE_DEVICES (unset = 1).
# The default BATCH_SIZE=256 reaches 512 on 2 GPUs; on 1 GPU set ACCUM=2, and on OOM lower
# BATCH_SIZE and raise ACCUM. For another global batch, raise ACCUM back to 512 or scale LR by
# sqrt(global/512), e.g. on 2 GPUs:
#   BATCH_SIZE=128 LR=$(python3 -c 'import math;print(6e-4*math.sqrt(128*2/512))') bash train_diffusiondrive.sh sdroute
_CVD="${CUDA_VISIBLE_DEVICES:-0}"
NGPU=$(echo "$_CVD" | tr ',' '\n' | grep -c .); [ "$NGPU" -lt 1 ] && NGPU=1
BATCH_SIZE="${BATCH_SIZE:-256}"      # per-GPU; global = BATCH_SIZE * NGPU * ACCUM
LR="${LR:-6e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
ACCUM="${ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
echo "[train] arm=$ARM global_batch=$((BATCH_SIZE*NGPU*ACCUM)) (bs=$BATCH_SIZE x ngpu=$NGPU x accum=$ACCUM) lr=$LR epochs=$MAX_EPOCHS"

exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
    agent=$AGENT \
    experiment_name="diffusiondrive_${ARM}_navtrain" \
    train_test_split=navtrain \
    cache_path="$CACHE" \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    agent.lr=$LR \
    dataloader.params.batch_size=$BATCH_SIZE \
    dataloader.params.num_workers=$NUM_WORKERS \
    trainer.params.max_epochs=$MAX_EPOCHS \
    trainer.params.accumulate_grad_batches=$ACCUM \
    trainer.params.gradient_clip_val=0.0 \
    +trainer.params.devices=$NGPU \
    trainer.params.strategy=ddp \
    "${PASS_HYDRA[@]}"
