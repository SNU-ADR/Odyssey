#!/usr/bin/env bash
# Train DrivoR, either arm.
#
#   bash train_drivor.sh baseline              upstream DrivoR: driving_command in, no route
#   bash train_drivor.sh sdroute               SD route in (24 route tokens, cross-attention in every
#                                              Block), driving_command dropped
#
#   bash train_drivor.sh <arm> smoke           fast_dev_run (1 batch, 1 device) -- run this first
#
# Environment: DRIVOR_DEVICES (GPUs, default 4), DRIVOR_LOADER_WORKERS (dataloader workers per
# rank, default 16), DRIVOR_RAY_WORKERS_PER_GPU (Ray workers for the scorer GT, default 8).
#
# Needs, once: bash cache_drivor.sh (dataset cache), bash cache_metric_drivor.sh train
# (train_metric_cache, the scorer head's GT) and the DINOv2 weights (smoke_env.sh says where).
#
# One script for both arms, on purpose: the arms are compared against each other, so every
# hyperparameter below is shared. Only `agent=` changes.
#
# Recipe (used for ckpts/drivor_sdroute.ckpt): AdamW, base lr 2e-4 at global batch 64, 25 epochs,
# 10% linear warmup then cosine to 0 (get_optimizers()), bf16-mixed, seed 2, 4 GPUs x 16.
# ckpts/drivor_baseline.ckpt is upstream's released NAVSIM-v1 model (trained with the upstream
# README's command), not a run of this script.
#
# Re-running resumes: run_training.py picks up the newest *.ckpt under
# exp/ke/<experiment_name>/*/lightning_logs/. Move that directory away to start over.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

ARM="${1:-}"
MODE="${2:-full}"
case "$ARM" in
    baseline)        AGENT=drivor_baseline_agent ;;
    sdroute)         AGENT=drivor_sdroute_agent ;;
    *)
        echo "usage: $0 {baseline|sdroute} [smoke]" >&2
        exit 2 ;;
esac

require_dinov2_weights "$AGENT"
# checkpoint_path '' is DrivoRAgent's training mode, which reads this fixed path for the scorer GT.
ls "$NAVSIM_EXP_ROOT"/train_metric_cache/metadata/*.csv >/dev/null 2>&1 || {
    echo "ERROR: no complete train metric cache at $NAVSIM_EXP_ROOT/train_metric_cache" >&2
    echo "       (metadata/*.csv is written last) -- run: bash cache_metric_drivor.sh train" >&2
    exit 1; }

# DrivoR takes its lr from lr_args (base_lr / base_batch_size), not agent.lr. The yamls' base_lr
# 5e-4 is upstream's default; the recipe overrides it here, next to the batch that scales it.
RECIPE=(
    agent.lr_args.base_lr=2e-4
    seed=2
    trainer.params.precision=bf16-mixed
)

if [ "$MODE" = "smoke" ]; then
    CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_drivor_navtrain_debug"
    [ -d "$CACHE" ] || { echo "ERROR: no cache at $CACHE -- run: bash cache_drivor.sh debug" >&2; exit 1; }
    exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
        agent=$AGENT \
        "${RECIPE[@]}" \
        experiment_name="smoke_drivor_${ARM}" \
        train_test_split=navtrain_debug \
        cache_path="$CACHE" \
        use_cache_without_dataset=true \
        force_cache_computation=false \
        dataloader.params.batch_size=2 \
        dataloader.params.num_workers=2 \
        trainer.params.max_epochs=1 \
        trainer.params.fast_dev_run=true \
        +trainer.params.devices=1 \
        trainer.params.strategy=auto
fi

CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_drivor_navtrain"
[ -d "$CACHE" ] || { echo "ERROR: no cache at $CACHE -- run: bash cache_drivor.sh" >&2; exit 1; }

# Global batch 64 is the recipe; the per-GPU batch follows from the device count, so lr and T_max
# (both computed from batch_size * agent.num_gpus) are the same on any GPU count.
# agent.num_gpus is not redundant with trainer.params.devices: with the yaml default (1), a 4-GPU
# run would silently get half the lr and a 4x longer schedule. Both are set from DEVICES below.
GLOBAL_BATCH=64
DEVICES="${DRIVOR_DEVICES:-4}"
if ! [[ "$DEVICES" =~ ^[1-9][0-9]*$ ]] || (( GLOBAL_BATCH % DEVICES != 0 )); then
    echo "ERROR: DRIVOR_DEVICES=$DEVICES must divide the global batch $GLOBAL_BATCH" >&2
    exit 2
fi
BATCH_PER_GPU=$((GLOBAL_BATCH / DEVICES))

# Loader workers are per rank, so the DDP job runs DEVICES x this many, on top of the Ray pool the
# ranks share for the scorer GT (DRIVOR_RAY_WORKERS_PER_GPU x num_gpus). The loaders only read the
# prebuilt cache; much higher values can exhaust /dev/shm and host RAM.
WORKERS_PER_RANK="${DRIVOR_LOADER_WORKERS:-16}"

# On some Blackwell hosts DDP dies at Lightning's first barrier with "CUDA error: an illegal memory
# access was encountered". If so, export NCCL_ALGO=Ring (see Notes in the OdysseyZoo README), and
# NCCL_SHM_DISABLE=1 if still needed. Neither is set here: they are host workarounds, not the recipe.
echo "=== train DrivoR: arm=$ARM agent=$AGENT devices=$DEVICES x batch $BATCH_PER_GPU ==="
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
    agent=$AGENT \
    "${RECIPE[@]}" \
    experiment_name="drivor_${ARM}_scratch25_lr2e4_bs64_seed2" \
    train_test_split=navtrain \
    cache_path="$CACHE" \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    dataloader.params.batch_size=$BATCH_PER_GPU \
    dataloader.params.num_workers=$WORKERS_PER_RANK \
    trainer.params.max_epochs=25 \
    agent.num_gpus=$DEVICES \
    +trainer.params.devices=$DEVICES \
    trainer.params.strategy=ddp
