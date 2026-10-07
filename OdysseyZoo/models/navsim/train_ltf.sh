#!/usr/bin/env bash
# Train LTF, either arm.
#
#   bash train_ltf.sh baseline              upstream LTF: driving_command in, no route
#   bash train_ltf.sh sdroute               SD route in (24 route tokens, cross-attention in every
#                                           decoder layer), driving_command removed
#
#   bash train_ltf.sh <arm> smoke           fast_dev_run (1 batch, 1 device); run this first
#   RESUME_CKPT=<ckpt> bash train_ltf.sh <arm>   resume an interrupted full run
#
# Both arms share this script so that lr, batch size, epochs, devices and seed stay identical
# across them; only `agent=` changes.
#
# Requires the cache: run `bash cache_ltf.sh` (or `bash cache_ltf.sh debug` for smoke runs) once.
# Both arms share it.
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

ARM="${1:-}"
MODE="${2:-full}"
case "$ARM" in
    baseline)        AGENT=ltf_baseline_agent ;;
    sdroute)         AGENT=ltf_sdroute_agent ;;
    *)
        echo "usage: $0 {baseline|sdroute} [smoke]" >&2
        exit 2 ;;
esac

if [ "$MODE" = "smoke" ]; then
    CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_ltf_navtrain_debug"
    [ -d "$CACHE" ] || { echo "ERROR: no cache at $CACHE -- run: bash cache_ltf.sh debug" >&2; exit 1; }
    exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
        agent=$AGENT \
        experiment_name="smoke_ltf_${ARM}" \
        train_test_split=navtrain_debug \
        cache_path="$CACHE" \
        use_cache_without_dataset=true \
        force_cache_computation=false \
        dataloader.params.batch_size=2 \
        dataloader.params.num_workers=2 \
        trainer.params.max_epochs=1 \
        trainer.params.fast_dev_run=true \
        ~trainer.params.strategy \
        +trainer.params.devices=1
fi

CACHE="$NAVSIM_DEVKIT_ROOT/exp/cache_ltf_navtrain"
[ -d "$CACHE" ] || { echo "ERROR: no cache at $CACHE -- run: bash cache_ltf.sh" >&2; exit 1; }

# RESUME_CKPT is an env var rather than an argument so it cannot turn into a per-arm setting.
# It is passed as a quoted hydra value because checkpoint names contain '=' (epoch=N-step=M.ckpt).
RESUME_ARGS=()
if [ -n "$RESUME_CKPT" ]; then
    [ -f "$RESUME_CKPT" ] || { echo "ERROR: RESUME_CKPT not found: $RESUME_CKPT" >&2; exit 1; }
    echo "=== resuming from: $RESUME_CKPT ==="
    RESUME_ARGS+=("resume_ckpt_path='$RESUME_CKPT'")
fi

# devices = number of GPUs in CUDA_VISIBLE_DEVICES (unset -> 1). The global batch is 16 x NGPU,
# so train both arms on the same GPU count.
_CVD="${CUDA_VISIBLE_DEVICES:-0}"; NGPU=$(echo "$_CVD" | tr ',' '\n' | grep -c .); [ "$NGPU" -lt 1 ] && NGPU=1
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
    agent=$AGENT \
    experiment_name="ltf_${ARM}_navtrain" \
    train_test_split=navtrain \
    cache_path="$CACHE" \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    agent.lr=1e-4 \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=16 \
    trainer.params.max_epochs=100 \
    +trainer.params.devices=$NGPU \
    trainer.params.strategy=ddp \
    "${RESUME_ARGS[@]}"
