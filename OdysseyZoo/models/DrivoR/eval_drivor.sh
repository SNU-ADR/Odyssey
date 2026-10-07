#!/usr/bin/env bash
# Open-loop PDM-score eval of DrivoR on navtest.
#
#   bash eval_drivor.sh baseline [ckpt]   upstream DrivoR: driving command in, no route
#   bash eval_drivor.sh sdroute  [ckpt]   SD route in, driving command dropped
#   ckpt defaults to ckpts/drivor_<arm>.ckpt.
#
# Environment: WORKERS (Ray workers, default 30).
#
# Needs, once: bash cache_metric_drivor.sh eval, and the DINOv2 weights (smoke_env.sh says where).
#
# One script for both arms: split, metric cache and scorer must be identical across arms or the
# comparison is void. Only the agent config and the ckpt change.
#
# The sdroute arm needs nothing extra here: run_pdm_score.py passes the Scene to
# DrivoRAgent.compute_trajectory(), which builds the route from it (see that method).
#
# A checkpoint is required, not only for the weights: checkpoint_path '' is DrivoRAgent's
# training mode (see drivor_baseline_agent.yaml).
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"

ARM="${1:-}"
CKPT="${2:-}"
case "$ARM" in
    baseline|sdroute) ;;
    *)
        echo "usage: $0 {baseline|sdroute} [checkpoint.ckpt]" >&2
        exit 2 ;;
esac
AGENT=drivor_${ARM}_agent
CKPT="${CKPT:-$NAVSIM_DEVKIT_ROOT/ckpts/drivor_${ARM}.ckpt}"
[ -f "$CKPT" ] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }
require_dinov2_weights "$AGENT"

# The navtest metric cache, built by cache_metric_drivor.sh eval. Check metadata/*.csv rather than
# the directory: a half-built cache lacks it, and MetricCacheLoader then fails with an IndexError.
METRIC_CACHE="$NAVSIM_EXP_ROOT/metric_cache"
ls "$METRIC_CACHE"/metadata/*.csv >/dev/null 2>&1 || {
    echo "ERROR: no complete navtest metric cache at $METRIC_CACHE (no metadata/*.csv)." >&2
    echo "       Build or finish it: bash cache_metric_drivor.sh eval (it resumes)." >&2
    exit 1; }

# Ray workers (the config's default worker); each builds its own agent and runs it on CPU.
WORKERS="${WORKERS:-30}"

echo "=== eval DrivoR: arm=$ARM agent=$AGENT split=navtest ==="
echo "    ckpt         = $CKPT"
echo "    metric cache = $METRIC_CACHE"
# The ckpt path is quoted: Lightning names files `epoch=N-step=M.ckpt`, and a bare `=` inside an
# override value breaks Hydra's override grammar.
# agent.batch_size / agent.scheduler_args interpolate training-config keys
# (dataloader.params.batch_size, trainer.params.max_epochs) that run_pdm_score's config lacks, so
# instantiating the agent would fail. Only get_optimizers() reads them, so they are set to null.
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=$AGENT \
    "agent.checkpoint_path='$CKPT'" \
    agent.batch_size=null \
    agent.scheduler_args=null \
    experiment_name="eval_drivor_${ARM}_navtest" \
    train_test_split=navtest \
    metric_cache_path="$METRIC_CACHE" \
    worker.threads_per_node=$WORKERS
