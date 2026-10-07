#!/usr/bin/env bash
# Open-loop PDM-score eval of LTF on navtest.
#
#   bash eval_ltf.sh baseline [ckpt]   upstream LTF: driving command in, no route
#   bash eval_ltf.sh sdroute  [ckpt]   SD route in (24 route tokens, cross-attn in every decoder
#                                      layer), driving command removed
#   ckpt defaults to ckpts/ltf_<arm>.ckpt.
#   env overrides: SPLIT (e.g. navtest_smoke), METRIC_CACHE, WORKERS (default 32),
#                  SCRIPT (scoring entry point, see below)
#
# Both arms use the same split, metric cache and scorer; only the agent and ckpt change.
# The sdroute agent sets requires_scene=True, so the scorer passes the Scene and
# TransfuserAgent.compute_trajectory() rebuilds the route from it.
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
CKPT="${CKPT:-$NAVSIM_DEVKIT_ROOT/ckpts/ltf_${ARM}.ckpt}"
[ -f "$CKPT" ] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }

# Metric cache built by cache_metric_ltf.sh (caches from other navsim versions are not compatible).
SPLIT="${SPLIT:-navtest}"
METRIC_CACHE="${METRIC_CACHE:-$NAVSIM_DEVKIT_ROOT/exp/metric_cache_${SPLIT}}"
[ -d "$METRIC_CACHE" ] || { echo "ERROR: no metric cache at $METRIC_CACHE -- run: bash cache_metric_ltf.sh" >&2; exit 1; }

# `-d` cannot tell a finished cache from an interrupted one. run_metric_caching.py writes
# metadata/*.csv only after the last scenario, and MetricCacheLoader needs it, so check for it here.
ls "$METRIC_CACHE"/metadata/*.csv >/dev/null 2>&1 || {
    echo "ERROR: metric cache at $METRIC_CACHE is INCOMPLETE (no metadata/*.csv)." >&2
    echo "       Finish it first: bash cache_metric_ltf.sh" >&2
    echo "       It does not resume: every scenario is recomputed." >&2
    exit 1; }

WORKERS="${WORKERS:-32}"

echo "=== eval LTF: arm=$ARM split=$SPLIT ==="
echo "    ckpt         = $CKPT"
echo "    metric cache = $METRIC_CACHE"

# The ckpt is passed as a quoted hydra value: lightning checkpoint names contain '='
# (epoch=N-step=M.ckpt), which hydra's override grammar rejects unquoted.
# navtest has no stage-two (synthetic) scenes, so it is scored with run_pdm_score_one_stage.py;
# run_pdm_score.py needs a *_two_stage split.
SCRIPT="${SCRIPT:-run_pdm_score_one_stage.py}"
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/$SCRIPT" \
    agent=ltf_${ARM}_agent \
    "agent.checkpoint_path='$CKPT'" \
    experiment_name="eval_ltf_${ARM}_${SPLIT}" \
    train_test_split=$SPLIT \
    metric_cache_path="$METRIC_CACHE" \
    worker=single_machine_thread_pool \
    worker.max_workers=$WORKERS
