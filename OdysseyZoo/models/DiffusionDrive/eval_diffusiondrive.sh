#!/usr/bin/env bash
# Open-loop navtest PDM-score evaluation of DiffusionDrive.
#
#   bash eval_diffusiondrive.sh baseline [ckpt] [hydra.override=value ...]
#   bash eval_diffusiondrive.sh sdroute  [ckpt] [hydra.override=value ...]
#   ckpt defaults to ckpts/diffusiondrive_<arm>.ckpt; an argument ending in .ckpt or .pth is taken as
#   the ckpt, and trailing hydra.override=value args are forwarded to run_pdm_score.py.
#   env overrides: SPLIT, METRIC_CACHE, WORKERS, EXP_NAME
#
# The sdroute arm needs the Scene to build its route at eval: TransfuserAgent sets
# requires_scene=True and overrides compute_trajectory(agent_input, scene). If the route is
# missing, the model raises and the token is marked valid=False rather than scored route-blind.
#
# `latent` must match the checkpoint. Both agent configs set latent: True (camera-only), as the
# shipped checkpoints were trained; pass agent.config.latent=False for a LiDAR checkpoint such as
# upstream's diffusiondrive_navsim_88p1_PDMS.pth. A mismatch fails the strict load in agent.initialize().
set -eo pipefail
cd "$(dirname "$0")"
source ./smoke_env.sh

ARM="${1:-}"
case "$ARM" in
    baseline|sdroute) ;;
    *)
        echo "usage: $0 {baseline|sdroute} [checkpoint.ckpt|checkpoint.pth] [hydra.override=value ...]" >&2
        exit 2 ;;
esac
shift
AGENT=diffusiondrive_${ARM}_agent
CKPT="$NAVSIM_DEVKIT_ROOT/ckpts/diffusiondrive_${ARM}.ckpt"
# The ckpt is recognised by its suffix: Lightning names ckpts epoch=N-step=M.ckpt, so '=' cannot
# tell a ckpt from an override.
if [[ "${1:-}" == *.ckpt || "${1:-}" == *.pth ]]; then CKPT="$1"; shift; fi
[ -f "$CKPT" ] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; }

# Trailing Hydra overrides (contain '='), e.g. agent.config.latent=False for a LiDAR ckpt.
PASS_HYDRA=()
for arg in "$@"; do
    case "$arg" in
        *=*) PASS_HYDRA+=("$arg") ;;
        *) echo "WARNING: ignoring '$arg' (expected hydra key=value)" >&2 ;;
    esac
done

SPLIT="${SPLIT:-navtest}"
WORKERS="${WORKERS:-32}"
METRIC_CACHE="${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_${SPLIT}}"

# Build the metric cache if it is missing or half-built. run_metric_caching writes
# metadata/*.csv only after the last scenario, and MetricCacheLoader reads that csv: a dir with
# no metadata csv dies on a bare "IndexError: list index out of range" minutes into the eval. So
# treat "no metadata csv" as "not built" and (re)build; it resumes, skipping cached scenarios.
if ! ls "$METRIC_CACHE"/metadata/*.csv >/dev/null 2>&1; then
    echo "=== metric cache missing/incomplete at $METRIC_CACHE -- building (resumes) ==="
    METRIC_CACHE="$METRIC_CACHE" SPLIT="$SPLIT" bash "$NAVSIM_DEVKIT_ROOT/cache_metric_diffusiondrive.sh"
fi

EXP_NAME="${EXP_NAME:-eval_diffusiondrive_${ARM}_${SPLIT}}"

echo "=== eval DiffusionDrive: arm=$ARM split=$SPLIT ==="
echo "    ckpt         = $CKPT"
echo "    metric cache = $METRIC_CACHE"
echo "    experiment   = $EXP_NAME  (-> $NAVSIM_EXP_ROOT/$EXP_NAME/<start timestamp>/<end timestamp>.csv)"

# The ckpt path is passed as a quoted hydra value: lightning names files
# `epoch=N-step=M.ckpt` and the bare `=` makes hydra's override grammar fail otherwise.
exec $PY "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    agent=$AGENT \
    "agent.checkpoint_path='$CKPT'" \
    experiment_name="$EXP_NAME" \
    train_test_split=$SPLIT \
    metric_cache_path="$METRIC_CACHE" \
    worker=single_machine_thread_pool \
    worker.max_workers=$WORKERS \
    "${PASS_HYDRA[@]}"
