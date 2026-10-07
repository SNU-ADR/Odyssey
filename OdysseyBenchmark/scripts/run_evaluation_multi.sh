#!/usr/bin/env bash
# Full evaluation: every benchmark scene x {nr, r} spread over GPUS. Re-running the same command
# continues an interrupted campaign (complete jobs are skipped, infrastructure failures retried).
# Usage: AGENT_CONFIG=my_model/agent.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_multi.sh [extra eval options]
set -euo pipefail
AGENT_CONFIG=${AGENT_CONFIG:?set AGENT_CONFIG to your agent config (e.g. OdysseyBenchmark/agents/ltf_sdroute.yaml)}
GPUS=${GPUS:-0,1,2,3}
REACT=${REACT:-nr,r}
RETRY_INFRA=${RETRY_INFRA:-2}
TIMEOUT=${TIMEOUT:-5400}
MAX_STEPS=${MAX_STEPS:-}            # empty = the full scene horizon (the benchmark)
SEED=${SEED:-}
MAP_SLOTS=${MAP_SLOTS:-}            # one nuPlan map directory per GPU; default derived from NUPLAN_MAPS_ROOT
RUNS_PER_GPU=${RUNS_PER_GPU:-1}      # simulations at once per GPU; above 1, MAP_SLOTS must list one directory per run
SAVE_PATH=${SAVE_PATH:-}            # campaign directory (relative: to the current directory); default experiments/simulation/eval_<model>
# Paths given relative to the current directory keep working after the cd below; an agent config
# that is not found here is looked up from the repository root, as before.
here() { case "$1" in /*|"") printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
[ ! -e "$AGENT_CONFIG" ] || AGENT_CONFIG=$(here "$AGENT_CONFIG")
SAVE_PATH=$(here "$SAVE_PATH")
HOST_ENV=$(here "${HOST_ENV:-}")
cd "$(dirname "$0")/../.."                 # the repository root
if [ -n "$HOST_ENV" ]; then source "$HOST_ENV"; fi
: "${ODYSSEY_SIM_PY:?source your environment first (a copy of OdysseyBenchmark/scripts/host_env.example.sh with your paths), or set HOST_ENV to that file}"
: "${ODYSSEY_SCENES_ROOT:?set ODYSSEY_SCENES_ROOT to the odyssey-scenes download}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}
exec "$ODYSSEY_SIM_PY" -m odyssey_runtime eval --agent "$AGENT_CONFIG" --gpus "$GPUS" --scenes all --react "$REACT" \
  --retry-infra "$RETRY_INFRA" --timeout "$TIMEOUT" --runs-per-gpu "$RUNS_PER_GPU" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} ${SEED:+--seed "$SEED"} ${MAP_SLOTS:+--map-slots "$MAP_SLOTS"} \
  ${SAVE_PATH:+--output "$SAVE_PATH"} "$@"
