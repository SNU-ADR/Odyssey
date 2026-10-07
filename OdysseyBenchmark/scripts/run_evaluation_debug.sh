#!/usr/bin/env bash
# Smoke test on three scenes before the full evaluation:
#   odyssey_scene011 nr (signal timetable), odyssey_scene075 r (reactive spawn gate off),
#   odyssey_scene002 nr (plain), 40 steps each.
# Usage: AGENT_CONFIG=my_model/agent.yaml GPUS=0 bash OdysseyBenchmark/scripts/run_evaluation_debug.sh [extra eval options]
set -euo pipefail
AGENT_CONFIG=${AGENT_CONFIG:?set AGENT_CONFIG to your agent config (e.g. OdysseyBenchmark/agents/ltf_sdroute.yaml)}
GPUS=${GPUS:-0}
MAX_STEPS=${MAX_STEPS:-40}
TIMEOUT=${TIMEOUT:-1200}
SAVE_PATH=${SAVE_PATH:-}            # campaign directory (relative: to the current directory); default experiments/simulation/eval_<model>_debug
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
exec "$ODYSSEY_SIM_PY" -m odyssey_runtime eval --agent "$AGENT_CONFIG" --gpus "$GPUS" --scenes debug \
  --max-steps "$MAX_STEPS" --timeout "$TIMEOUT" ${SAVE_PATH:+--output "$SAVE_PATH"} "$@"
