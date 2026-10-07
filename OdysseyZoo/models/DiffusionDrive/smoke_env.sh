# Env for every DiffusionDrive script here (cache_diffusiondrive.sh, cache_metric_diffusiondrive.sh,
# train_diffusiondrive.sh, eval_diffusiondrive.sh); each one sources this file.
# Every path below is derived from this file's own location or taken from the environment, so the
# repo runs from any checkout -- see OdysseyZoo's top-level README.md.
# OPENSCENE_DATA_ROOT  = OpenScene/navsim dataset root (required)
# NUPLAN_MAPS_ROOT     = nuPlan maps root (required; must be writable, nuPlan opens map.gpkg in WAL mode)
# PY                   = python with requirements.txt installed (default: `python` on PATH)
# The SD-route builder is OdysseyZoo's own sdroute/, loaded by navsim/agents/sdroute/route_target.py.
_SMOKE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="$_SMOKE_ENV_DIR"
: "${OPENSCENE_DATA_ROOT:?set OPENSCENE_DATA_ROOT to the OpenScene/navsim dataset root}"
: "${NUPLAN_MAPS_ROOT:?set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/exp"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PY:-python}"
