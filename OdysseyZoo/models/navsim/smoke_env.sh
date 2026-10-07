# Environment sourced by cache_ltf.sh, cache_metric_ltf.sh, train_ltf.sh and eval_ltf.sh.
# Paths come from this file's location or from the environment (see the OdysseyZoo README).
# OPENSCENE_DATA_ROOT = dataset tree in the layout navsim expects (navsim_logs/<split>,
#                       sensor_blobs/<split>); required, no default
# NUPLAN_MAPS_ROOT    = nuPlan maps root; required, no default
# PY                  = python interpreter; defaults to `python` on PATH, i.e. the active env
# NAVSIM_EXP_ROOT is not set here: each calling script exports it after sourcing this file.
_SMOKE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="$_SMOKE_ENV_DIR"
: "${OPENSCENE_DATA_ROOT:?set OPENSCENE_DATA_ROOT to the OpenScene dataset root (navsim_logs/, sensor_blobs/)}"
: "${NUPLAN_MAPS_ROOT:?set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PY:-python}"
