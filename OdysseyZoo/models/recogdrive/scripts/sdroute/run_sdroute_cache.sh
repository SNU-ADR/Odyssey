#!/usr/bin/env bash
# Build the SD-route cache <OUT>/<log>/<token>/sdroute_target.gz with only this repo's navsim on
# PYTHONPATH.
#
# The route comes from the shared builder (OdysseyZoo/sdroute) through the sdroute wrapper of the
# neighbouring LTF tree; the PDM helpers it uses come from this repo's navsim, which yields the same
# route bit for bit as the other repos' navsim.
#
# Env: OUT (default exp/sdroute_cache_navtrain), WORKERS (8), PY (python on PATH),
#      WRAPPER (../navsim/navsim/agents/sdroute/route_target.py).
# Extra args go to make_sdroute_cache.py. For the navtest cache (agent.sdroute_cache_path at
# evaluation) set OUT=<navtest cache root> and pass
#   --split-yaml navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml
#   --data-path $OPENSCENE_DATA_ROOT/navsim_logs/test --sensor-blobs-path $OPENSCENE_DATA_ROOT/sensor_blobs/test
#
# SD-route training cache (navtrain):
#   1. this script -> <OUT>
#   2. hidden-state caching: add agent.nav_prompt=sdroute_text agent.sdroute_cache_path=<OUT>
#      agent.hidden_state_dtype=bfloat16 to scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh;
#      agent.use_sdroute stays false there, since the route target is cache-only
#   3. scripts/sdroute/merge_route_cache.py --hidden <hidden-state cache> --route <OUT> --out <merged>
#   4. train: in a scripts/training/ script set agent=recogdrive_sdroute_agent cache_path=<merged>
#      use_cache_without_dataset=True
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="${WRAPPER:-${REPO_ROOT}/../navsim/navsim/agents/sdroute/route_target.py}"
PY="$(command -v "${PY:-python}")"   # the recogdrive env's python; defaults to `python` on PATH
: "${OPENSCENE_DATA_ROOT:?set OPENSCENE_DATA_ROOT to the OpenScene dataset root (navsim_logs/, sensor_blobs/)}"
: "${NUPLAN_MAPS_ROOT:?set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
OUT="${OUT:-${REPO_ROOT}/exp/sdroute_cache_navtrain}"
WORKERS="${WORKERS:-8}"

[ -f "${WRAPPER}" ] || { echo "sdroute wrapper not found: WRAPPER=${WRAPPER}" >&2; exit 1; }

cd "${REPO_ROOT}"
env -i HOME="${HOME}" PATH="$(dirname "${PY}"):/usr/bin:/bin" \
    PYTHONPATH="${REPO_ROOT}" \
    NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
    NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT}" \
    OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT}" \
    "${PY}" scripts/sdroute/make_sdroute_cache.py --out "${OUT}" --workers "${WORKERS}" \
        --wrapper "${WRAPPER}" "$@"
