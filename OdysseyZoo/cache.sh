#!/usr/bin/env bash
# OdysseyZoo: build the LTF or DrivoR dataset cache.
#
#   bash cache.sh <model> [debug|navtrain]
#   bash cache.sh --list
#
# Examples:
#   bash cache.sh ltf debug        small, for smoke runs
#   bash cache.sh drivor           full navtrain
#
# One cache per model serves both arms. It is built with the sdroute config, whose target
# builder adds the route tensor; the baseline ignores that tensor, so switching arms never
# needs a re-cache.
#
# The route comes from the shared builder in sdroute/, so all models see the same route.
set -eo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

declare -A CACHERS=(
    [ltf]="navsim:cache_ltf.sh"
    [drivor]="DrivoR:cache_drivor.sh"
)
ORDER="ltf drivor"

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ] || [ -z "${1:-}" ]; then
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    echo "models:"
    for m in $ORDER; do
        IFS=: read -r dir script <<<"${CACHERS[$m]}"
        printf "  %-12s models/%s/%s\n" "$m" "$dir" "$script"
    done
    echo
    echo "extra caches some models need:"
    echo "  ltf     models/navsim/cache_metric_ltf.sh    (PDM metric cache, for scoring)"
    echo "  drivor  models/DrivoR/cache_metric_drivor.sh [train|eval|all]  (train_metric_cache for training, navtest cache for scoring)"
    exit 0
fi

MODEL="$1"; shift
if [ -z "${CACHERS[$MODEL]:-}" ]; then
    echo "unknown model: $MODEL" >&2
    echo "pick one of: $ORDER" >&2
    exit 2
fi

IFS=: read -r dir script <<<"${CACHERS[$MODEL]}"
target="$ROOT/models/$dir/$script"
[ -f "$target" ] || { echo "missing: $target" >&2; exit 1; }

echo "==> caching $MODEL  ->  models/$dir/$script $*"
echo
exec bash "$target" "$@"
