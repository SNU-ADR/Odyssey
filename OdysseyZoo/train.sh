#!/usr/bin/env bash
# OdysseyZoo: one training entry point for LTF and DrivoR, either arm.
# DiffusionDrive, SafeDrive and ReCogDrive are not dispatched from here; call their own scripts.
#
#   bash train.sh <model> <arm> [smoke|full]
#   bash train.sh --list                       what exists, per model
#   bash train.sh --matrix                     the same as a grid
#
#   ODYSSEYZOO_DRY_RUN=1 bash train.sh ...    resolve and print, start nothing
#
# Examples:
#   bash train.sh ltf sdroute smoke            always do this before a full run
#   bash train.sh drivor baseline
#
# This only dispatches: each model keeps its own train_*.sh with its own lr / batch / devices,
# shared by both of its arms. Extra arguments are forwarded unchanged.
#
# Arms:
#   baseline         the upstream model: driving_command in, no route
#   sdroute          SD route in (24 segment tokens, gated cross-attention inside every
#                    decoder layer), driving_command dropped
set -eo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

# model -> "<dir>:<script>"
declare -A MODELS=(
    [ltf]="navsim:train_ltf.sh"
    [drivor]="DrivoR:train_drivor.sh"
)

# model -> arms it actually has
declare -A ARMS=(
    [ltf]="baseline sdroute"
    [drivor]="baseline sdroute"
)

ORDER="ltf drivor"
ALL_ARMS="baseline sdroute"

has_arm() {  # $1=model $2=arm
    case " ${ARMS[$1]} " in *" $2 "*) return 0 ;; *) return 1 ;; esac
}

list_models() {
    echo "models:"
    for m in $ORDER; do
        IFS=: read -r dir script <<<"${MODELS[$m]}"
        printf "  %-12s models/%s/%s\n" "$m" "$dir" "$script"
        printf "  %-12s arms: %s\n\n" "" "${ARMS[$m]}"
    done
}

print_matrix() {
    printf "%-13s" "model"
    for a in $ALL_ARMS; do printf "%-17s" "$a"; done
    echo
    printf "%-13s" ""
    for a in $ALL_ARMS; do printf "%-17s" "-----"; done
    echo
    for m in $ORDER; do
        printf "%-13s" "$m"
        for a in $ALL_ARMS; do
            if has_arm "$m" "$a"; then printf "%-17s" "yes"; else printf "%-17s" "--"; fi
        done
        echo
    done
}

case "${1:-}" in
    --list|-l)   list_models; exit 0 ;;
    --matrix|-m) print_matrix; exit 0 ;;
    ""|--help|-h)
        sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
        echo
        print_matrix
        exit 0 ;;
esac

MODEL="$1"
ARM="${2:-}"
shift 2 2>/dev/null || true

if [ -z "${MODELS[$MODEL]:-}" ]; then
    echo "unknown model: $MODEL" >&2
    echo "pick one of: $ORDER" >&2
    exit 2
fi

if [ -z "$ARM" ]; then
    echo "usage: $0 $MODEL <arm> [smoke|full]" >&2
    echo "arms for $MODEL: ${ARMS[$MODEL]}" >&2
    exit 2
fi

if ! has_arm "$MODEL" "$ARM"; then
    echo "$MODEL has no '$ARM' arm." >&2
    echo >&2
    echo "arms for $MODEL: ${ARMS[$MODEL]}" >&2
    exit 2
fi

IFS=: read -r dir script <<<"${MODELS[$MODEL]}"
target="$ROOT/models/$dir/$script"
[ -x "$target" ] || [ -f "$target" ] || { echo "missing: $target" >&2; exit 1; }

echo "==> $MODEL / $ARM  ->  models/$dir/$script $ARM $*"
# ODYSSEYZOO_DRY_RUN=1 stops here: the dispatch is resolved and printed, nothing starts.
if [ -n "${ODYSSEYZOO_DRY_RUN:-}" ]; then
    echo "(dry run -- not started)"
    exit 0
fi
echo
exec bash "$target" "$ARM" "$@"
