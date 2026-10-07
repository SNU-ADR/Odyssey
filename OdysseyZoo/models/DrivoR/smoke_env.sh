# Environment sourced by cache_drivor.sh, cache_metric_drivor.sh, train_drivor.sh and eval_drivor.sh.
# Every path below is derived from this file's own location or taken from the environment, so the
# repo runs from any checkout:
# OPENSCENE_DATA_ROOT = dataset tree in the layout navsim expects (navsim_logs/<split>,
#                       sensor_blobs/<split>); required, no default
# NUPLAN_MAPS_ROOT    = nuPlan maps root; required, no default
# PY                  = python of the DrivoR env (its own torch stack, not the shared navsim env);
#                       defaults to `python` on PATH, i.e. the active env
# NAVSIM_EXP_ROOT is not set here: each caller exports it right after sourcing this file.
_SMOKE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="$_SMOKE_ENV_DIR"
: "${OPENSCENE_DATA_ROOT:?set OPENSCENE_DATA_ROOT to the OpenScene dataset root (navsim_logs/, sensor_blobs/)}"
: "${NUPLAN_MAPS_ROOT:?set NUPLAN_MAPS_ROOT to the nuPlan maps root}"
export OPENSCENE_DATA_ROOT NUPLAN_MAPS_ROOT
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# nuplan comes from the vendored nuplan-devkit/, whatever the env's own editable install points at.
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$NAVSIM_DEVKIT_ROOT/nuplan-devkit${PYTHONPATH:+:$PYTHONPATH}"
export PY="${PY:-python}"

# DINOv2 ViT-S/14 weights. The agent yaml's `model_weights` is the one source of their location, a
# path relative to the cwd (every script here cd's to this directory first); this only checks that
# the file is there. The model itself does not fail on a missing file: dinov2_lora.py silently falls
# back to the Hugging Face hub (download or local HF cache), which fails on an offline node.
#   require_dinov2_weights <agent yaml name>
require_dinov2_weights() {
    local yaml="$NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/common/agent/$1.yaml" ws w p
    [ -f "$yaml" ] || { echo "ERROR: no agent config $yaml" >&2; return 1; }
    ws="$(sed -n 's/^ *model_weights: *//p' "$yaml" | sort -u)"
    [ -n "$ws" ] || { echo "ERROR: $yaml sets no model_weights" >&2; return 1; }
    for w in $ws; do
        case "$w" in /*) p="$w" ;; *) p="$PWD/$w" ;; esac
        [ -f "$p" ] && continue
        echo "ERROR: DINOv2 weights missing: $p (model_weights in $1.yaml)." >&2
        echo "       They are model.safetensors of timm/vit_small_patch14_reg4_dinov2.lvd142m on the" >&2
        echo "       Hugging Face hub (https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m):" >&2
        echo "         $PY -c \"from huggingface_hub import hf_hub_download; hf_hub_download(" >&2
        echo "             'timm/vit_small_patch14_reg4_dinov2.lvd142m', 'model.safetensors'," >&2
        echo "             local_dir='$(dirname "$p")')\"" >&2
        return 1
    done
}
