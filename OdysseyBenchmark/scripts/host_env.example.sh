# Odyssey shell environment. Copy this file, set the five paths in the first block, and source it
# before running anything (the OdysseyBenchmark/scripts/run_evaluation_*.sh wrappers also take it as HOST_ENV=<file>):
#
#   cp OdysseyBenchmark/scripts/host_env.example.sh ~/odyssey_env.sh     # then edit the first block
#   source ~/odyssey_env.sh && python OdysseyBenchmark/tools/check_env.py
#
# The rest follows the layout docs/install-source.md builds under ODYSSEY_RUNTIME_ROOT; change a line
# only where your install differs. ODYSSEY_* variables other than these are refused by the launcher.

# ---- your paths ------------------------------------------------------------------------------------
export ODYSSEY_ROOT=/absolute/path/to/odyssey                    # this checkout
export ODYSSEY_RUNTIME_ROOT=/absolute/path/to/odyssey-runtime    # interpreters, CUDA 12.8, caches, maps
export ODYSSEY_MODELS_ROOT=/absolute/path/to/odyssey-models      # hf download ADRLAB/odyssey-models
export ODYSSEY_SCENES_ROOT=/absolute/path/to/odyssey-scenes      # hf download ADRLAB/odyssey-scenes
export NUPLAN_MAPS_ROOT=$ODYSSEY_RUNTIME_ROOT/maps/final_gpu0     # map copy of the first GPU (OdysseyBenchmark/scripts/make_map_slots.sh)

# ---- interpreters (docs/install-source.md, OdysseyRenderer/fixer/README.md, docs/installation.md "ReCogDrive") -----
export ODYSSEY_SIM_PY=$ODYSSEY_RUNTIME_ROOT/envs/simulator/bin/python
export ODYSSEY_PLANNER_PY=$ODYSSEY_RUNTIME_ROOT/envs/planner/bin/python       # the shipped configs use it
export RECOGDRIVE_PLANNER_PY=$ODYSSEY_RUNTIME_ROOT/envs/recogdrive/bin/python # ReCogDrive configs only
export ODYSSEY_FIXER_ROOT=$ODYSSEY_ROOT/OdysseyRenderer/fixer
export ODYSSEY_FIXER_PY=$ODYSSEY_RUNTIME_ROOT/fixer/bin/python

# ---- assets ----------------------------------------------------------------------------------------
export ODYSSEY_ZOO_ROOT=$ODYSSEY_ROOT/OdysseyZoo                 # bundled with the checkout
export RECOGDRIVE_VLM_PATH=$ODYSSEY_MODELS_ROOT/models/recogdrive/ckpts/ReCogDrive-VLM-2B
export HF_HOME=$ODYSSEY_RUNTIME_ROOT/cache/huggingface           # timm backbones (DiffusionDrive, SafeDrive)
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1                                          # after the backbone downloads are in the cache

# ---- CUDA and Python -------------------------------------------------------------------------------
export CUDA_HOME=$ODYSSEY_RUNTIME_ROOT/cuda-12.8                 # gsplat needs nvcc 12.8 on PATH at run time
export PATH=$(dirname "$ODYSSEY_SIM_PY"):$CUDA_HOME/bin:$PATH
export PYTHONPATH=$ODYSSEY_ROOT/OdysseyBenchmark:$ODYSSEY_ROOT/OdysseyRenderer:$ODYSSEY_ROOT/OdysseyTrafficAgent:$ODYSSEY_ROOT/OdysseyBenchmark/odyssey_bridge:$ODYSSEY_ROOT/third_party/nvdiffrast
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
export TORCH_EXTENSIONS_DIR=$ODYSSEY_RUNTIME_ROOT/torch_extensions_${TORCH_CUDA_ARCH_LIST}   # one per GPU type
export TMPDIR=$ODYSSEY_RUNTIME_ROOT/tmp                          # JIT builds need an exec-mounted temp dir
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1                        # the native checkpoints are full pickles
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export ODYSSEY_FIXER_EMPTY_CACHE=0
mkdir -p "$TMPDIR"

# ---- report what does not exist yet (OdysseyBenchmark/tools/check_env.py checks the rest) ---------------------------
for _odyssey_var in ODYSSEY_SIM_PY ODYSSEY_PLANNER_PY ODYSSEY_FIXER_PY; do
  [ -x "${!_odyssey_var}" ] || echo "host_env: $_odyssey_var=${!_odyssey_var} is not an executable" >&2
done
for _odyssey_var in ODYSSEY_ROOT ODYSSEY_MODELS_ROOT ODYSSEY_SCENES_ROOT NUPLAN_MAPS_ROOT CUDA_HOME; do
  [ -d "${!_odyssey_var}" ] || echo "host_env: $_odyssey_var=${!_odyssey_var} does not exist" >&2
done
[ -n "$TORCH_CUDA_ARCH_LIST" ] || echo "host_env: nvidia-smi found no GPU; set TORCH_CUDA_ARCH_LIST by hand" >&2
unset _odyssey_var
