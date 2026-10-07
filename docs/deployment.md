# Reference deployment

The runtime was validated on one reference host. This page records that deployment and what a
new host needs; it is a tested snapshot, not a portable installer.

## Hardware and container

Linux x86-64, 4 × NVIDIA RTX PRO 5000 Blackwell (48 GiB, compute capability 12.0), driver
580.178.04, 8 GiB container `/dev/shm`. These are tested values, not minimum requirements.

The verified bootstrap is a frozen container image, supplied separately from Git. It holds the
simulator interpreter and CUDA build tools; it does not include the bind-mounted model and
data assets or the Fixer environment.

```bash
docker image load -i odyssey-environment.tar
IMAGE=<the supplied image tag>
docker image inspect "$IMAGE" --format '{{.Id}}'
WORKSPACE=/absolute/path/to/workspace        # the checkout and every linked asset
docker run --init --gpus all --shm-size=8g --entrypoint /bin/bash --name odyssey-sim \
  -v "$WORKSPACE:$WORKSPACE" -it "$IMAGE"
```

Use the NVIDIA Container Toolkit; `--init` reaps orphaned children. Preserve absolute mount
paths or recreate symlinks for the target layout. Do not copy another GPU's JIT cache.

## Interpreters

| Process | Tested environment | Purpose |
|---|---|---|
| Simulator (`ODYSSEY_SIM_PY`) | Python 3.9.23, Torch 2.7.1+cu128, NumPy 1.23.0, gsplat 1.4.0, nvdiffrast 0.4.0 | world, rendering, in-process scoring |
| Planner (`model.python` in the agent config) | the model's own interpreter (reference host: Torch 2.7.1) | native preprocessing and inference |
| Fixer (`ODYSSEY_FIXER_PY`) | Python 3.12, Torch 2.8.0+cu128 | image restoration |
| ReCogDrive planner (`RECOGDRIVE_PLANNER_PY`) | Python 3.10, Torch 2.7.1+cu128: the planner env's pinned packages plus transformers 4.37.2, tokenizers 0.15.1, sentencepiece 0.1.99, huggingface_hub 0.36.0, accelerate 0.27.2, peft 0.10.0, diffusers 0.30.3, timm 0.9.12, decord 0.6.0, einops-exts, click 8.2.1 (conda-forge Python; no FlashAttention2, InternVL falls back to eager attention) | ReCogDrive inference (its code needs Python >= 3.10) |

`OdysseyBenchmark/configs/deployment/environment-reference.json` is the installed package inventory.

## Shell

Start from `OdysseyBenchmark/scripts/host_env.example.sh` ([installation.md](installation.md)); it
exports the equivalents of:

```bash
export ODYSSEY_ROOT="$(pwd -P)"
export ODYSSEY_RUNTIME_ROOT=/absolute/path/to/odyssey-runtime      # caches, Fixer env, maps
export ODYSSEY_SIM_PY=/absolute/path/to/simulator/environment/bin/python
export ODYSSEY_PLANNER_PY=/absolute/path/to/model/environment/bin/python   # the shipped configs use ${ODYSSEY_PLANNER_PY}
export RECOGDRIVE_PLANNER_PY=/absolute/path/to/recogdrive/bin/python        # ReCogDrive configs only
export ODYSSEY_FIXER_ROOT="$ODYSSEY_ROOT/OdysseyRenderer/fixer"
export ODYSSEY_FIXER_PY="$ODYSSEY_RUNTIME_ROOT/fixer/bin/python"
export ODYSSEY_ZOO_ROOT="$ODYSSEY_ROOT/OdysseyZoo"                   # bundled with the checkout
export ODYSSEY_MODELS_ROOT=/absolute/path/to/odyssey-models
export RECOGDRIVE_VLM_PATH=$ODYSSEY_MODELS_ROOT/models/recogdrive/ckpts/ReCogDrive-VLM-2B   # ReCogDrive configs only
export ODYSSEY_SCENES_ROOT=/absolute/path/to/odyssey-scenes
export NUPLAN_MAPS_ROOT=/absolute/path/to/maps/final_gpu0            # one directory per concurrent run
export PYTHONPATH="$ODYSSEY_ROOT/OdysseyBenchmark:$ODYSSEY_ROOT/OdysseyRenderer:$ODYSSEY_ROOT/OdysseyTrafficAgent:$ODYSSEY_ROOT/OdysseyBenchmark/odyssey_bridge:$ODYSSEY_ROOT/third_party/nvdiffrast"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$ODYSSEY_SIM_PY"):$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader | tr -d ' ')"
export TORCH_EXTENSIONS_DIR="$ODYSSEY_RUNTIME_ROOT/torch_extensions_${TORCH_CUDA_ARCH_LIST}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export ODYSSEY_FIXER_EMPTY_CACHE=0                  # ODYSSEY_* names outside launch.py USER_ENV/LAUNCHER_ENV are refused
export HF_HOME="$ODYSSEY_RUNTIME_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1                     # after the backbone/tokenizer downloads are in the cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # the native checkpoints are full pickles
export TMPDIR="$ODYSSEY_RUNTIME_ROOT/tmp"   # where /tmp is mounted noexec: JIT builds run from TMPDIR
```

On a mixed-GPU host select the intended device before computing the architecture and use a
distinct extension directory per GPU type. The launcher places simulator, planner and Fixer on
the GPU given by `--gpu`; the restorer is fixed to `fixer_h1b16` (the h1_b16_e1 weights in torch).
The launcher sets `ODYSSEY_RESTORER=fixer_h1b16` for every run; it does not need to be set in the shell.

Installation check: `python OdysseyBenchmark/tools/check_env.py --gpus 0,1,2,3`, then one short episode
(`python -m odyssey_runtime run --agent OdysseyBenchmark/agents/ltf_sdroute.yaml --scene odyssey_scene001 --react nr --gpu 0 --max-steps 40`)
and its reactive counterpart. A successful run has exit code 0, the completion flag and a
scored CSV; import checks alone do not exercise every CUDA path.

## Assets on the reference host

- OdysseyZoo: bundled at `./OdysseyZoo` (the 2026-09-30 OdysseyZoo release with the
  SafeDrive native fix applied); its `ckpts/` and DrivoR `weights/` are links to the weight
  copies under `$ODYSSEY_RUNTIME_ROOT/native-models/`, ignored by git.
- Weights: `$ODYSSEY_RUNTIME_ROOT/native-models/odyssey-models-20260930` (the published
  release, SHA256SUMS verified); scenes: `$ODYSSEY_RUNTIME_ROOT/hf/odyssey-scenes`
  (700 files, MD5SUMS verified).
- Maps: `$ODYSSEY_RUNTIME_ROOT/maps/final_gpu{0,1,2,3}` and `final_slot{4,5,6,7}`
  (copies for two runs per GPU).
- Fixer: `OdysseyRenderer/fixer/models/` (link) with `finetuned/h1_b16_e1_model_11001.pkl` (published in
  ADRLAB/odyssey-models) and `base/` (identical to nvidia/Fixer).
- `OdysseyBenchmark/configs/deployment/odysseyzoo-reference.json` pins odyssey_scene001's files, the nuPlan maps,
  the Fixer weights, the OdysseyZoo sources, the four sdroute checkpoints, the DINOv2 and ResNet-34
  backbones, nvdiffrast v0.4.0 and the image identity: `python OdysseyBenchmark/scripts/check_deployment.py [--hashes]`
  (on any host; it reads ODYSSEY_SCENES_ROOT, NUPLAN_MAPS_ROOT, ODYSSEY_MODELS_ROOT and HF_HUB_CACHE,
  and ignores downloads and build output beside the sources).

## Names

The simulator package is `odyssey`, the launcher package `odyssey_runtime`, the planner
bridge `OdysseyBenchmark/odyssey_bridge`; runtime variables are `ODYSSEY_*`; each run writes
`odyssey_output/`.
