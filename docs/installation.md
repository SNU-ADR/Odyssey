# Installation

Everything the runtime needs before the first run. The [README](../README.md) gives the short
version; [install-source.md](install-source.md) builds the simulator and planner environments
from source, and [deployment.md](deployment.md) records the reference host's versions and
paths.

The runtime needs three interpreters (simulator, planner, Fixer; ReCogDrive needs a fourth, see
below), the simulator's CUDA build dependencies (gsplat, and
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) cloned into `third_party/nvdiffrast`), the
published weights and scenes, nuPlan maps and the Fixer weights. Build the simulator and planner
environments with [docs/install-source.md](install-source.md).

nvdiffrast is cloned into the checkout and installed into the simulator interpreter (CUDA 12.8 on
`PATH`; a container that already has it installed needs neither step):

```bash
git clone --branch v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast third_party/nvdiffrast
"$ODYSSEY_SIM_PY" -m pip install --no-deps --no-build-isolation ./third_party/nvdiffrast
```

| | what it takes |
|---|---|
| disk | about 300 GB: scenes 217 GB, weights about 11 GB (ReCogDrive's VLM included), interpreters and CUDA 12.8 about 40 GB, 1.4 GB per nuPlan map copy, about 20 GB free per campaign |
| GPU | NVIDIA with CUDA 12.8 drivers; one episode uses about 15-25 GB (simulator 5-16 GB, Fixer about 5.5 GB, planner about 2 GB for the shipped models other than ReCogDrive) |
| time | about one working day to build the interpreters, plus the downloads |
| accounts | Hugging Face (the scenes are gated) and a nuPlan account for the maps |

**OdysseyZoo** (model code and the SD-route builder the scorer imports) is bundled at
`./OdysseyZoo`, the runtime's default `ODYSSEY_ZOO_ROOT`: NAVSIM (with LTF), DrivoR,
DiffusionDrive, SafeDrive and ReCogDrive, each with its licence (`OdysseyZoo/NOTICE`,
`OdysseyZoo/models/*/LICENSE`), and the OpenStreetMap-derived route graphs in
`OdysseyZoo/sdroute/`. SafeDrive's native fix (`OdysseyBenchmark/patches/safedrive-odysseyzoo-v2.patch`) is
already applied there; for a separate OdysseyZoo copy, point `ODYSSEY_ZOO_ROOT` at it and run
`python OdysseyBenchmark/scripts/apply_safedrive_patch.py --repo "$ODYSSEY_ZOO_ROOT/models/SafeDrive"`.

**Weights** — [ADRLAB/odyssey-models](https://huggingface.co/ADRLAB/odyssey-models): baseline
and sdroute checkpoints for LTF, DrivoR, DiffusionDrive, SafeDrive and ReCogDrive in
OdysseyZoo's `models/<Model>/ckpts/` layout, ReCogDrive's VLM, and the Fixer weights
(`OdysseyRenderer/fixer/`, below). Licences: CC BY-NC-SA 4.0 for the checkpoints we trained, Apache-2.0 for the
redistributed upstream files, the NVIDIA Open Model License for the Fixer weights (model card).

```bash
hf download ADRLAB/odyssey-models --include "models/*" --include SHA256SUMS --local-dir /abs/path/odyssey-models
(cd /abs/path/odyssey-models && sha256sum -c --ignore-missing SHA256SUMS)
export ODYSSEY_MODELS_ROOT=/abs/path/odyssey-models   # unset: ./OdysseyZoo, if you download into it instead
export RECOGDRIVE_VLM_PATH=$ODYSSEY_MODELS_ROOT/models/recogdrive/ckpts/ReCogDrive-VLM-2B   # ReCogDrive only
export RECOGDRIVE_PLANNER_PY=/abs/path/recogdrive/bin/python   # ReCogDrive only (below)
```

**ReCogDrive** (only for `OdysseyBenchmark/agents/recogdrive*.yaml`): its code needs Python >= 3.10 and
transformers 4.37.2, so its configs run in their own interpreter, `RECOGDRIVE_PLANNER_PY`, not
`ODYSSEY_PLANNER_PY`: the planner environment's package list reinstalled on Python 3.10, plus
ReCogDrive's own packages:

```bash
P=/abs/path/recogdrive/bin/python
conda create -y -p /abs/path/recogdrive --override-channels -c conda-forge python=3.10 pip
"$P" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$ODYSSEY_PLANNER_PY" -m pip freeze \
  | grep -vE '^(torch=|torchvision|gsplat|mmcv|nuplan-devkit|diffusers|huggingface[-_]hub|timm|nvidia-|#|-e )' \
  | grep -v '@ file' > recogdrive_requirements.txt
"$P" -m pip install --no-deps -r recogdrive_requirements.txt
"$P" -m pip install --no-deps setuptools==65.5.1   # pip freeze omits it; lightning needs pkg_resources
"$P" -m pip install --no-deps "git+https://github.com/motional/nuplan-devkit/@ce3c323af01c0d7ec5672f7832ef53f9c679aab0"
"$P" -m pip install --no-deps transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 \
  huggingface_hub==0.36.0 accelerate==0.27.2 peft==0.10.0 diffusers==0.30.3 timm==0.9.12 \
  decord==0.6.0 einops-exts click==8.2.1
export RECOGDRIVE_PLANNER_PY="$P"
python -m odyssey_runtime check --agent OdysseyBenchmark/agents/recogdrive_sdroute.yaml   # VLM, checkpoint and camera load
```

diffusers, huggingface_hub and peft are pinned to versions that work with transformers 4.37.2.
FlashAttention2 is optional (InternVL falls back to eager attention). ReCogDrive runs its 2B VLM
every step, so a 2,000-step episode takes about 30 minutes. The three configs share this
interpreter and the VLM: `recogdrive_baseline` takes the driving command, `recogdrive_sdroute`
the SD route (in the planner and as text in the VLM prompt), and `recogdrive_sdroute_il` is the
same model before GRPO.

Backbone initialisation weights are separate: DrivoR needs DINOv2 ViT-S at
`OdysseyZoo/models/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`;
DiffusionDrive and SafeDrive load `timm/resnet34.a1_in1k` from the Hugging Face cache
(`HF_HOME`), or from a file named in their config's `overrides`.

```bash
hf download timm/vit_small_patch14_reg4_dinov2.lvd142m \
  --local-dir OdysseyZoo/models/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m   # DrivoR
hf download timm/resnet34.a1_in1k                                     # DiffusionDrive, SafeDrive (into HF_HOME)
```

**Scenes** — [ADRLAB/odyssey-scenes](https://huggingface.co/datasets/ADRLAB/odyssey-scenes):
one folder per scene (reconstruction, scenario, road surface, route, traffic-light labels,
checkpoint inventory). Use the download as it is: every file is read in place by its path,
and scenes are named as published (`odyssey_scene001`, `scene001` or `001`). The dataset is
gated: accept its terms on the dataset page, then log in (`hf auth login`) before downloading;
without that the download fails with 401.

```bash
hf download ADRLAB/odyssey-scenes --repo-type dataset --local-dir /abs/path/odyssey-scenes
(cd /abs/path/odyssey-scenes && md5sum -c MD5SUMS)
export ODYSSEY_SCENES_ROOT=/abs/path/odyssey-scenes
```

To try the pipeline first, download only the three scenes of the debug run plus odyssey_scene001,
which the checks and the first test episode use (about 12 GB), and add the rest later into the same
directory; `OdysseyBenchmark/tools/check_env.py` reports the scenes still missing:

```bash
hf download ADRLAB/odyssey-scenes --repo-type dataset --local-dir /abs/path/odyssey-scenes \
  --include scenes.csv --include MD5SUMS --include "odyssey_scene001/*" \
  --include "odyssey_scene002/*" --include "odyssey_scene011/*" --include "odyssey_scene075/*"
```

**Fixer** (image restoration, run in torch): its own interpreter ([OdysseyRenderer/fixer/README.md](../OdysseyRenderer/fixer/README.md))
and weights, fetched into this checkout:

```bash
hf download ADRLAB/odyssey-models --include "fixer/*" --local-dir OdysseyRenderer
hf download nvidia/Fixer --include "base/*" --local-dir OdysseyRenderer/fixer/models
(cd OdysseyRenderer/fixer/models && sha256sum -c SHA256SUMS)
```

**Maps, shell.** The nuPlan maps (`nuplan-maps-v1.0.json` and the four cities' `map.gpkg`) come
from `nuplan-maps-v1.1.zip` of the [nuPlan download](https://www.nuscenes.org/nuplan#download),
which unpacks to `nuplan-maps-v1.0/`; the archive named `nuplan-maps-v1.0.zip` has a different
Boston `map.gpkg`. `OdysseyBenchmark/configs/deployment/odysseyzoo-reference.json` lists the expected sha256. Each
concurrent run needs its own copy (`final_gpu0`, `final_gpu1`, …; `NUPLAN_MAPS_ROOT` names one
and the batch runner derives the rest by GPU index):

```bash
bash OdysseyBenchmark/scripts/make_map_slots.sh /abs/path/nuplan-maps-v1.0 /abs/path/maps 0,1,2,3   # final_gpu0..3, one copy per GPU
export NUPLAN_MAPS_ROOT=/abs/path/maps/final_gpu0
```

The first argument is the unpacked `nuplan-maps-v1.0/` directory (it holds `nuplan-maps-v1.0.json`).

**Shell.** Copy `OdysseyBenchmark/scripts/host_env.example.sh`, set the five paths in its first block, and source
it in every shell that runs Odyssey (the evaluation scripts also take it as `HOST_ENV=<file>`).
It reports paths that do not exist yet. Then check the host:

```bash
cp OdysseyBenchmark/scripts/host_env.example.sh ~/odyssey_env.sh      # edit the first block
source ~/odyssey_env.sh
python OdysseyBenchmark/tools/check_env.py --gpus 0,1,2,3
```

`check_env.py` checks the interpreters (and that torch sees a GPU in each), nvdiffrast in the
simulator interpreter, nvcc 12.8, the
environment variables the launcher refuses, OdysseyZoo, the 100 scenes, the map slots, the GPUs
(inside `CUDA_VISIBLE_DEVICES`, when that is set) and the Fixer weights. It does not build a
model: `python -m odyssey_runtime check --agent <config>` does that, including the backbone
weights each model loads ([porting.md](porting.md)).
