# OdysseyZoo

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/ADRLAB/odyssey-models)

End-to-end driving planners prepared for the **Odyssey** closed-loop benchmark.
Each model comes in two versions: the original **baseline** and an **sdroute** version that
replaces NAVSIM's 4-way `driving_command` with a standard-definition (SD) route.

- One folder per model under `models/`, vendored from its upstream repo.
- Upstream code is changed only on the SD-route path, plus a few infra fixes listed in `NOTICE`.
- Every checkpoint is trained on NAVSIM `navtrain`. The `sdroute` checkpoints are trained by us (so are the DiffusionDrive and SafeDrive baselines); the LTF, DrivoR and ReCogDrive baselines are the upstream releases. The [model card](https://huggingface.co/ADRLAB/odyssey-models) lists the source of each checkpoint.
- Every model is checked open-loop on `navtest` (PDMS), then run closed-loop in Odyssey.
- One route builder (`sdroute/`) feeds every model, so they all see the same route tensor.

## Models

| Model | Upstream | Folder | `<model>` |
|---|---|---|---|
| LTF (camera-only TransFuser) | [autonomousvision/navsim](https://github.com/autonomousvision/navsim) | `models/navsim` | `ltf` |
| DrivoR | [valeoai/DrivoR](https://github.com/valeoai/DrivoR) | `models/DrivoR` | `drivor` |
| DiffusionDrive | [hustvl/DiffusionDrive](https://github.com/hustvl/DiffusionDrive) | `models/DiffusionDrive` | `diffusiondrive` |
| SafeDrive | [SPA-junghokim/SafeDrive](https://github.com/SPA-junghokim/SafeDrive) | `models/SafeDrive` | `safedrive` |
| ReCogDrive | [xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive) | `models/recogdrive` | `recogdrive` |

Every model names its two versions the same way:

| Version | What it is | Agent config | Checkpoint |
|---|---|---|---|
| `baseline` | the upstream model: `driving_command` in, no route (DiffusionDrive: camera-only, `latent: True`; SafeDrive: its camera-only, pedestrian-detecting variant) | `<model>_baseline_agent` | `ckpts/<model>_baseline.ckpt` |
| `sdroute` | `driving_command` removed; the SD route is the only navigation signal (ReCogDrive: one exception in its VLM prompt, below) | `<model>_sdroute_agent` | `ckpts/<model>_sdroute.ckpt` |

- Checkpoints live in `models/<Model>/ckpts/` and are not in git. They are published on Hugging Face, [ADRLAB/odyssey-models](https://huggingface.co/ADRLAB/odyssey-models), in the same layout:
  ```bash
  hf download ADRLAB/odyssey-models --include "models/*" --local-dir .   # run inside OdysseyZoo/
  ```
- **How the route enters (LTF, DrivoR, DiffusionDrive, SafeDrive):**
  - The 120 route points are grouped into 24 segment tokens, each covering 5 points (~5 m).
  - Each decoder layer gets one gated cross-attention over those tokens, placed after the memory cross-attention and before the FFN.
  - The gate is zero-initialised, so the route branch starts as an exact no-op.
- **ReCogDrive enters it differently:**
  - The route becomes one token added to the planner's ego-status conditioning.
  - The VLM prompt's navigation line is written from the same route as text (the next maneuver and its distance).
  - Only when the route is too short to describe (fewer than 10 valid points, or under 10 m) does that line fall back to the log's driving command. The planner never receives the command.
  - An extra checkpoint, `recogdrive_sdroute_il.ckpt`, is the same model after imitation learning only (before GRPO). It loads with `recogdrive_sdroute_agent` too.

## Setup

1. **Data** (NAVSIM layout):
   ```bash
   export OPENSCENE_DATA_ROOT=/path/to/navsim      # navsim_logs/{trainval,test}, sensor_blobs/{trainval,test}
   export NUPLAN_MAPS_ROOT=/path/to/navsim/maps    # must be writable, see Notes
   ```
   - SafeDrive reads `DATA_ROOT` instead. Its maps are `$DATA_ROOT/maps`.
2. **Environments:**

   | Env | Models | How to install |
   |---|---|---|
   | navsim | LTF, DiffusionDrive, SafeDrive | `models/navsim/docs/install.md`, then each model's `requirements.txt`. SafeDrive also needs mmcv / mmdet / spconv (`models/SafeDrive/docs/install.md`). |
   | own env | DrivoR | `models/DrivoR/README.md` |
   | own env | ReCogDrive | `models/recogdrive/docs/Installation.md` |

   - Each model's scripts put its own `navsim/` first on `PYTHONPATH`, so several models can share one env.
   - These environments are for training and open-loop evaluation. In the Odyssey closed loop, every shipped model except ReCogDrive runs in the one planner interpreter `ODYSSEY_PLANNER_PY` (DrivoR included); ReCogDrive uses `RECOGDRIVE_PLANNER_PY` ([docs/installation.md](../docs/installation.md)).
3. **Weights that are not in git:**
   - **DrivoR:** DINOv2 ViT-S goes in `models/DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/`. If it is missing, `smoke_env.sh` prints the download command.
   - **DiffusionDrive:** `timm/resnet34.a1_in1k`, from the HF hub or the local HF cache.
   - **ReCogDrive:** the VLM goes in `models/recogdrive/ckpts/ReCogDrive-VLM-2B/`. It is part of the checkpoint download above.

## Run

Run each script from inside its model folder. `<arm>` is `baseline` or `sdroute`.

| Model | Train cache | Eval metric cache | Train | Eval (navtest PDMS) |
|---|---|---|---|---|
| LTF | `cache_ltf.sh` | `cache_metric_ltf.sh` | `train_ltf.sh <arm>` | `eval_ltf.sh <arm>` |
| DrivoR | `cache_drivor.sh`, `cache_metric_drivor.sh train` | `cache_metric_drivor.sh eval` | `train_drivor.sh <arm>` | `eval_drivor.sh <arm>` |
| DiffusionDrive | `cache_diffusiondrive.sh` | `cache_metric_diffusiondrive.sh` | `train_diffusiondrive.sh <arm>` | `eval_diffusiondrive.sh <arm>` |
| SafeDrive | `cache_safedrive.sh` | `cache_metric_safedrive.sh` | `train_safedrive.sh <arm>` | `eval_safedrive.sh <arm>` |
| ReCogDrive | `scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh`; sdroute: the 4 steps in the header of `scripts/sdroute/run_sdroute_cache.sh` | `scripts/cache_dataset/run_metric_caching.sh` | `scripts/training/` | `scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b.sh` (below) |

- **Eval checkpoint:** an optional argument after `<arm>`. The default is `ckpts/<model>_<arm>.ckpt`.
- **Smoke run first** (LTF, DrivoR, DiffusionDrive): `train_*.sh <arm> smoke` runs one batch (`fast_dev_run`). Do this before a full run.
- **SafeDrive:**
  - Only phase-3 checkpoints ship; `train_safedrive.sh` runs phase 1 → 2 → 3, and `baseline` / `sdroute` share one phase-1 run.
  - Besides `baseline` / `sdroute` it also takes `paper` (the upstream camera + LiDAR model) and `camonly` (camera-only, detecting vehicles only); no checkpoint of those ships here.
  - Details: `models/SafeDrive/docs/train_eval.md`.
- **ReCogDrive eval:** the script takes no arguments; edit its overrides.
  - Set: `agent=recogdrive_<arm>_agent agent.checkpoint_path=ckpts/recogdrive_<arm>.ckpt agent.vlm_path=ckpts/ReCogDrive-VLM-2B agent.cache_hidden_state=False`.
  - `sdroute` also needs `agent.sdroute_cache_path=<navtest route cache>`, built by `scripts/sdroute/run_sdroute_cache.sh` with the navtest arguments in its header.
- **Dispatchers:** `bash train.sh <ltf|drivor> <arm> [smoke]` and `bash cache.sh ...` in this folder forward to the LTF and DrivoR scripts.

## SD route

**What the model receives:**

| Key | Shape | Content |
|---|---|---|
| `route_centerline` | (120, 5) float32 | `[x, y, dx, dy, heading]` in the ego frame, in metres, with 1 m spacing out to 120 m. `(dx, dy)` is the step to the next point. |
| `route_centerline_mask` | (120,) bool | Valid points. With no route, the tensor is all zeros and the mask is all False. |

**Where it comes from:**
- `sdroute/` is the only builder. Every model calls it through `navsim/agents/sdroute/route_target.py`, via `build_sdroute_target(scene, frame_idx)`.
- **Pipeline:**
  1. Take the nuPlan HD route centerline, walked along the GT ego path.
  2. Map-match it onto the OSM road graph (`sdroute/assets/osm_raw_<city>.json`).
  3. Resample at 1 m.
- **Open-loop** (navtrain / navtest): the route is built from the NAVSIM scene. It uses the GT future to know which way the ego went, the way a navigation app knows the destination.
- **Odyssey closed-loop:** the simulator supplies the same two tensors, so the model code does not change.

## Adding a new model

LTF (`models/navsim/navsim/agents/transfuser/`) is the smallest reference diff.

1. **Vendor** the upstream repo as `models/<Model>/`, with its own `navsim/` tree and LICENSE.
   - Name things like the other models: `<model>_baseline_agent` / `<model>_sdroute_agent`, `ckpts/<model>_{baseline,sdroute}.ckpt`, and `eval_<model>.sh <arm> [ckpt]`.
2. **Copy** `navsim/agents/sdroute/` from LTF, DrivoR, DiffusionDrive or SafeDrive and leave it unchanged. All copies are identical (`md5sum models/*/navsim/agents/sdroute/*.py`), and they must stay that way.
3. **Config flags:**
   ```python
   use_sdroute: bool = False         # SD-route arm on/off
   drop_driving_command: bool = False   # True in the SD-route arm
   sdroute_num_heads: int = 8
   sdroute_init_seed: int = 0           # route weights come from a forked RNG, so the other weights keep their init
   ```
4. **Target builder:** when `use_sdroute` is on, add `route_centerline` and `route_centerline_mask` from `build_sdroute_target(scene, num_history_frames - 1)`.
   - This is a target, not a feature, because it needs `scene.map_api`.
5. **Model:** replace `nn.TransformerDecoder` with `SDRouteDecoder(...)` and call `decoder(query, keyval, route, route_mask)`.
   - For a custom decoder, add one `SDRouteSegmentEncoder` and one `SDRouteCrossAttention` per layer, placed after the memory cross-attention and before the FFN.
   - `assert_matches_stock()` checks that a layer with no route equals `nn.TransformerDecoderLayer`.
6. **Eval path (easy to miss):** `compute_trajectory()` calls `forward(features)` without targets, so by default the route never arrives. The model then plans route-blind and raises no error.
   - Fix it by setting `requires_scene=True` when `use_sdroute` is on.
   - Also override `compute_trajectory(agent_input, scene)` to build the route exactly as in step 4.
7. **Drop the command** in the SD-route arm. NAVSIM's `status_feature` is `[command(4), vel(2), acc(2)]`, so use `[..., 4:]` and shrink the status encoder to match.
8. **Train on navtrain:**
   - Build one cache with the SD-route config and use it for both arms.
   - Keep lr, batch, epochs and seed the same across arms; only `agent=` changes.
9. **Check before trusting a score:**
   - The smoke run passes.
   - The eval CSV has the expected number of valid tokens.
   - `route_centerline_mask.any()` is True for almost every token.
10. **Odyssey closed-loop:** the model must output a NAVSIM `Trajectory` of 8 poses at 0.5 s (4 s). Wire it in on the simulator side with an agent config, as described in [docs/porting.md](../docs/porting.md).

## Layout

```
OdysseyZoo/
├── sdroute/              SD-route builder (the only copy)
│   └── assets/           OSM road graphs for the 4 nuPlan cities
├── models/
│   ├── navsim/           LTF
│   ├── DrivoR/
│   ├── DiffusionDrive/
│   ├── SafeDrive/
│   └── recogdrive/
├── train.sh, cache.sh    dispatchers (LTF, DrivoR)
└── NOTICE                upstream licenses and what changed here
```

## Notes

- **`NUPLAN_MAPS_ROOT` must be writable.** nuPlan opens `map.gpkg` in SQLite WAL mode, so a read-only copy fails with `attempt to write a readonly database`. If the dataset is shared, copy the maps.
- **Check the valid-token count after every eval.** `run_pdm_score` catches per-token errors and logs them instead of stopping, so a partial run can look complete.
- **DiffusionDrive and ReCogDrive sample noise without a seed**, so PDMS moves between runs. Compare several runs, not one.
- **Multi-GPU DDP on some Blackwell hosts** can die at the first barrier with `illegal memory access`. Set `export NCCL_ALGO=Ring`.
- **License:** each `models/<Model>/` keeps its upstream license. See `NOTICE`.
- **Checkpoint licenses:** checkpoints trained by us are CC BY-NC-SA 4.0; the upstream releases keep their own licenses. See the license column of the [model card](https://huggingface.co/ADRLAB/odyssey-models).
