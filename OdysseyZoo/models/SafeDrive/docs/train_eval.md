# SafeDrive Training and Evaluation

## Variants

Every script takes the variant as its first argument.

| variant | model | phase-3 agent config | released checkpoint |
| --- | --- | --- | --- |
| `paper` | the paper's model: camera + LiDAR, vehicles | `SafeDrive_Phase3_Planner_FullTrain` | upstream's, see [README](../README.md) |
| `camonly` | camera-only, vehicles | `SafeDrive_Phase3_Planner_FullTrain_CamOnly` | none |
| `baseline` | camera-only, detects pedestrians too | `safedrive_baseline_agent` | `ckpts/safedrive_baseline.ckpt` |
| `sdroute` | `baseline` + SD route, driving command dropped | `safedrive_sdroute_agent` | `ckpts/safedrive_sdroute.ckpt` |

The `baseline` and `sdroute` checkpoints (phase 3 only, not in git) are on
[ADRLAB/odyssey-models](https://huggingface.co/ADRLAB/odyssey-models); `train_safedrive.sh` trains
all three phases.
The agent lives in [`navsim/agents/safedrive/`](../navsim/agents/safedrive); every entrypoint needs an
explicit `agent=...`, there is no default.

**Camera-only.** The paper model reads LiDAR in exactly one place, the BEV encoder's
`lidar_cross_attn`, where the SECOND voxel features are key/value and the BEV query is a learned
embedding. The `_CamOnly` configs set `use_lidar: False` and drop that attention from
`operation_order`, so they differ from the paper model only there (and the SECOND branch is not
built). Keep `+second_lidar=True` anyway: it selects the custom collate, which also decodes a
path-only camera cache and keeps the token lists, not just the lidar loading.

## Caches

```bash
bash cache_safedrive.sh          # training: feature cache, train metric cache, SD route
bash cache_metric_safedrive.sh   # evaluation: navtest metric cache
```

- One feature cache serves every variant. It is built with the paper's phase-3 config; the
  camera-only variants ignore its LiDAR entries, and step 3 adds the route the `sdroute` variant
  reads.
- `CACHE_IMAGES=false` (default) caches image paths, ~75 GB for navtrain, and decodes the images in
  the dataloader (~0.35 s CPU per sample, so keep `NUM_WORKERS` around 20+ per GPU). `true` caches
  uint8 images, ~360 GB. The pixels the model sees are the same either way.
- The train metric cache is the ground truth that phases 2 and 3 roll their own plans against;
  it took ~6 h with 10 workers on a 128-core machine.
- The dataset root is `DATA_ROOT` (default `./dataset`, maps at `$DATA_ROOT/maps`). Training needs
  the navtrain (`trainval`) logs and sensors.

## Training

```bash
bash train_safedrive.sh <variant>     # phase 1 (90 epochs) -> phase 2 (5) -> phase 3 (10)
```

- **Phases.** Phase 1 trains perception only, phase 2 trains the planner and safety heads on frozen
  perception, and phase 3 fine-tunes end to end. Each phase loads the previous phase's `last.ckpt` as
  weights, not as a resume: freezing changes the optimizer's param groups. For `paper`, upstream's
  phase-2 config freezes only the two BEV-segmentation heads; the camera-only phase-2 configs freeze
  all of perception.
- **Shared phase 1.** `baseline` and `sdroute` share one phase-1 run (phase 1 never reads the route),
  as the released pair did. Train one of them first; the other reuses it.
- **Re-running continues.** A finished phase writes `DONE` next to its checkpoints and is skipped. An
  interrupted phase is not retrained over its own output: continue it with
  `RESUME_P<n>=<its last.ckpt>`, or move its directory under `exp/` away.
- **Recipe.** bf16-mixed; global batch 64 for `paper` (upstream's 2 GPUs x 32) and 128 for the
  camera-only variants (4 GPUs x 32), as read off the released phase-3 checkpoints (`camonly` has
  none and follows the other camera-only variants).
  `SAFEDRIVE_DEVICES` sets the GPU count and the per-GPU batch follows.
- **Outputs.** `exp/safedrive_<variant>/phase3_e2e/lightning_logs/checkpoints/`; the released
  checkpoints are phase-3 epoch 9.

## Evaluation

```bash
bash eval_safedrive.sh baseline            # or sdroute; an optional 2nd argument overrides the ckpt
bash eval_safedrive.sh paper <ckpt>        # paper / camonly: no checkpoint ships, pass one
```

Evaluation runs without a feature cache (`cache_path=null`) and reads navtest sensor data directly,
so the navtest metric cache is all it needs.

## What the script does

Stage 1 runs a GPU forward pass over navtest and writes the selected trajectories; stage 2 scores
them with the PDM simulator and writes a CSV (per-token rows plus a final `average` row) under
`$NAVSIM_EXP_ROOT/<EXP>/`.

Stage 1 has to go through `run_training.py`. The code that injects `scoring_test` and the
`*_test_weight` values into the model exists only there -- the model's own `scoring_test` is
hardcoded to `False` in `__init__`, so a Hydra override cannot switch it on and weights handed to
`run_evaluation_gpu.py` are silently ignored.

## Test-time score weights

Candidate trajectories are ranked by a weighted sum of the safety subscores. The weights are
inference-time only -- no retraining is needed to change them. They live in the `WEIGHTS` block of
`eval_safedrive.sh`, one line per metric.

> **`no_EP_TC_sum_scoring` decides the formula** and the two branches do not agree:
> `True` gives `EP_w*log(EP) + TTC_w*log(TTC)`, `False` (the dataclass default)
> gives `W_w*log(EP_w*EP + TTC_w*TTC)`. The weights in `eval_safedrive.sh` are upstream's,
> searched offline for the paper model against the first form, so the script sets the flag on.
> Turning it off changes the ranking and lowers the score.

`eval_safedrive.sh` uses these weights unchanged for every variant. The Odyssey closed-loop runtime
ranks the `baseline` / `sdroute` plans with retuned weights instead
(`projects/OdysseyBridge/planners/safedrive.py` in the Odyssey repository), so open-loop scores from
this script and closed-loop driving use different rankings.
