# The benchmark

Closed-loop evaluation of an end-to-end driving planner in reconstructed nuPlan scenes:
exported scene assets → Gaussian rendering → Fixer image restoration → the planner's own native
preprocessing and inference → vehicle and traffic simulation → episode scoring. The planner
receives rendered camera images, its ego state and navigation inputs, and returns a
trajectory; a fixed controller tracks it.

## Scene set

- 100 scenes, published as `ADRLAB/odyssey-scenes` (`scenes.csv`: `odyssey_scene001` …
  `odyssey_scene100`). The scene set is fixed; the
  benchmark denominator is 100 per traffic mode.
- Mini set (`--scenes mini`, `OdysseyBenchmark/scripts/run_evaluation_mini.sh`): 30 scenes at the full horizon for
  development — odyssey_scene 003, 007, 008, 009, 010, 015, 017, 018, 025, 028, 029, 033, 035,
  036, 037, 049, 051, 063, 064, 067, 068, 069, 076, 078, 083, 084, 086, 093, 096, 099. A mini-set
  number is labelled as such by the merger and is not comparable with the 100-scene number.
- Two traffic modes per scene: **nr** (non-reactive: other vehicles replay the log, led by the
  sector ahead of the ego) and **r** (reactive: other vehicles follow an IDM policy). They are
  reported as separate sheets; the merger can also print their mean, which is not a benchmark number.
- Fixed per-scene rules (`OdysseyBenchmark/odyssey_runtime/launch.py`): the reactive spawn gate is disabled for
  odyssey_scene041, 045 and 075; the benchmark's signal timetable
  (`OdysseyBenchmark/data/tlc_timetable/`) applies to 13 scenes (odyssey_scene011,
  021, 025, 032, 052, 056, 060, 061, 067, 072, 075, 081, 086), together with the scene's own
  traffic-light labels; in the other scenes the labels are attached when the scene has signal nodes
  (except odyssey_scene048, 053 and 099).
  Every run records the rules that applied in its job record (`applied_rules`).
- World cadence 0.1 s, 1.5 s warm-up, horizon = twice the exported appearance horizon: the source log is
  100 s (1,000 frames), so an episode runs at most 200 s.
  `--max-steps` bounds a run for smoke tests; the benchmark runs the full horizon.
- An episode ends by **route deviation** when the ego is 30 m from the SD route, or by
  **destination arrival** when it has covered 99 % of the route and is within 10 m of its end
  (5 m of the logged end for a scene without a usable route), or by **route exhaustion** when the
  ego has driven past the end of the SD route, so a model that reads the route has nothing left
  to plan on; otherwise at the time limit.
  These values are fixed in `OdysseyTrafficAgent/defines.py` and are part of the score.

## What the planner gets and must return

- Cameras: a subset of the nuPlan eight-camera rig (`CAM_F0 CAM_L0 CAM_R0 CAM_L1 CAM_R1 CAM_L2
  CAM_R2 CAM_B0`) at the rig's native 1920×1080, at the past times the agent config declares.
  Only the declared cameras are rendered. No LiDAR.
- Ego history: state, acceleration (with the derived lateral term), driving command, at
  0.5 s spacing over the declared window.
- Navigation: the log-derived driving command and/or the SD route centerline
  (`route_centerline` (1,120,5) with a mask, 120 points over 120 m), as declared.
- Output: `(N, 3)` future ego poses (x, y, heading) in the ego frame every `plan_dt` seconds
  (0.5 s for the shipped models); the runtime resamples to 0.1 s for the `two_stage_controller`.
  The controller is part of the benchmark and cannot be replaced.

## Score

Per scene (paper Appendix C.3): `RouteDS = 100 · RC · P_SD · P_col · P_off · P_TL · P_PLC`
(`OdysseyBenchmark/odyssey_benchmark/ds_formula.py`).

| term | meaning |
|---|---|
| RC | fraction of the SD route completed (HMM map-matching of the driven trajectory) |
| P_SD | 0 when the episode ended by leaving the route (30 m) or map-matching finds travel outside it, else 1 |
| P_col | per at-fault collision: 0.5 pedestrian, 0.6 vehicle/bicycle |
| P_off | 1 − (distance driven off drivable area or against traffic) / distance driven |
| P_TL | 0.7 per designated intersection entered against a red light: the whole car past the stop line and within 4 m to either side of a red connector, touching it or not (designated signal scenes) |
| P_PLC | 0.7 per failed pre-lane-change evaluation, 0.9 per late one |

Per sheet (model × traffic mode) the merger reports the paper's table, with SDC, PLCA, PLCS and
Comf. in percent: **RouteDS** (mean over
the 100 scenes), **SDC** (mean P_SD), **PLCA** and **PLCS** (pre-lane-change accuracy and
score: the credit of the sheet's evaluation stop lines, 1 per clean pass and 0.5 per late one,
over all the stop lines reached and over all the stop lines on the routes; pooled over the sheet,
not a mean of the per-scene values), **Eff.** (median
efficiency: 100 = the surrounding traffic's speed) and **Comf.** (share of episodes within every
comfort limit), plus the RouteDS components RC, P_SD, P_col, P_off, P_TL, P_PLC (Table 11).

## Failures

| outcome | counted as |
|---|---|
| the model failed (exception, bad output shape or NaN, checkpoint/config mismatch) | **DS 0** for that scene; the other metrics stay blank |
| the infrastructure failed (renderer, Fixer, simulator, timeout, killed) | retried automatically; if still failing, it blocks the final score and is listed to rerun |
| the episode could not be scored (scoring error, no scoring inputs) | blocks the final score; rerun, or investigate |

The denominator is always 100. A sheet is FINAL only when every scene is complete or a model
failure, and every scored run carries the same rules.

## Reproducibility

- Compare scores computed with the same release of the scoring code; each campaign's
  `manifest.json` records the code commit.
- The renderer and the Fixer are deterministic on one host, so a configuration repeats exactly
  there. Another GPU type, driver or software build renders and restores slightly differently;
  compare runs made on one host and report the GPU.
- `--seed` sets the planner's RNG; without it every planner process starts from PyTorch's
  default seed, so DiffusionDrive's sampled noise repeats as well.
- One scene is not the benchmark: validate on the debug set, iterate on the mini set, report the
  full 100 × {nr, r}.

## What differs from Bench2Drive

| | Bench2Drive | this benchmark |
|---|---|---|
| planner output | control (steer/throttle/brake) | trajectory; fixed controller |
| sensors | agent declares any rig | fixed nuPlan rig, subset of 8 cameras, no LiDAR |
| navigation | global route for every agent | driving command and/or SD route, declared per model |
| background traffic | CARLA traffic manager + scenario actors | log replay (nr) or IDM (r), two sheets |
| score | RC × infraction factors (ped 0.5, veh 0.6, …) | DS above; not comparable numbers |
| failed route | scored on what it drove | model failure = 0, infra failure = rerun |
| parallelism | static split per GPU | one queue, one worker per GPU, resumable |

Do not place numbers from the two benchmarks next to each other.
