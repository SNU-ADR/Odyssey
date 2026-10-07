# Getting started

From an installed host to the 100-scene sheet. Install first ([installation.md](installation.md)).
Run every command from the repository root, after sourcing your environment file (the `Shell`
section of [installation.md](installation.md); it sets `PYTHONPATH` to the component folders).

## The steps

```bash
python OdysseyBenchmark/tools/check_env.py --agent OdysseyBenchmark/agents/ltf_sdroute.yaml                         # 0. host, assets, interpreters
python -m odyssey_runtime check --agent OdysseyBenchmark/agents/ltf_sdroute.yaml                                  # 1. config, model build, one synthetic step (--device cpu: no GPU)
AGENT_CONFIG=OdysseyBenchmark/agents/ltf_sdroute.yaml GPUS=0       bash OdysseyBenchmark/scripts/run_evaluation_debug.sh   # 2. three scenes, short
AGENT_CONFIG=OdysseyBenchmark/agents/ltf_sdroute.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_mini.sh    # 3. mini set: 30 scenes x {nr, r}
AGENT_CONFIG=OdysseyBenchmark/agents/ltf_sdroute.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_multi.sh   # 4. the benchmark: 100 scenes x {nr, r}
python OdysseyBenchmark/tools/merge_results.py -f experiments/simulation/eval_ltf_sdroute                        # 5. the sheet
```

The commands run the shipped LTF (SD route) config. To evaluate your own model, replace
`OdysseyBenchmark/agents/ltf_sdroute.yaml` with its agent config; its results then go to
`experiments/simulation/eval_<model.name>`.

- Step 2 runs three scenes for 40 steps each. Its RouteDS says nothing about the model; it only
  checks that the pipeline runs end to end.
- Step 4 runs 200 jobs from one queue. Running the same command again resumes an interrupted
  campaign.
- Outputs go to `experiments/simulation/eval_<model>[_debug|_mini]/`.
- The merger prints one sheet per traffic mode and writes `runs.csv`, `summary.csv` and
  `merged.json`.

## Port your model

A model is one YAML file. The shipped configs under `OdysseyBenchmark/agents/` are the examples:

```yaml
model:
  name: my_model
  python: /envs/my_model/bin/python           # the model's own interpreter
  repo: /work/my_navsim_fork                  # a navsim-style repository
  agent_config: my_agent                      # <repo>/navsim/planning/script/config/common/agent/my_agent.yaml
  checkpoint: /work/ckpts/epoch_99.ckpt
  overrides: []                               # Hydra overrides
history_times_s: [-1.5, -1.0, -0.5, 0.0]
camera_times_s: {CAM_F0: [0.0], CAM_L0: [0.0], CAM_R0: [0.0]}   # which of the 8 rig cameras, at which past times
navigation: {driving_command: true, sd_route: none}             # what the model reads; checked at start-up
output: {plan_dt: 0.5, shape: [8, 3]}                           # N poses every plan_dt seconds
```

A NAVSIM `AbstractAgent` runs from this config alone. Anything else needs a small subclass,
referenced by `model.adapter` (`OdysseyBenchmark/agents/template_adapter.py`). See
[porting.md](porting.md).

## Check the model first

```bash
python -m odyssey_runtime check --agent my_model/agent.yaml               # on GPU 0 (or --gpu N)
python -m odyssey_runtime check --agent my_model/agent.yaml --device cpu  # without a GPU
```

`check` builds the model in its interpreter and runs one synthetic step through the same code
a run uses. It prints the feature shapes the model receives and its plan, and fails when the
plan does not change with the SD route the model is declared to read. A model that fails here
would score 0 on every scene ([porting.md](porting.md)).

## Debug run

```bash
AGENT_CONFIG=my_model/agent.yaml GPUS=0 bash OdysseyBenchmark/scripts/run_evaluation_debug.sh
```

`GPUS` lists physical GPU indices, as `nvidia-smi` shows them. When `CUDA_VISIBLE_DEVICES` is
exported (a scheduler's allocation, for example), the listed GPUs must be among its entries.
Relative `AGENT_CONFIG`, `SAVE_PATH` and `HOST_ENV` paths are taken from the current directory.

Three scenes, 40 steps each: odyssey_scene011 nr (signal timetable), odyssey_scene075 r
(reactive spawn gate off), odyssey_scene002 nr (plain). A passing job ends `complete` with a
scene row in its CSV and no scoring error; the campaign directory (`experiments/simulation/eval_<model>_debug/`) holds every log.
The 40-step RouteDS says nothing about the model: odyssey_scene011 and odyssey_scene075 start
from standstill, so the ego covers only a few metres in 4 s and RouteDS can be 0 (RC 0, P_SD 0).
`python -m odyssey_runtime run --agent … --scene odyssey_scene001 --react nr --gpu 0 --max-steps 40`
runs one episode by hand (`--dry` prints the command, `--record` keeps images and plans,
`--audit` hashes every model tensor per step).

## Evaluate

```bash
AGENT_CONFIG=my_model/agent.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_mini.sh    # 30 scenes x {nr, r}
AGENT_CONFIG=my_model/agent.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_multi.sh   # 100 scenes x {nr, r}
```

Two (or more) episodes per GPU need one map copy per concurrent run; `make_map_slots.sh` makes
them and prints the list in the order the runner assigns them:

```bash
MAP_SLOTS=$(bash OdysseyBenchmark/scripts/make_map_slots.sh /abs/path/nuplan-maps-v1.0 /abs/path/maps 0,1,2,3 2)   # 8 copies
AGENT_CONFIG=my_model/agent.yaml GPUS=0,1,2,3 RUNS_PER_GPU=2 MAP_SLOTS=$MAP_SLOTS \
  bash OdysseyBenchmark/scripts/run_evaluation_multi.sh
```

The mini set (`--scenes mini`: 30 scenes, `OdysseyBenchmark/odyssey_runtime/eval.py: MINI_SET`, at the full
horizon, campaign `eval_<model>_mini/`) is a quick, comparable development number; the
merger labels it `mini set, not the benchmark`. The benchmark is the full set: 200 jobs
(100 scenes × nr, r) from one queue, one worker per GPU, each with its own map
slot. The full scene horizon takes about 10-35 minutes per episode with one run per GPU, so the
benchmark is about 50 GPU-hours (half a day on four GPUs). `RUNS_PER_GPU=2` (eval
`--runs-per-gpu 2`, with `MAP_SLOTS` listing one map directory per concurrent run) runs two
episodes per GPU; on 48 GB GPUs the shipped models fit two per GPU. Running the same command again
continues an interrupted campaign: complete jobs are skipped, infrastructure failures
(renderer, Fixer, simulator, timeout) are retried (`RETRY_INFRA`, default 2), model failures
stand unless `--retry-failed`. Attempts count across invocations: a job whose infrastructure
retries are used up stays failed when the command is repeated; raise `RETRY_INFRA` to give it
more attempts without re-running model failures. The runner stops after five failed jobs in a
row (`--max-consecutive-failures`), which usually means the model or the host is broken.
Run one `--react` list per campaign directory (the default `nr,r` runs both modes): a campaign
refuses a different list, model, checkpoint or option. A simulator that has written its result but does not exit
within two minutes is stopped and its job counts as complete. Every job leaves a record with
its failure class and reason
(`jobs/`), and `manifest.json` records the scene set, the config, checkpoint identity, code
commit, host, GPUs and the Fixer checkpoint. Variables: `GPUS`, `REACT`, `MAX_STEPS`, `SEED`,
`TIMEOUT`, `RETRY_INFRA`, `MAP_SLOTS`, `RUNS_PER_GPU`, `SAVE_PATH`;
`python -m odyssey_runtime eval --help` lists the rest. Three of them run nothing: `--dry`
prints the job table, the GPU-to-map-slot assignment and the first job's command; `--status`
prints the campaign's job table (pass the same `--scenes`); `--preflight full` composes every
job's run before starting (about 10 s for 200 jobs). Progress goes to `eval.log` in the campaign
directory.

## Results

```bash
python OdysseyBenchmark/tools/merge_results.py -f experiments/simulation/eval_my_model
```

Writes `runs.csv` (one row per scene × mode), `summary.csv` / `summary.json` (one row per
mode) and `merged.json`, and prints:

```
sheet                        n  RouteDS    SDC   PLCA   PLCS    Eff.  Comf.  status
my_model_nr            100/100    41.23   95.0   71.0   62.0    77.7   96.0  FINAL
my_model_r             100/100    38.50   …                                  FINAL
```

RouteDS is the mean of `100 · RC · P_SD · P_col · P_off · P_TL · P_PLC` over the 100 scenes
(the paper's definition; the components are in `summary.csv`); a scene the **model** failed on
counts 0. SDC, PLCA, PLCS and Comf. are in percent. Each run also leaves its own result row, `odyssey_output/routeds_<NR|R>.csv`, filed under the published
scene name. An infrastructure failure, a timeout or an
unscored episode blocks the sheet (it is printed, exit code 1, and the jobs to rerun are
listed); `--allow-incomplete` prints a provisional number instead. nr and r are separate
sheets. Column definitions, the failure policy and the differences from Bench2Drive:
[docs/benchmark.md](benchmark.md).

## Reproducibility

- Runs repeat exactly on one host: the same configuration gives the same images, plans and
  scores. Another GPU type, driver or software build renders and restores slightly differently,
  and closed-loop driving amplifies that, so compare runs made on one host (and report the GPU).
- Compare scores computed with the same release of the scoring code (`manifest.json` records
  the code commit); `rollout_trajectory.npz` holds the inputs a run was scored from.
- `--seed` sets the planner's RNG. Without it every planner process starts from PyTorch's default
  seed, so DiffusionDrive's sampled noise repeats too.
- The launcher refuses to start while an `ODYSSEY_*` environment variable is set that is neither in
  `USER_ENV` (install locations, timing and diagnostic settings) nor one the launcher sets itself
  (`LAUNCHER_ENV`; both in `OdysseyBenchmark/odyssey_runtime/launch.py`): such a variable could
  change the run without showing in the result.

