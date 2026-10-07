# Results format

## Campaign directory (`python -m odyssey_runtime eval`)

```
<campaign>/
  manifest.json              what was run, where, and the state of every job (rebuilt after each job)
  inputs/agent.yaml          copy of the agent config
  inputs/profile.json        the effective runtime profile derived from it
  inputs/scenes.csv          the scene table used
  jobs/<scene>_<react>.json  one record per job
  runs/<scene>_<react>/attempt_NN/   the launcher's output for each attempt
  eval.log
  runs.csv  summary.csv  summary.json  merged.json     written by OdysseyBenchmark/tools/merge_results.py
```

### Job record (`odyssey_eval_job/1`)

| field | meaning |
|---|---|
| `status` | `pending`, `running`, `complete`, `model_fail`, `infra_fail`, `timeout`, `input_missing` |
| `final` | no further attempt will be made |
| `attempt`, `max_attempts`, `run_dir` | the attempt the record describes |
| `attempts[]` | per attempt: status, `failure_class` (`model` / `infra` / `input`), `kind`, `reason`, `returncode`, `exit_code`, `flag`, `timed_out`, `stopped_after_finish` (the simulator wrote its result but had to be stopped on exit), timestamps, `duration_s`, `gpu`, `worker` (`<gpu>` or `<gpu>.<k>` with several runs per GPU), `map_slot`, `code_commit`, `last_step`, `log`, `log_tail`, `pruned` |
| `result` | (complete only) `scene`, `RouteDS`, `term_reason`, `steps`, `RC`, `P_SD`, `scoring_error`, `tl_set`, `plc_rule`, `csv` — from the run's result row |
| `failure` | (failed only) `class`, `kind`, `reason`, `log`, `log_tail` |
| `applied_rules` | `gate_off`, `tlc_timetable_set`, `tl_control_path` as applied by the launcher |

Failure kinds: model — `planner_startup`, `planner_exception`, `planner_died`, `bad_plan`,
`profile_mismatch`; infra — `planner_timeout`, `planner_startup_env`, `fixer`, `simulator`,
`signal`, `wall_clock`, `scoring_error`, `runtime_contract`, `no_report`, `interrupted`;
input — `input_missing`.

### Manifest (`odyssey_eval_manifest/1`)

`campaign` (name, directory, fingerprint of agent config + checkpoint + traffic modes + options, times,
argv), `benchmark` (scenes root and its csv hash, reacts, `job_table` of every planned job,
max_steps, seed, timeout, retries, denominator, `restorer`), `model` (name, agent config path and hash,
profile summary, python, repo and its commit, hydra agent config, adapter, checkpoint path,
size and head hash), `environment` (host, GPUs, runs per GPU, map slots, interpreters, the
Fixer preset and its checkpoint size and head hash, code commit, key variables), `summary` (counts per status, mean DS over the fixed
denominator) and `jobs[]` (one line per job record).

Running the same `eval` command again continues the campaign. A different agent config,
checkpoint, `--react` list (in its order), `--max-steps`, `--seed` or `--record` is refused for
the same directory: run both traffic modes in one campaign (the default `nr,r`), or give each its
own `--output`. The scene selection is not part of the fingerprint; the default directory name
already separates the debug, mini and full sets.

## Per-run output (the launcher)

```
launch.json                              the command, scene, checkpoint, agent and restorer of the run
exit_code.txt  simulation.log            0 = success (see below); the simulator's whole log
runtime/profile.json  runtime/timings.jsonl  runtime/planner_worker.log  runtime/fixer_worker_<pid>.log
odyssey_output/simulation_completed.flag
odyssey_output/runner_report.json        the simulator's own success/error report
odyssey_output/routeds_{NR|R}.csv        the run's result row, below
odyssey_output/rollout_trajectory.npz    the inputs the run was scored from
odyssey_output/sensor_blobs/<scene>/CAM_*/   camera images the model received, after the Fixer (--record; deleted after a successful batch job unless --keep-outputs)
```

The per-scene rules a run was launched with are part of the command in `launch.json`.

### Result row (`routeds_{NR|R}.csv`, one scene per run)

| column | meaning |
|---|---|
| `scene`, `react`, `steps`, `term_reason` | the published scene name (`odyssey_sceneNNN`), `nr`/`r`, simulated steps, how the episode ended: `destination_arrival`, `route_deviation`, `route_exhausted` (the ego drove past the end of the SD route, so a route-reading model had nothing left to plan on), `time_limit`; otherwise `log_end` (stopped after the log's own length, before the time limit), `stopped` (stopped earlier, e.g. a dead planner or a truncated scene) or `done` (the environment ended it without naming a reason) |
| `RouteDS` | `100 · RC · P_SD · P_col · P_off · P_TL · P_PLC` |
| `RC`, `P_SD`, `P_col`, `P_off`, `P_TL`, `P_PLC` | route completion and the five penalty factors (paper Appendix C.3) |
| `PLCA`, `PLCS` | pre-lane-change accuracy (over stop lines reached) and score (over every evaluation stop line on the route); blank when the route has no evaluation stop line |
| `Eff`, `Comf` | efficiency (100 = the surrounding traffic's speed) and comfort (1 = every limit respected) |
| `collision_count`, `tl_violation_count`, `plc_stops`, `plc_reached`, `plc_pass`, `plc_late`, `plc_fail` | the counts behind the penalties |
| `tl_set`, `plc_rule`, `scoring_error` | the rules applied, a scoring failure (blank when scored) |

Rendered images and camera paths are filed under the published scene name.

`exit_code.txt`: 0 = the simulator exited 0 and wrote the completion flag; 124 = wall-clock
timeout; otherwise the exit code, or 1 for exit 0 without the flag. `--record` adds
`frames/`, `plan_traj/` and rendered images under `sensor_blobs/`; `--audit` adds
`runtime/model_audit/` (a hash of every model input/output tensor per step). The batch runner
deletes the heavy outputs of successful runs unless `--keep-outputs`.

## Merged results (`OdysseyBenchmark/tools/merge_results.py`)

- `runs.csv`: one row per expected scene × react: `sheet, agent, scene, react, status,
  reason, action, RouteDS_final` (0 for a model failure, blank when the job blocks the score),
  every result-row column above, then `exit_code, flag, duration_s, attempts, gpu, run_dir,
  csv_path, record_source`.
- `summary.csv` / `summary.json`: one row per react — `final`, `scene_set` (`all`, `mini`, `debug`,
  `custom`), `benchmark_set` (the expected
  set is the full 100 scenes), counts per status, the paper's table `RouteDS` (fixed
  denominator), `SDC`, `PLCA`, `PLCS` (pooled over the sheet's stop lines from the `plc_*` counts,
  `n_PLC` runs with a stop line), `Eff.` (median), `Comf.` (SDC, PLCA, PLCS and Comf. in percent;
  the per-run files keep fractions), the components `RC, P_SD,
  P_col, P_off, P_TL, P_PLC` (Table 11), `RouteDS_scored` (scored runs only), `term_reasons`,
  rules, code commit, agent and checkpoint identity. The JSON also lists `jobs_to_rerun`.
  An `all` row (mean of nr and r RouteDS) is added only when both are FINAL and is marked
  optional; it is not a benchmark number.
- `merged.json`: `"driving score"` per react (null unless FINAL or `--allow-incomplete`),
  `eval num`, `expected num`, and `_checkpoint.records[]` with one record per route.

Exit code 0 only when every requested sheet is FINAL (or with `--allow-incomplete`).
