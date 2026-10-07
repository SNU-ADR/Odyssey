# OdysseyTrafficAgent

The closed-loop simulator. It loads a scenario, steps the world at 0.1 s, moves the other road users
(log replay led by the sector ahead of the ego, or IDM), executes the planner's trajectory through
the ego controller, applies the signal timetable, ends the rollout on arrival, departure or the step
budget, and records what happened for scoring.

## Layout

```
defines.py               every physical and protocol constant (clocks, arrival and departure
                         thresholds); odyssey.utils.cadence re-exports it
odyssey/
  runner/                run_simulation.py (the Hydra entry point), executor, worker pool
  envs/                  base_env.py: the episode loop, termination (arrival, route departure,
                         budget) and the calls into the managers
  manager/               scenario, map, agent, render, data and metric managers; signal_patch and
                         tlc_timetable_set (signal timetable application); sdroute_metric (the SD
                         route file and the tracker termination reads); frenet
  components/agents/     ego and actor agents: policies (IDM, PDM, log replay), controllers and
                         trackers, motion models, navigation, observations, the planner client
  scenario/, components/maps/   scenario description, hybrid and sector replay, lane graph
  configs/               default_runner.yaml and the Hydra groups (renderer, worker, logging)
  engine/, base_class/, common/, utils/
tests/                   simulator tests
```

## Entry points

- `odyssey/runner/run_simulation.py key=value ...`: one simulation, configured by
  `odyssey/configs/default_runner.yaml` plus overrides. The benchmark launcher builds this command;
  it is not meant to be run by hand.
- `odyssey.envs.base_env.BaseEnv`: the environment; `done_function()` decides arrival and departure.

Two configuration keys connect it to the benchmark:

- `scorer`: a dotted module path, imported with importlib, that the metric manager records for and
  scores with (`odyssey/manager/metric_manager.py`, `SCORER_API`). Required with
  `with_metric_manager=true`; the launcher passes `odyssey_benchmark.scorer`.
- `tlc_timetable_sets_dir`: where a `tlc_timetable_set` name is looked up; the launcher passes
  `OdysseyBenchmark/data` with `tlc_timetable_set=tlc_timetable` (the benchmark's one timetable).

## How the others use it

OdysseyBenchmark launches it as a subprocess (`odyssey_runtime/launch.py`) and imports its types for
scoring (PDM scorer, route tracker, signal timetable). It renders through OdysseyRenderer
(`odyssey_renderer`). It imports no benchmark module at module level; scoring reaches it only
through `scorer`. It does talk to the benchmark's runtime session (`odyssey_runtime.session`):
the planner client (`planner_client`) imports it when it is built, and the data manager and the
executor import it when `ODYSSEY_RUNTIME_PROFILE` is set; all three imports are inside functions.
`OdysseyTrafficAgent` must be on `PYTHONPATH`.

Tests: `python -m pytest -q OdysseyTrafficAgent/tests` from the repository root, in the simulator
interpreter.

## License and provenance

Apache-2.0 (`LICENSE` at the repository root). The simulator started from WorldEngine's SimEngine
(Apache-2.0; modified files carry a notice), with structure from MetaDrive, PDM-Closed from tuPlan
Garage and nuPlan-based utilities from nuplan-devkit (all Apache-2.0). Details:
`THIRD_PARTY_NOTICES.md`.
