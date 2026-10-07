# OdysseyBenchmark

The benchmark: it runs a planning model against the simulator on the published scenes and scores
the drive. It holds the launcher and batch runner, the model boundary (planner adapters and the
link to the simulator), the scorer, the agent configs, and the benchmark's fixed data.

## Layout

```
odyssey_runtime/         the launcher (launch.py), batch runner (eval.py), model check (check.py),
                         planner worker, restorer worker, runtime session and recording; the
                         simulator talks to both workers over a socket pair and shared memory
                         (transport.py)
odyssey_bridge/          the model boundary: planner adapters (planners/), helpers the planner side
                         shares (ipc_common.py: trajectory resampling, the derived lateral
                         acceleration, camera-only navsim input), the SD-route sidecar
                         (route_sidecar.py) and the hand-drawn SD edges the scorer's graph overlays
                         (sd_route.py)
odyssey_benchmark/       scoring: driving_metrics (the one scoring path, replay), ds_formula,
                         lane_follow, traffic_efficiency, comfort, tlc_metric, tlc_filter,
                         tlc_timetable, the SD-route RC/SDF scorer (sdroute_score, sdroute_prefix*,
                         sdroute_sdf), and scorer.py, the module the simulator is configured with
agents/                  the shipped agent configs (one YAML per model), ablations/, and
                         template_adapter.py
data/                    sd_roadblock_map/ (SD-route edges) and tlc_timetable/ (the traffic-light
                         timetable)
configs/deployment/      the reference deployment manifest and package inventory
scripts/                 run_evaluation_{debug,mini,multi}.sh, host_env.example.sh, make_map_slots.sh,
                         apply_safedrive_patch.py, check_deployment.py
tools/                   check_env.py, merge_results.py
patches/                 the SafeDrive native patch for OdysseyZoo
tests/                   benchmark tests (runtime/ for the launcher and workers)
```

## Entry points

Run from the repository root after sourcing the environment (your copy of
`OdysseyBenchmark/scripts/host_env.example.sh`), which puts `OdysseyBenchmark`, `OdysseyRenderer`,
`OdysseyTrafficAgent` and `OdysseyBenchmark/odyssey_bridge` on `PYTHONPATH`:

```bash
python OdysseyBenchmark/tools/check_env.py --gpus 0,1,2,3
python -m odyssey_runtime check --agent OdysseyBenchmark/agents/ltf_sdroute.yaml
python -m odyssey_runtime run --agent OdysseyBenchmark/agents/ltf_sdroute.yaml --scene odyssey_scene001 --react nr --gpu 0 --max-steps 40
AGENT_CONFIG=my_model/agent.yaml GPUS=0,1,2,3 bash OdysseyBenchmark/scripts/run_evaluation_multi.sh
python OdysseyBenchmark/tools/merge_results.py -f experiments/simulation/eval_my_model
```

To recompute a finished run's scores from the inputs it was scored from, by the same code as the
run: `odyssey_benchmark.driving_metrics.replay(np.load(".../rollout_trajectory.npz", allow_pickle=True))`.
This is for analysis; it does not update the run's result files or the campaign's records.

## How it uses the others

The launcher starts the simulator (`OdysseyTrafficAgent/odyssey/runner/run_simulation.py`) with
`scorer=odyssey_benchmark.scorer` and the timetable in `OdysseyBenchmark/data/tlc_timetable`;
the simulator's metric manager calls the scorer to pin each frame and to score the run at its end.
The scorer imports simulator types (the PDM scorer, the SD-route tracker, the signal timetable). The
runtime starts the Fixer worker from OdysseyRenderer. Models come from `OdysseyZoo/`.

Tests: `python -m pytest -q OdysseyBenchmark/tests` from the repository root, in the simulator
interpreter.

## License and provenance

Apache-2.0 (`LICENSE` at the repository root). `data/tlc_timetable/` and `data/sd_roadblock_map/` are
derived from nuPlan (nuPlan terms of use); `patches/` modifies SafeDrive code in OdysseyZoo (MIT).
Details: `THIRD_PARTY_NOTICES.md`.
