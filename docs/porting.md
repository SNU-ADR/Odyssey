# Porting a model

One file ports a model: an agent config (YAML). The shipped examples under `OdysseyBenchmark/agents/` cover
the published models; copy the closest one.

## The agent config

```yaml
model:
  name: my_model                         # a label: job names, run directories, the results sheet
  python: /envs/my_model/bin/python      # the interpreter your model runs in
  repo: /work/my_navsim_fork             # put on sys.path and made the working directory
  agent_config: my_agent                 # <repo>/navsim/planning/script/config/common/agent/my_agent.yaml
  checkpoint: /work/ckpts/epoch_99.ckpt
  overrides: []                          # Hydra overrides applied to that yaml
  # adapter: my_adapter.py:MyPlanner     # only when the generic adapter cannot run the model
  # extra_pythonpath: [../side_pkgs]     # extra import roots for the planner process only
history_times_s: [-1.5, -1.0, -0.5, 0.0] # past frames the model receives (0.1 s multiples, ending at 0)
camera_times_s:                          # which rig cameras, at which of those times
  CAM_F0: [0.0]
  CAM_L0: [0.0]
  CAM_R0: [0.0]
navigation: {driving_command: true, sd_route: none}   # see below
output: {plan_dt: 0.5, shape: [8, 3]}    # N poses every plan_dt seconds
feature_shapes:                          # optional: expected model input tensor shapes
  camera_feature: [1, 3, 256, 1024]
  status_feature: [1, 8]
# seed: 2026                             # optional planner RNG seed
# stateful: true                         # optional, with reset_method: the method called at episode start
```

Paths may use `${VAR}` and `~`; relative paths are taken from the config file's directory (a
bare `python` command name is looked up on `PATH`). Fields fixed by the benchmark (world
cadence, native rendering, ego-frame output) are not configurable.

`history_times_s` must match the model's training: as many entries as its history frames, at the
same spacing (NAVSIM agents: 4 frames, 0.5 s apart). `camera_times_s` lists the cameras the
feature builders read and at which of those times; `get_sensor_config()` gives them as indices
counted from the oldest history frame (index 3 of 4 = time 0.0). `output` is the model's
`TrajectorySampling`: `num_poses` and `interval_length`.

### `navigation`

| `driving_command` | `sd_route` | the model |
|---|---|---|
| `true` | `none` | reads NAVSIM's driving command (the baselines) |
| `false` | `targets` | drops the command; receives `route_centerline` / `route_centerline_mask` in `targets` of `forward(features, targets)` (LTF, DiffusionDrive, ReCogDrive) |
| `false` | `features` | drops the command; receives them inside `features` (DrivoR, SafeDrive) |

The declaration is checked against the loaded model at start-up (`drop_driving_command`,
`use_sdroute`) and a run stops on a mismatch. With `sd_route` declared, a scene without a
route file is refused before rendering.

## What the generic adapter expects

A NAVSIM `AbstractAgent`:

- a Hydra yaml under `<repo>/navsim/planning/script/config/common/agent/` whose target takes
  `checkpoint_path`;
- `initialize()`, `get_sensor_config()`, `get_feature_builders()`;
- `forward(features)` (or `forward(features, targets)` for `sd_route: targets`) returning a
  dict with `"trajectory"` of shape `(B, N, 3)`.

The runtime narrows the model's native sensor request to the cameras declared in the config
(a model may request more views than it reads; it may not read a view the config does not
declare), batches every tensor it finds in the feature dict (also inside lists), moves it to
the GPU and checks `feature_shapes` when given.

## What the runtime supplies

| input | content |
|---|---|
| cameras | the declared rig cameras (`CAM_F0`, `CAM_L0`, `CAM_R0`, `CAM_L1`, `CAM_R1`, `CAM_L2`, `CAM_R2`, `CAM_B0`) at the declared times: RGB uint8, 1920 x 1080, after one JPEG round trip (OpenCV's default quality), undistorted pinhole (`distortion` 0, K ≈ [[1545, 0, 960], [0, 1545, 560], [0, 0, 1]]), with the rig's `sensor2lidar` extrinsics. No LiDAR. |
| ego status | NAVSIM's `ego_pose`, `ego_velocity`, `ego_acceleration` (ego frame) per history frame; the lateral acceleration is v_x·ω, since the simulator's plant leaves it at zero |
| driving command | one-hot [left, straight, right, unknown] from the log, as in NAVSIM |
| SD route | `route_centerline` (1, 120, 5) = [x, y, dx, dy, heading] in the current ego frame, 1 m apart over 120 m, and `route_centerline_mask` (1, 120); see [OdysseyZoo/README.md](../OdysseyZoo/README.md) "SD route" |
| cadence | the planner is called every 0.1 s simulation tick; before the episode has enough past frames, the history repeats the first frame |

The model returns N poses (x, y, heading) in the current ego frame (x forward, y left), strictly
in the future: t = plan_dt, 2·plan_dt, …, N·plan_dt. It shares the GPU with the renderer and the
Fixer, and one episode (about 2,000 ticks) must finish within the per-attempt timeout (5,400 s
by default). The frame dicts an adapter receives also carry the simulator's annotations
(`anns`, `gt_*`); a planner must not read them.

## When an adapter is needed

Start from `OdysseyBenchmark/agents/template_adapter.py` and reference it with `model.adapter`. Override only
what differs:

| situation | override |
|---|---|
| poses every 0.1 s instead of 0.5 s | `PLAN_DT = 0.1` (and `output.plan_dt: 0.1`) |
| the feature builders do not produce an input the model needs | `_build_features(agent_input, scene_list)` |
| the model is not called as `agent.forward(features)` | `_forward(features)` |
| the trajectory is under another key or shape | `_extract_trajectory(out)` |
| a wrong yaml would load silently | a guard in `build()` |
| the model opens camera files by path under `sensor_blobs_root` (ReCogDrive) | `CAMERA_FILES = True`: the session writes each step's JPEGs before the request, also without `--record` |
| the model is not built by Hydra from `<repo>/navsim/.../agent/<agent_config>.yaml` | `_build_agent()`; it must set `self.agent` to a module with `get_sensor_config()` and `get_feature_builders()` (a NAVSIM agent yaml still has to exist for `agent_config`) |
| the model takes something other than NAVSIM's `AgentInput` | `_build_agent_input(scene_list)` |
| the model has no `_config.drop_driving_command` | the `uses_driving_command` property, checked against `navigation.driving_command` |
| the adapter feeds the route itself | `SD_ROUTE = "features"` or `"targets"`, matching `navigation.sd_route` |
| the model keeps state across steps | `stateful: true` and `reset_method: <name>` in the config; the method is looked up on the built agent and called at every episode start |

`adapter:` takes `file.py:Class` (relative to the config) or `package.module:Class`. The
class must derive from `odyssey_bridge.planners.base.NavsimPlanner`.

## Checking before the long run

```bash
python -m odyssey_runtime check --agent my_model/agent.yaml --no-build   # config and paths (< 1 s)
python -m odyssey_runtime check --agent my_model/agent.yaml              # plus model build and one synthetic step
python -m odyssey_runtime check --agent my_model/agent.yaml --device cpu # the same without a GPU
python -m odyssey_runtime check --agent my_model/agent.yaml --run        # plus a 20-step audited rollout
```

The build stage starts the real planner process and reports the adapter, the narrowed sensor
request (and the sensors it dropped), the navigation inputs the model reads and its plan spacing.
Then one synthetic step goes through the same code a run uses: the declared cameras with the
rig's calibration, an ego at 5 m/s, a straight route. `check` prints the feature shapes the
model received (copy them into `feature_shapes`) and its plan, and repeats the step with a left
command and with a route that turns left. A model declared to read the route whose plan does
not change has not received it, typically because `sd_route` names `features` for a model that
reads `targets` (or the reverse); `check` fails then, as it does when the step raises (a camera
the model reads but the config does not declare, a wrong `output.shape`). Failures come back
with the model's traceback and a hint (e.g. a Hydra interpolation to add to `overrides`).

The build and the step run on `--gpu` (default: the first of `CUDA_VISIBLE_DEVICES`, else 0)
or, with `--device cpu`, without a GPU; a model that queries the GPU while it is built
(ReCogDrive) needs a GPU. `--run` renders a short episode and needs a GPU, the scenes and the Fixer.

## Shipped configs

| config | model | navigation | adapter |
|---|---|---|---|
| `ltf_baseline`, `ltf_sdroute` | LTF (TransFuser latent) | command / route in targets | generic |
| `diffusiondrive_baseline`, `diffusiondrive_sdroute` | DiffusionDrive | command / route in targets | generic |
| `drivor_baseline`, `drivor_sdroute` | DrivoR (4 views) | command / route in features | generic (+2 overrides) |
| `safedrive_baseline`, `safedrive_sdroute` | SafeDrive (3 views × 3 times) | command / route in features | generic |
| `recogdrive_baseline`, `recogdrive_sdroute`, `recogdrive_sdroute_il` | ReCogDrive (VLM) | command / route in targets | `planners/recogdrive*.py` |

ReCogDrive needs its Stage-1 VLM (in the published weights; `RECOGDRIVE_VLM_PATH`) and its own
interpreter, `RECOGDRIVE_PLANNER_PY` ([installation.md](installation.md) "ReCogDrive").
