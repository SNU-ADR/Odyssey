"""Check a model before the long run: its config, then a build and one step in the planner process.

    python -m odyssey_runtime check --agent my_model/agent.yaml [--gpu N | --device cpu] [--run [--steps 20]]

1. static   the config loads, every path exists, the contract is printed      (< 1 s)
2. build    the planner interpreter builds the model exactly as a run would   (10-60 s)
            and reports the adapter, sensor request, navigation and plan dt
3. probe    one synthetic step through the same code a run uses: the feature shapes the model
            receives and its plan; the plan must change when the SD route (or the driving
            command) changes, or the model is not reading it where the runtime feeds it
4. --run    a short odyssey_scene001 rollout with tensor auditing: observed feature shapes and step time

The build and the probe run on one GPU (--gpu, default: the first of CUDA_VISIBLE_DEVICES, else 0)
or, with --device cpu, without a GPU.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

from .agent_config import is_agent_config, load

HINTS = (
    ("InterpolationKeyError", "the agent yaml references a key its training config supplied; add it to model.overrides, e.g. batch_size=1"),
    ("InterpolationResolutionError", "the agent yaml references a key its training config supplied; add it to model.overrides"),
    ("PLAN_DT differs", "output.plan_dt must equal the adapter's PLAN_DT (0.5 s unless the adapter sets PLAN_DT)"),
    ("native sensor history", "camera_times_s does not match the model's sensor request; declare the cameras and times it reads"),
    ("would never read indices", "camera_times_s declares cameras or times the model does not request. Remove them; or, if "
                                 "only the indices differ, make history_times_s as long as the model's history (indices "
                                 "count from the oldest history frame)"),
    ("declared in feature_shapes", "a key in feature_shapes is not one of the model's features (the error lists them)"),
    ("model feature shape", "feature_shapes disagrees with what the feature builders produce (the error shows both)"),
    ("planner output shape", "output.shape must be the (N, 3) plan the model returns"),
    ("checkpoint not found", "model.checkpoint does not exist on this host"),
    ("contradicts sd_route", "navigation.sd_route disagrees with the model config's use_sdroute"),
    ("driving_command", "navigation.driving_command disagrees with the model config's drop_driving_command "
                        "(an adapter that reads no _config should override uses_driving_command)"),
    ("is not a NavsimPlanner subclass", "model.adapter must name a class derived from odyssey_bridge.planners.base.NavsimPlanner"),
    ("adapter file not found", "model.adapter path is relative to the agent config file"),
    ("No module named", "a dependency is missing in model.python's environment (or the adapter's imports)"),
    ("No CUDA GPUs are available", "no GPU is visible: pass --gpu <index>, or --device cpu to check without a GPU"),
    ("CUDA out of memory", "the model does not fit next to the renderer on this GPU"),
    ("size mismatch", "the checkpoint does not match the agent yaml's architecture"),
)


def hint_for(text):
    for needle, hint in HINTS:
        if needle in text:
            return hint
    return ""


def default_gpu(environ):
    visible = [v.strip() for v in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    return visible[0] if visible else "0"


def contract_table(cfg):
    model, profile = cfg.model, cfg.profile
    cameras = {name: list(times) for name, times in profile.cameras.items()}
    rows = [
        ("name", model.name), ("python", model.python), ("repo", model.repo),
        ("agent_config", model.agent_config), ("checkpoint", model.checkpoint),
        ("adapter", model.adapter or "generic (NavsimPlanner)"),
        ("overrides", " ".join(model.overrides) or "-"),
        ("history_times_s", list(profile.times)),
        ("cameras", f"{len(cameras)} {cameras}"),
        ("navigation", profile.navigation),
        ("output", f"{profile.data['output']['shape']} every {profile.data['output']['plan_dt']} s"),
        ("feature_shapes", profile.data.get("feature_shapes") or "-"),
    ]
    width = max(len(k) for k, _ in rows)
    lines = [f"  {k:<{width}}  {v}" for k, v in rows]
    return "\n".join(lines)


def dry_build(cfg, root, gpu, timeout, device="cuda", probe=True):
    """Start the real planner worker with the config a run would give it, read its metadata, stop.

    Returns (metadata or None, error text, seconds, log path). The log is kept for inspection.
    """
    from .probe import write_route
    from .transport import SharedWorker

    log = Path(tempfile.mkdtemp(prefix="odyssey-check-")) / "planner_worker.log"
    with tempfile.TemporaryDirectory(prefix="odyssey-check-") as tmp:
        worker_cfg = dict(
            profile=cfg.profile.data, cfg=cfg.model.agent_config, checkpoint=cfg.model.checkpoint,
            repo=cfg.model.repo, sensor_root=tmp, history_stride=5, device=device,
            # A synthetic route file: the build needs one for an SD-route model, not a scene.
            route_file=write_route(Path(tmp) / "route.npz", turn=False),
            overrides=list(cfg.model.overrides), adapter=cfg.model.adapter,
        )
        if probe:
            worker_cfg["probe_dir"] = tmp
        visible = "" if device == "cpu" else str(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, ODYSSEY_INJECT_AY="1", ODYSSEY_ROOT=str(root))
        env.setdefault("ODYSSEY_ZOO_ROOT", str(root / "OdysseyZoo"))
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        for key in ("LD_LIBRARY_PATH", "PYTHONHOME"):
            env.pop(key, None)
        extra = cfg.planner_env().get("ODYSSEY_PLANNER_PYTHONPATH")
        if extra:
            env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")
        factory = "odyssey_runtime.probe:probe_worker" if probe else "odyssey_runtime.planner:PlannerWorker"
        start = time.perf_counter()
        try:
            worker = SharedWorker(cfg.model.python, factory, worker_cfg, 64 * 1024 ** 2, log, env=env, timeout=timeout)
        except Exception as error:
            # The worker log holds the traceback; the exception repeats it as an escaped repr.
            text = log.read_text(errors="replace")[-4000:] if log.exists() else ""
            return None, text.strip() or str(error), time.perf_counter() - start, log
        try:
            return worker.ready, "", time.perf_counter() - start, log
        finally:
            worker.close()


def report_probe(probe, ready, navigation):
    """Print the probe's result; return False when it found a problem that would fail a run."""
    if not probe.get("ok"):
        print(f"FAIL    inference on a synthetic step: {probe.get('error', '?')}", file=sys.stderr)
        tb = probe.get("traceback", "").rstrip().splitlines()
        if tb:
            print("  " + "\n  ".join(tb[-12:]), file=sys.stderr)
        dropped = [s for s in ready.get("dropped_sensors", []) if not s.startswith("lidar")]
        lidar = [s for s in ready.get("dropped_sensors", []) if s.startswith("lidar")]
        if dropped or lidar:
            print("hint: the runtime renders only the declared cameras and no LiDAR; the model's native request "
                  f"also had {' '.join(dropped + lidar)}. If the model reads them, declare the cameras in "
                  "camera_times_s (LiDAR is not available: use a camera-only variant)", file=sys.stderr)
        else:
            hint = hint_for(probe.get("traceback", "") + probe.get("error", ""))
            if hint:
                print(f"hint: {hint}", file=sys.stderr)
        return False
    plan = probe["plan"]
    last = plan[-1] if plan else []
    print(f"ok      inference in {probe['planner_ms']:.0f} ms: plan {[len(plan), len(last)]}, "
          f"last pose x {last[0]:.2f} y {last[1]:.2f} heading {last[2]:.3f} (synthetic input, ego at 5 m/s)")
    print(f"  {'features':<16} {probe['features']}")
    good = True
    if "route_changes_plan" in probe:
        fed = navigation["sd_route"]
        if probe["route_changes_plan"]:
            print(f"ok      the plan changes with the SD route (fed via {fed})")
        else:
            other = "targets" if fed == "features" else "features"
            print(f"FAIL    the plan does not change when the SD route changes: the model does not read the route "
                  f"the runtime feeds via {fed!r}", file=sys.stderr)
            print(f"hint: if the model's forward() reads route_centerline from {other}, declare "
                  f"navigation.sd_route: {other}", file=sys.stderr)
            good = False
    elif navigation["sd_route"] != "none":
        print("  route            fed by the adapter itself; not probed")
    if "command_changes_plan" in probe:
        if probe["command_changes_plan"]:
            print("ok      the plan changes with the driving command")
        else:
            print("WARN    the plan does not change with the driving command, although the model is declared "
                  "to read it (navigation.driving_command: true)")
    return good


def short_run(config, root, gpu, scene, steps):
    """A bounded rollout with auditing; returns (status, output dir)."""
    from .launch import main as run

    output = root / "experiments/simulation" / f"check_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    argv = ["--scene", scene, "--react", "nr", "--gpu", str(gpu), "--max-steps", str(steps),
            "--audit", "--output", str(output)]
    print(f"\nrollout: python -m odyssey_runtime run --agent {config} {shlex.join(argv)}", flush=True)
    return run(config, argv), output


def observed(output):
    audit = sorted((output / "runtime/model_audit").glob("*.json"))
    lines = []
    if audit:
        first = json.loads(audit[-1].read_text())
        lines.append("  observed model tensors (last audited step):")
        for key, value in first.items():
            if isinstance(value, dict) and "shape" in value:
                lines.append(f"    {key:<32} {value['shape']} {value.get('dtype', '')}")
    timings = output / "runtime/timings.jsonl"
    if timings.exists():
        ms = [json.loads(line)["planner_ms"] for line in timings.read_text().splitlines() if line.strip()]
        ms = [m for m in ms if m is not None]
        if ms:
            ms.sort()
            lines.append(f"  planner step time: median {ms[len(ms) // 2]:.0f} ms, max {ms[-1]:.0f} ms over {len(ms)} steps")
    return "\n".join(lines)


def main(config_path, argv=None):
    root = Path(__file__).resolve().parents[2]          # the repository root
    parser = argparse.ArgumentParser(prog="python -m odyssey_runtime check --agent CONFIG", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", help="physical GPU index (default: the first of CUDA_VISIBLE_DEVICES, else 0)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="cpu builds and probes the model without a GPU (slower; --run still needs a GPU)")
    parser.add_argument("--no-build", action="store_true", help="static checks only")
    parser.add_argument("--no-probe", action="store_true", help="build only; skip the synthetic inference step")
    parser.add_argument("--run", action="store_true", help="also run a short audited rollout (GPU, scenes and Fixer)")
    parser.add_argument("--scene", default="odyssey_scene001")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=900, help="seconds allowed for the model build")
    args = parser.parse_args(argv)
    if args.run and args.device == "cpu":
        parser.error("--run renders and needs a GPU; drop --device cpu")
    gpu = args.gpu or default_gpu(os.environ)
    config = Path(config_path)
    if not is_agent_config(config):
        print(f"{config}: `check` needs an agent config (.yaml)", file=sys.stderr)
        return 2
    try:
        cfg = load(config)
    except ValueError as error:
        print(f"FAIL config: {error}", file=sys.stderr)
        return 2
    print(f"config  {config}\n{contract_table(cfg)}")
    problems = cfg.verify_paths()
    if problems:
        print("FAIL paths:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    print("ok      config and paths")
    if args.no_build:
        return 0
    if args.device == "cuda":
        from .launch import gpu_visibility_error
        error = gpu_visibility_error([gpu], os.environ)
        if error:
            print(f"FAIL gpu: {error}", file=sys.stderr)
            return 2
    ready, error, seconds, log = dry_build(cfg, root, gpu, args.timeout, device=args.device, probe=not args.no_probe)
    if ready is None:
        print(f"FAIL build ({seconds:.0f} s):\n{error.rstrip()}\n(full log: {log})", file=sys.stderr)
        hint = hint_for(error)
        if args.device == "cpu" and "No CUDA GPUs" in error:
            hint = "this model (or its adapter) needs a GPU to build; check it with --gpu <index> instead of --device cpu"
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1
    where = "on the CPU" if args.device == "cpu" else f"on GPU {gpu}"
    print(f"ok      build in {ready.get('build_seconds', seconds):.0f} s {where} ({ready.get('device_name', '?')})")
    for key in ("adapter", "plan_dt", "cameras", "sensor_config", "dropped_sensors", "navigation", "overrides"):
        if key in ready:
            print(f"  {key:<16} {ready[key]}")
    if "probe" in ready and not report_probe(ready["probe"], ready, cfg.profile.navigation):
        print(f"(planner log: {log})", file=sys.stderr)
        return 1
    if not args.run:
        return 0
    status, output = short_run(config, root, gpu, args.scene, args.steps)
    if status:
        from .eval import reason_line
        sim_log = output / "simulation.log"
        text = sim_log.read_text(errors="replace")[-20000:] if sim_log.exists() else ""
        print(f"FAIL rollout exit {status}: {reason_line(text) if text else 'no simulation.log'}\n"
              f"(log: {sim_log})", file=sys.stderr)
        hint = hint_for(text)
        if hint:
            print(f"hint: {hint}", file=sys.stderr)
        return 1
    print(f"ok      rollout {args.steps} steps -> {output}\n{observed(output)}")
    return 0
