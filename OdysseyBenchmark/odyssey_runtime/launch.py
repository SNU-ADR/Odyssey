"""Launch one simulation and finalize its metrics, without analysis jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import agent_config as agent_config_module
from .agent_config import AgentConfig
from . import scenes
from odyssey_benchmark.tl_sets import DATA_DIR as TL_SETS_DIR, NAME as TL_SET


#: Per-scene benchmark rules, by published scene name. The traffic-light timetable (TL_SET) is
#: benchmark data (OdysseyBenchmark/data/tlc_timetable); the simulator is told where it lives.
SPAWN_GATE_OFF = ('odyssey_scene041', 'odyssey_scene045', 'odyssey_scene075')   # reactive spawn gate disabled
NO_TL_LABELS = ('odyssey_scene048', 'odyssey_scene053', 'odyssey_scene099')     # signal labels not attached
#: The image restorer: Fixer with the h1_b16_e1 weights, run in torch.
RESTORER = 'fixer_h1b16'
#: The scorer the simulator records for and scores with (OdysseyBenchmark/odyssey_benchmark).
SCORER = 'odyssey_benchmark.scorer'


#: ODYSSEY_* variables a user's shell may set: install locations, infrastructure timing, and
#: diagnostics that only write more files (SAVE_PRE_RESTORE: each image also before the Fixer).
USER_ENV = frozenset('ODYSSEY_' + name for name in (
    'ROOT', 'RUNTIME_ROOT', 'SIM_PY', 'FIXER_PY', 'FIXER_ROOT', 'FIXER_EMPTY_CACHE', 'ZOO_ROOT',
    'MODELS_ROOT', 'SCENES_ROOT', 'MAP_SLOTS', 'PLANNER_PY', 'PLAN_WAIT_TIMEOUT_S', 'FIXER_START_TIMEOUT_S',
    'SAVE_PRE_RESTORE'))
#: ODYSSEY_* variables this launcher sets for every run (a value in the shell is replaced).
LAUNCHER_ENV = frozenset('ODYSSEY_' + name for name in (
    'ROOT', 'ZOO_ROOT', 'RUNTIME_PROFILE', 'RUNTIME_OUTPUT', 'PLANNER', 'PLANNER_PY', 'PLANNER_REPO',
    'PLANNER_CKPT', 'PLANNER_CFG', 'PLANNER_ADAPTER', 'PLANNER_PYTHONPATH', 'SCENE_NAME', 'INJECT_AY',
    'HISTORY_STRIDE', 'ROLLOUT_DT', 'ROUTE_FILE', 'RENDER_CAMS',
    'OMNIRE_ACTOR_POSE_SOURCE', 'RESTORER', 'RENDER_GPU_UNDISTORT', 'FIXER_GPU'))


class EnvironmentRefused(RuntimeError):
    """The launch environment carries variables the benchmark does not accept."""


def check_benchmark_environment(environ=None):
    """Refuse ODYSSEY_* variables outside USER_ENV and LAUNCHER_ENV.

    Every other one the code reads is a research toggle (route noise, acceleration bias, extra
    planner overrides, ...) that would change what is measured without showing up in the result."""
    names = sorted(k for k in (os.environ if environ is None else environ)
                   if k.startswith('ODYSSEY_') and k not in USER_ENV and k not in LAUNCHER_ENV)
    if names:
        raise EnvironmentRefused(
            'These environment variables are not part of the benchmark and would change the run:\n'
            + '\n'.join(f'  {k}' for k in names) + '\nUnset them, then run again.')


def max_steps_arg(text):
    """argparse type for --max-steps; build_run accepts at least 20 steps, in multiples of 5."""
    value = int(text)
    if value < 20 or value % 5:
        raise argparse.ArgumentTypeError(f'must be at least 20 and a multiple of 5 (got {value})')
    return value


def gpus_outside_visible(gpus, environ):
    """GPUs that a CUDA_VISIBLE_DEVICES list exported in the shell does not include.

    Each run sets CUDA_VISIBLE_DEVICES to the GPU it is given, so `--gpu`/`--gpus` take physical
    indices (as nvidia-smi lists them) and an inherited list would not restrict them by itself.
    A list of UUIDs, or an unset/empty variable, is not checked.
    """
    visible = [v.strip() for v in str(environ.get('CUDA_VISIBLE_DEVICES', '')).split(',') if v.strip()]
    if not visible or not all(v.isdigit() for v in visible):
        return []
    return [str(g) for g in gpus if str(g).split('.')[0] not in visible]


def gpu_visibility_error(gpus, environ):
    hidden = gpus_outside_visible(gpus, environ)
    if not hidden:
        return ''
    visible = environ.get('CUDA_VISIBLE_DEVICES')
    return (f'GPU {",".join(hidden)} is not in CUDA_VISIBLE_DEVICES={visible}. GPU numbers are physical '
            f'indices as nvidia-smi lists them; use {visible}, or unset CUDA_VISIBLE_DEVICES')


def build_run(root, profile_path, scene, react, max_steps, gpu, output,
              *, repo=None, checkpoint=None, agent_config=None, record=False, seed=None,
              audit=False, environ=None):
    """Compose one run: the simulator command, its environment and the effective profile.

    ``profile_path`` is an agent config (.yaml) or a loaded AgentConfig. ``environ`` replaces
    os.environ as the base environment (the batch runner composes runs for several GPUs
    concurrently and cannot share the process environment).
    """
    base = dict(os.environ if environ is None else environ)
    check_benchmark_environment(base)
    root, output = Path(root).resolve(), Path(output).resolve()
    agent = profile_path if isinstance(profile_path, AgentConfig) else agent_config_module.load(profile_path, environ=base)
    agent = agent.with_model(repo=repo, checkpoint=checkpoint, agent_config=agent_config).with_seed(seed)
    profile = agent.profile
    repo, checkpoint, agent_config = agent.model.repo, agent.model.checkpoint, agent.model.agent_config
    if react not in ('nr', 'r'):
        raise ValueError('react must be nr or r')
    if max_steps is not None and (max_steps < 20 or max_steps % 5):
        raise ValueError('max_steps must be >=20 and divisible by 5')
    inputs = scenes.resolve(scene, base)
    scene, metadata = inputs.name, inputs.metadata
    if not inputs.checkpoint.is_file():
        raise FileNotFoundError(inputs.checkpoint)
    if profile.navigation['sd_route'] != 'none' and not inputs.route.is_file():
        raise FileNotFoundError(f'{inputs.route}: the profile declares an SD route input, but this scene has no route')
    # The benchmark uses a 1.5s warmup and a 2x saved appearance horizon.
    # Model-specific history remains in the profile; world/scoring cadence is fixed.
    if profile.dt != 0.1:
        raise ValueError('the benchmark world cadence must be 0.1 seconds')
    history = 16
    horizon = 2 * len(metadata['omnire_timestamps_us'])
    future = (max_steps + 1 if max_steps is not None else horizon) - history
    if future <= 0:
        raise ValueError('invalid simulation horizon')
    model = profile.data['planner_id']
    # OdysseyZoo is required: it holds the native models and the only SD-route builder, which
    # the scorer also imports.
    zoo = Path(os.path.abspath(base.get('ODYSSEY_ZOO_ROOT', root / 'OdysseyZoo')))
    if not (zoo / 'sdroute/graph.py').is_file():
        raise FileNotFoundError(f'OdysseyZoo not found at {zoo}; install it or set ODYSSEY_ZOO_ROOT')
    # Resolve relative deployment paths before the simulator and adapters change cwd.
    # abspath deliberately preserves the external checkout's symlink identity.
    repo, checkpoint = os.path.abspath(repo), os.path.abspath(checkpoint)
    effective = dict(profile.data)
    # Persist only scoring outputs by default. Diagnostic recording is explicit.
    for key in ('record_images', 'record_legacy_ipc'):
        effective[key] = bool(record)
    if audit:
        effective['audit'] = True
    env = dict(base)
    # The agent config selects the planner interpreter, model and adapter.
    env.update(ODYSSEY_ROOT=str(root), ODYSSEY_ZOO_ROOT=str(zoo),
               ODYSSEY_RUNTIME_PROFILE=str(output / 'runtime/profile.json'),
               ODYSSEY_RUNTIME_OUTPUT=str(output / 'runtime'),
               **agent.planner_env(),
               ODYSSEY_SCENE_NAME=scene, ODYSSEY_INJECT_AY='1',
               ODYSSEY_HISTORY_STRIDE='5', ODYSSEY_ROLLOUT_DT='0.1',
               ODYSSEY_ROUTE_FILE=str(inputs.route),
               ODYSSEY_RENDER_CAMS=','.join(c.lower() for c in profile.cameras),
               ODYSSEY_OMNIRE_ACTOR_POSE_SOURCE='scenario', ODYSSEY_RESTORER=RESTORER,
               ODYSSEY_RENDER_GPU_UNDISTORT='0', CUDA_VISIBLE_DEVICES=str(gpu),
               ODYSSEY_FIXER_GPU=str(gpu))
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'OdysseyBenchmark'), str(root / 'OdysseyRenderer'),
        str(root / 'OdysseyTrafficAgent'), str(root / 'OdysseyBenchmark/odyssey_bridge'),
        str(root / 'third_party/nvdiffrast'), env.get('PYTHONPATH','')])
    if not env.get('NUPLAN_MAPS_ROOT'):
        raise ValueError('NUPLAN_MAPS_ROOT is not set: point it at a writable copy of the nuPlan maps '
                         '(OdysseyBenchmark/scripts/make_map_slots.sh)')
    maps = Path(env['NUPLAN_MAPS_ROOT'])
    env['NUPLAN_MAPS_ROOT'] = str(maps)
    command = [env.get('ODYSSEY_SIM_PY', sys.executable),
               str(root / 'OdysseyTrafficAgent/odyssey/runner/run_simulation.py')]
    options = dict(
        data_file_path=str(inputs.scenario), scene_checkpoint=str(inputs.checkpoint), nuplan_map_root=str(maps),
        output_dir=str(output / 'odyssey_output'), data_output_dir=str(output / 'odyssey_output'),
        runner_report_file='runner_report.json', job_name=f'{scene}_{react}_{model}',
        num_future=future, num_history=history, use_planner_actions='true',
        ego_controller='two_stage_controller', ego_policy='env_input_policy',
        ego_client='planner_client', ego_navigation='trajectory_navigation',
        with_metric_manager='true', scorer=SCORER, record_frame_index=str(bool(record)).lower(),
        driving_command_mode='log_progress', renderer='omnire',
        **{'renderer.lift':'all', 'renderer.road_surface':str(inputs.road_surface)},
        q_lateral='[1.0,10.0,0.0]', gt_budget_multiplier=1.0,
        plc_rule='plc', tl_set=TL_SET, tlc_timetable_sets_dir=str(TL_SETS_DIR))
    if max_steps is not None:
        options['max_step'] = max_steps + 1
    if react == 'r':
        options.update(agent_policy='nuplan_idm_policy', agent_navigation='trajectory_navigation')
        if scene in SPAWN_GATE_OFF:
            options['spawn_ego_tight_ahead_gate'] = 'false'
    else:
        options.update(agent_replay='sector_lead', sector_by='inter')
    # Benchmark signal settings, as in the paper's runs: a scene of the timetable is driven with
    # that timetable and its own labels, every other scene with its own labels when its
    # reconstruction has signal nodes.
    from odyssey.manager.tlc_timetable_set import load_set, scene_code
    if scene_code(inputs.scenario_key) in load_set(TL_SET, TL_SETS_DIR)['scenes']:
        options['tlc_timetable_set'] = TL_SET
        options['tl_control_path'] = str(inputs.tl_control)
    elif scene not in NO_TL_LABELS:
        if json.loads(inputs.inventory.read_text()).get('TrafficLightNodes'):
            options['tl_control_path'] = str(inputs.tl_control)
    command += [f'{key}={value}' for key,value in options.items()]
    return dict(command=command, env=env, profile=effective, output=str(output),
                scene=scene, react=react, checkpoint=str(inputs.checkpoint), agent=agent,
                restorer=RESTORER)


def materialize(spec, output=None):
    """Create the run directory as a run needs it: a new directory, runtime/profile.json, launch.json."""
    output = Path(output or spec['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runtime').mkdir()
    (output / 'runtime/profile.json').write_text(json.dumps(spec['profile'], indent=2))
    (output / 'launch.json').write_text(json.dumps(dry_view(spec), indent=2))
    return output


#: a simulator still alive this long after writing its completion flag and runner report is
#: stuck in interpreter teardown; it is stopped and the run counts as finished.
TEARDOWN_GRACE_S = 120
POLL_S = 5


def finished_but_alive(output, now=None):
    """True when the run's completion flag and runner report exist and the report is older than
    TEARDOWN_GRACE_S."""
    out = Path(output) / 'odyssey_output'
    report = out / 'runner_report.json'
    if not ((out / 'simulation_completed.flag').is_file() and report.is_file()):
        return False
    return (now or time.time()) - report.stat().st_mtime > TEARDOWN_GRACE_S


def execute(spec, *, timeout=None, grace_s=30, on_start=None):
    """Run a materialised spec to completion, or to ``timeout`` seconds.

    The simulator gets its own session so a timeout or interrupt can stop it together with
    everything it started. Status: 0 = exit 0 with the completion flag; 124 = wall-clock
    timeout; otherwise the exit code, or 1 for exit 0 without the flag. Writes exit_code.txt.
    A simulator that finished its run (flag and runner report written) but has not exited
    TEARDOWN_GRACE_S later is stopped; that run has status 0 and ``stopped_after_finish``.
    ``on_start(proc)`` is called with the simulator process as soon as it exists, so a caller
    running this in a thread (the batch runner) can stop it on its own interrupt.
    """
    root, output = Path(spec['env']['ODYSSEY_ROOT']), Path(spec['output'])
    started = time.time()
    timed_out = stopped_after_finish = False
    with (output / 'simulation.log').open('w') as log:
        proc = subprocess.Popen(spec['command'], env=spec['env'], cwd=root / 'OdysseyTrafficAgent',
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if on_start is not None:
            on_start(proc)
        try:
            while True:
                wait = POLL_S if timeout is None else max(0.1, min(POLL_S, started + timeout - time.time()))
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if timeout is not None and time.time() - started >= timeout:
                    timed_out = True
                    terminate_tree(proc, grace_s)
                    break
                if finished_but_alive(output):
                    stopped_after_finish = True
                    terminate_tree(proc, grace_s)
                    break
        except KeyboardInterrupt:
            terminate_tree(proc, grace_s)
            raise
    completed = (output / 'odyssey_output/simulation_completed.flag').is_file()
    rc = proc.returncode
    if timed_out:
        status = 124
    elif stopped_after_finish and completed:
        status = 0
    else:
        status = rc if rc else (0 if completed else 1)
    (output / 'exit_code.txt').write_text(f'{status}\n')
    ended = time.time()
    return dict(status=status, returncode=rc, completed=completed, timed_out=timed_out,
                stopped_after_finish=stopped_after_finish, pid=proc.pid,
                started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(timespec='seconds'),
                ended_at=datetime.fromtimestamp(ended, timezone.utc).isoformat(timespec='seconds'),
                duration_s=round(ended - started, 1))


def terminate_tree(proc, grace_s=30):
    """SIGTERM the simulator's process group, SIGKILL what survives, then sweep its workers."""
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    sweep_workers(proc.pid)


def worker_processes(parent_pid, scan=None):
    """PIDs of planner/Fixer workers started for ``parent_pid``.

    Workers run in their own sessions (transport.py) and name their parent as the last
    argument of ``-m odyssey_runtime.worker ...``; a parent that was killed cannot deliver
    the usual parent-death signal, so they are found by that argument.
    """
    found = []
    for pid, argv in (scan() if scan else _scan_proc()):
        if 'odyssey_runtime.worker' in argv and argv[-1:] == [str(parent_pid)]:
            found.append(pid)
    return found


def _scan_proc():
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            argv = Path('/proc', entry, 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        yield int(entry), [a.decode(errors='replace') for a in argv if a]


def sweep_workers(parent_pid, grace_s=10, scan=None):
    pids = worker_processes(parent_pid, scan)
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 0)):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait
        while pids and time.time() < deadline:
            time.sleep(0.2)
            pids = worker_processes(parent_pid, scan)
        pids = worker_processes(parent_pid, scan)
        if not pids:
            break
    return pids


def dry_view(spec):
    """The spec as recorded in launch.json: no environment, the agent config as a path."""
    view = {k: v for k, v in spec.items() if k not in ('env', 'agent')}
    view['agent'] = dict(path=str(spec['agent'].path), name=spec['agent'].model.name,
                         adapter=spec['agent'].model.adapter)
    return view


def main(profile_path, argv=None):
    root = Path(__file__).resolve().parents[2]          # the repository root
    parser = argparse.ArgumentParser(prog='python -m odyssey_runtime run --agent CONFIG', description=__doc__)
    parser.add_argument('--scene', default='odyssey_scene001', help='published scene name: odyssey_scene001, scene001 or 001')
    parser.add_argument('--react', choices=['nr','r'], default='nr')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max-steps', type=max_steps_arg)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repo')
    parser.add_argument('--checkpoint')
    parser.add_argument('--agent-config')
    parser.add_argument('--seed', type=int, help='Persist a planner RNG seed for this run')
    parser.add_argument('--record', action='store_true', help='Save model-input images/plans for explicit diagnostics')
    parser.add_argument('--audit', action='store_true', help='Hash every model input/output tensor per step (runtime/model_audit)')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args(argv)
    output = args.output or root / 'experiments/simulation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    gpu_error = gpu_visibility_error([args.gpu], os.environ)
    if gpu_error:
        print(f'error: {gpu_error}', file=sys.stderr)
        return 2
    try:
        spec = build_run(root, profile_path, args.scene, args.react, args.max_steps, args.gpu,
                         output, repo=args.repo, checkpoint=args.checkpoint, agent_config=args.agent_config,
                         record=args.record, seed=args.seed, audit=args.audit)
    except EnvironmentRefused as error:
        print(error, file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    if args.dry:
        print(json.dumps(dry_view(spec), indent=2))
        return 0
    problems = spec['agent'].verify_paths()
    if problems:
        print(f"{spec['agent'].path}:\n  " + '\n  '.join(problems) +
              '\n(published weights: https://huggingface.co/ADRLAB/odyssey-models, ODYSSEY_MODELS_ROOT)', file=sys.stderr)
        return 2
    materialize(spec, output)

    def stop(signum, frame):                # `kill` stops the run like Ctrl-C: execute() then
        raise KeyboardInterrupt             # takes the simulator and its workers down with it
    signal.signal(signal.SIGTERM, stop)
    status = execute(spec)['status']
    print(f'exit={status}; output={output}', flush=True)
    if status:
        print((output/'simulation.log').read_text()[-6000:])
    return status
