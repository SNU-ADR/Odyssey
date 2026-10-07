"""Batch evaluation: every benchmark scene x {nr, r} on one or more GPUs, resumable.

    python -m odyssey_runtime eval --agent my_model/agent.yaml --gpus 0,1,2,3 --output experiments/simulation/eval_my_model

One worker thread per GPU takes jobs from a shared queue and runs each through the single-run
launcher (``launch.build_run`` -> ``materialize`` -> ``execute``). Every job leaves a record under
``jobs/`` that says what happened and why; ``manifest.json`` is rebuilt from those records after
each job and is what ``OdysseyBenchmark/tools/merge_results.py`` reads. Running the same command again continues
where it stopped: complete jobs are skipped, infrastructure failures are retried, model failures
stand unless ``--retry-failed``.

Campaign layout::

    <output>/manifest.json            what was run, where, and the state of every job
    <output>/inputs/                  the agent config, effective profile and scene table used
    <output>/jobs/<scene>_<react>.json   one record per job (attempts, failure class, scores)
    <output>/runs/<scene>_<react>/attempt_NN/   the launcher's output for each attempt
    <output>/eval.log
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

from . import scenes
from .agent_config import is_agent_config, load as load_agent
from .launch import RESTORER
from .launch import max_steps_arg

JOB_SCHEMA = "odyssey_eval_job/1"
MANIFEST_SCHEMA = "odyssey_eval_manifest/1"
FINAL = ("complete", "model_fail", "input_missing")
RETRYABLE = ("infra_fail", "timeout")
#: three scenes that exercise the benchmark's special rules: a signal-timetable scene, the
#: reactive scene whose spawn gate is disabled, and a plain scene.
DEBUG_SET = (("odyssey_scene011", "nr"), ("odyssey_scene075", "r"), ("odyssey_scene002", "nr"))
#: The published 30-scene mini set (``--scenes mini``): a quick, comparable subset for development.
#: Its numbers are a mini-set result, never the 100-scene benchmark number.
MINI_SET = tuple(f"odyssey_scene{n:03d}" for n in (
    3, 7, 8, 9, 10, 15, 17, 18, 25, 28, 29, 33, 35, 36, 37,
    49, 51, 63, 64, 67, 68, 69, 76, 78, 83, 84, 86, 93, 96, 99))
SCENE_SETS = ("all", "mini", "debug")
HEAVY = ("odyssey_output/sensor_blobs", "odyssey_output/openscene_format/sensor_blobs", "frames", "plan_traj",
         "runtime/model_audit")
MIN_FREE_GB = 20


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))
    os.replace(tmp, path)


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit
        while True:
            chunk = f.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def checkpoint_identity(path):
    """Size plus the hash of the first MB: cheap, and different for any retrained weights."""
    st = os.stat(path)
    return dict(path=str(path), bytes=st.st_size, head_1mb_sha256=sha256_file(path, 1 << 20),
                mtime=datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"))


def git_state(root):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = run("rev-parse", "HEAD")
    return dict(commit=commit or None, branch=run("rev-parse", "--abbrev-ref", "HEAD") or None,
                dirty=bool(run("status", "--porcelain")) if commit else None)


# ----------------------------------------------------------------------------- jobs
class Job:
    __slots__ = ("scene", "react", "token")

    def __init__(self, row, react):
        self.scene, self.react, self.token = row["scene"], react, row["token"]

    @property
    def job_id(self):
        return f"{self.scene}_{self.react}"

    def as_dict(self):
        return dict(job_id=self.job_id, scene=self.scene, react=self.react, token=self.token)


def scene_number(row):
    m = re.search(r"(\d+)$", row["scene"])
    return int(m.group(1)) if m else row["scene"]


def plan_jobs(rows, selection, reacts):
    """Jobs for ``selection`` ('all', 'mini', 'debug', a list of names/codes or '@file') x ``reacts``."""
    by_name = {r["scene"]: r for r in rows}
    if selection == "debug":
        pairs = []
        for name, react in DEBUG_SET:
            if name not in by_name:
                raise ValueError(f"debug scene {name} is not in scenes.csv")
            pairs.append((by_name[name], react))
    else:
        if selection == "all":
            chosen = sorted(rows, key=scene_number)
        elif selection == "mini":
            chosen = [scenes.find_release_row(rows, n) for n in MINI_SET]
        else:
            names = selection
            if isinstance(names, str):
                names = [n for n in re.split(r"[,\s]+", names) if n]
            if len(names) == 1 and names[0].startswith("@"):
                listing = Path(names[0][1:])
                names = []
                for line in listing.read_text().splitlines():
                    line = line.split("#", 1)[0].strip()
                    names += [n for n in re.split(r"[,\s]+", line) if n]
            chosen = [scenes.find_release_row(rows, n) for n in names]
        pairs = [(row, react) for row in chosen for react in reacts]
    seen, jobs = set(), []
    for row, react in pairs:
        job = Job(row, react)
        if job.job_id not in seen:
            seen.add(job.job_id)
            jobs.append(job)
    return jobs


def installed_gpus():
    """GPU indices nvidia-smi lists, or None where nvidia-smi is unavailable (a planning host)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def plan_workers(gpus, runs_per_gpu=1):
    """Worker ids: the GPU index for the first run on a GPU, '<gpu>.<k>' for its k-th (k >= 2)."""
    if len(set(gpus)) != len(gpus):
        raise ValueError("duplicate GPUs: list each GPU once and use --runs-per-gpu")
    if runs_per_gpu < 1:
        raise ValueError("--runs-per-gpu must be at least 1")
    return [g if k == 1 else f"{g}.{k}" for k in range(1, runs_per_gpu + 1) for g in gpus]


def gpu_of(worker):
    return worker.split(".", 1)[0]


def assign_slots(workers, slots_arg, environ):
    """One nuPlan map slot per worker: concurrent runs must not share a slot's lock directory."""
    if len(set(workers)) != len(workers):
        raise ValueError("duplicate workers")
    if slots_arg:
        slots = [s for s in re.split(r"[,\s]+", slots_arg) if s]
        if len(slots) != len(workers):
            raise ValueError(f"--map-slots lists {len(slots)} slots for {len(workers)} runs at a time")
    elif any("." in w for w in workers):
        raise ValueError("several runs per GPU need one map directory each: pass --map-slots (or ODYSSEY_MAP_SLOTS) "
                         f"with {len(workers)} directories")
    else:
        base = os.path.realpath(environ.get("NUPLAN_MAPS_ROOT", ""))
        m = re.fullmatch(r"(.*?)(\d+)", base)
        if not m:
            raise ValueError(f"cannot derive map slots from NUPLAN_MAPS_ROOT={base!r}; "
                             "pass --map-slots or set ODYSSEY_MAP_SLOTS (one directory per GPU)")
        slots = [m.group(1) + str(gpu) for gpu in workers]
    real = [os.path.realpath(s) for s in slots]
    for slot, path in zip(slots, real):
        if not os.path.isfile(os.path.join(path, "nuplan-maps-v1.0.json")):
            raise ValueError(f"map slot {slot} is not a nuPlan map directory (nuplan-maps-v1.0.json missing)")
    if len(set(real)) != len(real):
        raise ValueError(f"map slots must be distinct directories: {slots}")
    return dict(zip(workers, [str(Path(s)) for s in slots]))


def run_environ(base, slot, model_python, scenes_root):
    env = dict(base)
    env.update(NUPLAN_MAPS_ROOT=slot, ODYSSEY_PLANNER_PY=model_python,
               ODYSSEY_SCENES_ROOT=str(scenes_root), HYDRA_FULL_ERROR="1")
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    return env


# ----------------------------------------------------------------------------- inspection
def result_files(run_dir):
    """The run's result row file: odyssey_output/routeds_{NR|R}.csv (older runs: odyssey_output/openscene_format/)."""
    out = Path(run_dir, "odyssey_output")
    return sorted(out.glob("routeds_*.csv")) or sorted(out.glob("openscene_format/routeds_*.csv"))


def read_scene_row(run_dir):
    """The run's result row (odyssey_output/routeds_{NR|R}.csv)."""
    for path in result_files(run_dir):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            return rows[0], str(path.relative_to(run_dir))
    return None, None


def read_runner_report(run_dir):
    reports = sorted(Path(run_dir, "odyssey_output").glob("runner_report*.json"))
    if not reports:
        return None
    try:
        data = json.loads(reports[-1].read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def tail(path, lines=40):
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def error_lines(text):
    return [line for line in text.splitlines() if re.match(r"^\w+(Error|Exception|Exit)\b", line)]


def applied_rules(run_dir):
    """The per-scene benchmark rules the run was launched with (from launch.json's command)."""
    try:
        command = json.loads(Path(run_dir, "launch.json").read_text()).get("command") or []
    except (OSError, ValueError):
        return None
    options = dict(a.split("=", 1) for a in command if "=" in a)
    return dict(gate_off=options.get("spawn_ego_tight_ahead_gate") == "false",
                tlc_timetable_set=options.get("tlc_timetable_set"),
                tl_control_path="tl_control_path" in options)


def last_step(run_dir):
    path = Path(run_dir, "runtime/timings.jsonl")
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return int(json.loads(line)["step"])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def inspect_run(run_dir):
    run_dir = Path(run_dir)
    exit_code = None
    try:
        exit_code = int(run_dir.joinpath("exit_code.txt").read_text().strip())
    except (OSError, ValueError):
        pass
    row, csv_path = read_scene_row(run_dir)
    return dict(exit_code=exit_code, flag=run_dir.joinpath("odyssey_output/simulation_completed.flag").is_file(),
                report=read_runner_report(run_dir), row=row, csv_path=csv_path,
                sim_log=run_dir / "simulation.log", planner_log=run_dir / "runtime/planner_worker.log",
                last_step=last_step(run_dir))


MODEL_PATTERNS = (
    (r"planner output differs from declared finite trajectory shape|planner output shape .* differs from the declared"
     r"|planner output contains non-finite values", "bad_plan"),
    (r"differs from profile|profile declares|contradicts sd_route|would never read indices|selected planner differs"
     r"|declared in feature_shapes", "profile_mismatch"),
    (r"PLAN_DT differs|reset method is not callable", "profile_mismatch"),
)
INFRA_PATTERNS = (
    (r"worker response deadline expired", "planner_timeout"),
    (r"unexpected bytes after camera batch|camera batch missing|stale episode/step|world cadence differs|no planner response", "runtime_contract"),
    (r"Fixer .*identity mismatch|restore_batch|FixerWorker|fixer_worker", "fixer"),
)
ENV_PATTERNS = r"CUDA out of memory|CUDA error|cudaErrorMemoryAllocation|No such file or directory: '[^']*python|No module named 'odyssey_runtime'|Killed|out of memory|Segmentation fault"


def classify_text(text, planner_tail, model_paths):
    for pattern, kind in INFRA_PATTERNS:
        if re.search(pattern, text):
            return "infra_fail", kind
    for pattern, kind in MODEL_PATTERNS:
        if re.search(pattern, text):
            return "model_fail", kind
    if "worker startup failed" in text:
        env_problem = re.search(ENV_PATTERNS, text + "\n" + planner_tail)
        return ("infra_fail", "planner_startup_env") if env_problem else ("model_fail", "planner_startup")
    if "worker exited or control connection failed" in text:
        died = re.search(r"Killed|out of memory|Segmentation fault|CUDA error", planner_tail)
        return ("infra_fail", "planner_died") if died else ("model_fail", "planner_died")
    frames = re.findall(r'File "([^"]+)"', text)
    for frame in frames:
        if "odyssey_runtime/planner.py" in frame or "odyssey_bridge/planners/" in frame \
                or any(p and frame.startswith(p) for p in model_paths):
            return "model_fail", "planner_exception"
    return "infra_fail", "simulator"


def reason_line(text):
    """The innermost error line. A worker's traceback arrives embedded as a repr (literal \\n)."""
    text = text.replace("\\n", "\n")
    lines = error_lines(text)
    return (lines[-1] if lines else (text.strip().splitlines() or [""])[-1]).strip("'\"} ")[:400]


def classify(run_dir, outcome, model_paths=(), input_error=None):
    """-> dict(status, failure_class, kind, reason, log_tail). ``outcome`` is execute()'s result or None."""
    if input_error is not None:
        return dict(status="input_missing", failure_class="input", kind="input_missing", reason=str(input_error)[:400], log_tail="")
    facts = inspect_run(run_dir)
    rc = outcome["returncode"] if outcome else facts["exit_code"]
    if outcome and outcome.get("stopped_after_finish"):
        rc = 0                    # the run finished; only the process's teardown was stopped
    timed_out = bool(outcome and outcome["timed_out"]) or facts["exit_code"] == 124
    sim_tail = tail(facts["sim_log"], 40)
    planner_tail = tail(facts["planner_log"], 20)
    row, report = facts["row"], facts["report"]
    if timed_out:
        return dict(status="timeout", failure_class="infra", kind="wall_clock",
                    reason=f"no completion within the time limit; last step {facts['last_step']}", log_tail=sim_tail)
    if rc is not None and rc < 0:
        return dict(status="infra_fail", failure_class="infra", kind="signal",
                    reason=f"simulator killed by signal {-rc}" + ("" if error_lines(sim_tail) else " (no traceback: possible OOM kill)"),
                    log_tail=sim_tail)
    if facts["flag"] and not rc:
        if row is not None and not row.get("scoring_error") and row.get("RouteDS") not in (None, ""):
            return dict(status="complete", failure_class=None, kind=None, reason=None, log_tail="")
        why = "metrics csv missing" if row is None else (row.get("scoring_error") or "no dense scoring inputs")
        return dict(status="infra_fail", failure_class="infra", kind="scoring_error", reason=str(why)[:400], log_tail=sim_tail)
    if rc == 0 or (rc is None and facts["exit_code"] == 1):
        if report is not None and not report.get("succeeded", True):
            message = str(report.get("error_message") or "")
            status, kind = classify_text(message, planner_tail, model_paths)
            return dict(status=status, failure_class=status.split("_")[0], kind=kind, reason=reason_line(message),
                        log_tail=sim_tail + ("\n--- planner_worker.log\n" + planner_tail if "planner" in kind else ""))
        return dict(status="infra_fail", failure_class="infra", kind="no_report",
                    reason="exit 0 without the completion flag or a runner report", log_tail=sim_tail)
    if rc:
        text = tail(facts["sim_log"], 200)
        status, kind = classify_text(text, planner_tail, model_paths)
        return dict(status=status, failure_class=status.split("_")[0], kind=kind, reason=reason_line(text),
                    log_tail=sim_tail + ("\n--- planner_worker.log\n" + planner_tail if "planner" in kind else ""))
    return dict(status="infra_fail", failure_class="infra", kind="interrupted", reason="run interrupted before it finished", log_tail=sim_tail)


def result_from(run_dir):
    row, csv_path = read_scene_row(run_dir)
    if row is None:
        return None
    keys = ("scene", "RouteDS", "term_reason", "steps", "RC", "P_SD", "scoring_error",
            "tl_set", "plc_rule")
    out = {k: row.get(k) for k in keys if k in row}
    out["csv"] = csv_path
    return out


def prune(run_dir):
    removed = []
    for rel in HEAVY:
        path = Path(run_dir, rel)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(rel)
    return removed


# ----------------------------------------------------------------------------- campaign
class Campaign:
    def __init__(self, output, log=None):
        self.dir = Path(output)
        self.jobs_dir = self.dir / "jobs"
        self.runs_dir = self.dir / "runs"
        self.lock = threading.Lock()
        self.log_path = self.dir / "eval.log"

    def log(self, message):
        line = f"{utcnow()} {message}"
        print(line, flush=True)
        with self.lock:
            with self.log_path.open("a") as f:
                f.write(line + "\n")

    def record_path(self, job_id):
        return self.jobs_dir / f"{job_id}.json"

    def load_record(self, job):
        path = self.record_path(job.job_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        return dict(schema=JOB_SCHEMA, **job.as_dict(), status="pending", final=False, attempt=0,
                    max_attempts=None, run_dir=None, fingerprint=None, updated_at=None, attempts=[],
                    result=None, failure=None, applied_rules=None)

    def save_record(self, record):
        record["updated_at"] = utcnow()
        write_json_atomic(self.record_path(record["job_id"]), record)

    def records(self):
        out = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                out.append(json.loads(path.read_text()))
            except ValueError:
                continue
        return out


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def needs_run(record, retry_infra, retry_failed):
    """-> (run?, note). The resume rule."""
    status = record["status"]
    if status == "pending":
        return True, "pending"
    if status == "complete":
        return False, "complete"
    if status == "running":
        last = record["attempts"][-1] if record["attempts"] else {}
        if pid_alive(last.get("runner_pid")):
            return False, "running elsewhere"
        return True, "interrupted"
    if status in ("model_fail", "input_missing"):
        return (True, "retry-failed") if retry_failed else (False, status)
    if status in RETRYABLE:
        if record["attempt"] < 1 + retry_infra:
            return True, f"retry {status}"
        return (True, "retry-failed") if retry_failed else (False, f"{status} (retries exhausted)")
    return True, status


def summarize(records, reacts):
    counts = {k: 0 for k in ("total", "pending", "running", "complete", "model_fail", "infra_fail", "timeout", "input_missing", "exhausted")}
    scores = {r: [] for r in reacts}
    for rec in records:
        counts["total"] += 1
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if rec["status"] in RETRYABLE and rec.get("max_attempts") and rec["attempt"] >= rec["max_attempts"]:
            counts["exhausted"] += 1
        if rec.get("result"):
            try:
                scores[rec["react"]].append(float(rec["result"]["RouteDS"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif rec["status"] == "model_fail":
            scores.setdefault(rec["react"], []).append(0.0)
    mean = {}
    for react, values in scores.items():
        expected = sum(1 for r in records if r["react"] == react)
        mean[react] = round(sum(values) / expected, 4) if expected and len(values) == expected else None
    counts["mean_driving_score_den100"] = mean
    return counts


# ----------------------------------------------------------------------------- runner
class Evaluator:
    def __init__(self, args, root):
        self.args, self.root = args, root
        self.environ = dict(os.environ)
        self.scenes_root = Path(args.scenes_root or self.environ.get("ODYSSEY_SCENES_ROOT") or "")
        if not str(self.scenes_root):
            raise ValueError("the published scenes are required: set ODYSSEY_SCENES_ROOT or pass --scenes-root")
        self.agent = load_agent(args.agent, environ=self.environ)
        self.rows = scenes.release_table(self.scenes_root)
        self.reacts = [r for r in re.split(r"[,\s]+", args.react) if r]
        for react in self.reacts:
            if react not in ("nr", "r"):
                raise ValueError(f"--react must list nr and/or r, got {react!r}")
        self.scene_set = args.scenes if args.scenes in SCENE_SETS else "custom"
        self.jobs = plan_jobs(self.rows, args.scenes, self.reacts)
        self.gpus = [g for g in re.split(r"[,\s]+", args.gpus) if g]
        from .launch import gpu_visibility_error
        gpu_error = gpu_visibility_error(self.gpus, self.environ)
        if gpu_error:
            raise ValueError(gpu_error)
        present = installed_gpus()
        absent = [g for g in self.gpus if present is not None and g not in present]
        if absent:
            raise ValueError(f"GPU {','.join(absent)} is not on this host (nvidia-smi lists {', '.join(present) or 'none'})")
        self.workers = plan_workers(self.gpus, args.runs_per_gpu)
        self.slots = assign_slots(self.workers, args.map_slots or self.environ.get("ODYSSEY_MAP_SLOTS"), self.environ)
        self.timeout = args.timeout or (1800 if args.max_steps else 5400)
        self.max_attempts = 1 + args.retry_infra
        name = f"eval_{self.agent.model.name}" + ("" if self.scene_set == "all" else f"_{self.scene_set}")
        self.campaign = Campaign(Path(args.output) if args.output else root / "experiments/simulation" / name)
        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.consecutive_failures = 0
        self.inflight = {}
        self.fingerprint = self.campaign_fingerprint()

    # -- identity --------------------------------------------------------------
    def campaign_fingerprint(self):
        m = self.agent.model
        parts = dict(agent_config_sha256=sha256_file(self.agent.path), profile=self.agent.profile.data,
                     checkpoint=checkpoint_identity(m.checkpoint) if os.path.isfile(m.checkpoint) else {"path": m.checkpoint},
                     model=dict(repo=m.repo, agent_config=m.agent_config, python=m.python, adapter=m.adapter),
                     benchmark=dict(scenes_csv_sha256=sha256_file(self.scenes_root / "scenes.csv"), react=self.reacts,
                                    max_steps=self.args.max_steps, seed=self.args.seed, record=bool(self.args.record)))
        parts["checkpoint"].pop("mtime", None)
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    # -- preflight -------------------------------------------------------------
    def preflight(self):
        problems = []
        from .launch import check_benchmark_environment, EnvironmentRefused
        try:
            check_benchmark_environment(self.environ)
        except EnvironmentRefused as error:
            problems.append(str(error))
        problems += self.agent.verify_paths()
        for var in ("ODYSSEY_SIM_PY", "ODYSSEY_FIXER_PY"):
            value = self.environ.get(var)
            if not value or not os.path.isfile(value):
                problems.append(f"{var} is not set to an existing interpreter (source the host environment first)")
        zoo = Path(self.environ.get("ODYSSEY_ZOO_ROOT", self.root / "OdysseyZoo"))
        if not (zoo / "sdroute/graph.py").is_file():
            problems.append(f"OdysseyZoo not found at {zoo} (ODYSSEY_ZOO_ROOT)")
        for job in {j.scene: j for j in self.jobs}.values():
            folder = self.scenes_root / job.scene
            missing = [f for f in scenes.RELEASE_FILES if not (folder / f).is_file()]
            if missing:
                problems.append(f"{job.scene}: missing {missing}")
        try:
            free_gb = shutil.disk_usage(self.campaign.dir.parent if not self.campaign.dir.exists() else self.campaign.dir).free / 1e9
            if free_gb < MIN_FREE_GB:
                problems.append(f"only {free_gb:.0f} GB free under {self.campaign.dir.parent}; {MIN_FREE_GB} GB required")
        except OSError:
            pass
        if self.args.preflight == "full":
            from .launch import build_run
            for job in self.jobs:
                try:
                    build_run(self.root, self.agent, job.scene, job.react, self.args.max_steps, "0",
                              self.campaign.dir / ".preflight" / job.job_id, record=self.args.record,
                              seed=self.args.seed, environ=self.run_environ(self.workers[0]))
                except Exception as error:
                    problems.append(f"{job.job_id}: {error}")
        return problems

    def run_environ(self, worker):
        return run_environ(self.environ, self.slots[worker], self.agent.model.python, self.scenes_root)

    # -- manifest --------------------------------------------------------------
    def environment_record(self):
        gpus = []
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout
            for line in out.splitlines():
                idx, name, uuid = [x.strip() for x in line.split(",", 2)]
                if idx in self.gpus:
                    gpus.append(dict(index=int(idx), name=name, uuid=uuid))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        fixer = dict(preset=RESTORER)
        try:
            from odyssey_renderer.omnire.restorer_client import FIXER_PRESETS
            fixer_root = Path(self.environ.get("ODYSSEY_FIXER_ROOT", self.root / "OdysseyRenderer/fixer"))
            ckpt = fixer_root / FIXER_PRESETS[RESTORER]
            fixer["checkpoint"] = checkpoint_identity(ckpt) if ckpt.is_file() else {"path": str(ckpt), "missing": True}
        except (ImportError, KeyError, OSError) as error:
            fixer["error"] = repr(error)
        return dict(host=socket.gethostname(), gpus=gpus, runs_per_gpu=self.args.runs_per_gpu, map_slots=self.slots,
                    interpreters=dict(simulation=self.environ.get("ODYSSEY_SIM_PY"), planner=self.agent.model.python,
                                      fixer=self.environ.get("ODYSSEY_FIXER_PY")),
                    fixer=fixer, code=git_state(self.root),
                    env={k: self.environ.get(k) for k in ("ODYSSEY_ZOO_ROOT", "ODYSSEY_MODELS_ROOT", "ODYSSEY_SCENES_ROOT",
                                                            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")})

    def write_inputs(self):
        inputs = self.campaign.dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.agent.path, inputs / "agent.yaml")
        profile = dict(self.agent.profile.data)
        if self.args.seed is not None:
            profile["seed"] = self.args.seed
        write_json_atomic(inputs / "profile.json", profile)
        shutil.copyfile(self.scenes_root / "scenes.csv", inputs / "scenes.csv")

    def rebuild_manifest(self, created_at=None):
        with self.campaign.lock:
            records = self.campaign.records()
            path = self.campaign.dir / "manifest.json"
            previous = {}
            if path.exists():
                try:
                    previous = json.loads(path.read_text())
                except ValueError:
                    previous = {}
            m = self.agent.model
            manifest = dict(
                schema=MANIFEST_SCHEMA,
                campaign=dict(name=self.campaign.dir.name, dir=str(self.campaign.dir), fingerprint=self.fingerprint,
                              created_at=(previous.get("campaign") or {}).get("created_at") or created_at or utcnow(),
                              updated_at=utcnow(), argv=sys.argv[1:], runner_version=1),
                benchmark=dict(scenes_root=str(self.scenes_root), scenes_csv_sha256=sha256_file(self.scenes_root / "scenes.csv"),
                               scenes=sorted({j.scene for j in self.jobs}), reacts=self.reacts, jobs_total=len(self.jobs),
                               max_steps=self.args.max_steps, seed=self.args.seed, record=bool(self.args.record),
                               restorer=RESTORER,
                               denominator=len({j.scene for j in self.jobs}), scene_set=self.scene_set,
                               debug_set=self.scene_set == "debug",
                               timeout_s=self.timeout, retry_infra=self.args.retry_infra,
                               # every planned job, so the merger's expected set does not depend on what ran
                               job_table=[dict(scene=j.scene, token=j.token, react=j.react) for j in self.jobs]),
                model=dict(name=m.name,
                           agent_config=dict(path=str(self.agent.path), sha256=sha256_file(self.agent.path), copy="inputs/agent.yaml"),
                           profile=dict(copy="inputs/profile.json", planner_id=self.agent.profile.data["planner_id"],
                                        cameras=list(self.agent.profile.cameras), navigation=self.agent.profile.navigation),
                           python=m.python, repo=m.repo, repo_commit=git_state(m.repo).get("commit"),
                           hydra_agent_config=m.agent_config, adapter=m.adapter,
                           checkpoint=checkpoint_identity(m.checkpoint) if os.path.isfile(m.checkpoint) else {"path": m.checkpoint}),
                environment=previous.get("environment") or self.environment_record(),
                summary=summarize(records, self.reacts),
                jobs=[dict(job_id=r["job_id"], scene=r["scene"], react=r["react"], status=r["status"],
                           final=r["final"], attempt=r["attempt"], run_dir=r["run_dir"],
                           RouteDS=(r.get("result") or {}).get("RouteDS"),
                           failure_class=(r.get("failure") or {}).get("class"), kind=(r.get("failure") or {}).get("kind"),
                           reason=(r.get("failure") or {}).get("reason"),
                           duration_s=(r["attempts"][-1].get("duration_s") if r["attempts"] else None),
                           gpu=(r["attempts"][-1].get("gpu") if r["attempts"] else None)) for r in records],
            )
            write_json_atomic(path, manifest)
            return manifest

    # -- one job ---------------------------------------------------------------
    def run_job(self, job, worker):
        from .launch import build_run, materialize, execute, EnvironmentRefused

        gpu = gpu_of(worker)
        record = self.campaign.load_record(job)
        attempt = record["attempt"] + 1
        run_dir = self.campaign.runs_dir / job.job_id / f"attempt_{attempt:02d}"
        while run_dir.exists():                         # a crashed runner may have left this attempt behind
            attempt += 1
            run_dir = self.campaign.runs_dir / job.job_id / f"attempt_{attempt:02d}"
        code = git_state(self.root)
        info = dict(attempt=attempt, run_dir=str(run_dir.relative_to(self.campaign.dir)), status="running",
                    failure_class=None, kind=None, reason=None, returncode=None, exit_code=None, flag=False,
                    timed_out=False, started_at=utcnow(), ended_at=None, duration_s=None, report_duration_s=None,
                    gpu=gpu, worker=worker, map_slot=self.slots[worker], runner_pid=os.getpid(), code_commit=code.get("commit"),
                    code_dirty=code.get("dirty"), last_step=None, scoring_error=None,
                    log=str((run_dir / "simulation.log").relative_to(self.campaign.dir)), log_tail="", pruned=[])
        record.update(status="running", final=False, attempt=attempt, max_attempts=self.max_attempts,
                      run_dir=info["run_dir"], fingerprint=self.fingerprint)
        record["attempts"].append(info)
        self.campaign.save_record(record)
        self.campaign.log(f"start {job.job_id} attempt {attempt} gpu {worker}")
        verdict, outcome = None, None
        try:
            spec = build_run(self.root, self.agent, job.scene, job.react, self.args.max_steps, gpu, run_dir,
                             record=self.args.record, seed=self.args.seed, environ=self.run_environ(worker))
        except (ValueError, FileNotFoundError, OSError, EnvironmentRefused) as error:
            verdict = classify(run_dir, None, input_error=error)
        if verdict is None:
            materialize(spec, run_dir)
            try:
                outcome = execute(spec, timeout=self.timeout,
                                  on_start=lambda proc: self.inflight.__setitem__(job.job_id, proc))
            finally:
                self.inflight.pop(job.job_id, None)
            verdict = classify(run_dir, outcome, model_paths=(self.agent.model.repo,))
            report = read_runner_report(run_dir)
            info.update(returncode=outcome["returncode"], exit_code=outcome["status"], timed_out=outcome["timed_out"],
                        stopped_after_finish=outcome.get("stopped_after_finish", False),
                        started_at=outcome["started_at"], ended_at=outcome["ended_at"], duration_s=outcome["duration_s"],
                        flag=outcome["completed"], report_duration_s=(report or {}).get("duration_s"),
                        last_step=last_step(run_dir))
            row, _ = read_scene_row(run_dir)
            info["scoring_error"] = (row or {}).get("scoring_error") or None
            if verdict["status"] == "complete" and not self.args.keep_outputs:
                info["pruned"] = prune(run_dir)
        else:
            info.update(ended_at=utcnow(), duration_s=0.0)
        info.update(status=verdict["status"], failure_class=verdict["failure_class"], kind=verdict["kind"],
                    reason=verdict["reason"], log_tail=verdict["log_tail"])
        record["status"] = verdict["status"]
        record["applied_rules"] = applied_rules(run_dir)
        if verdict["status"] == "complete":
            record["result"], record["failure"], record["final"] = result_from(run_dir), None, True
        else:
            record["result"] = None
            record["failure"] = dict(class_=verdict["failure_class"], kind=verdict["kind"], reason=verdict["reason"],
                                     log=info["log"], log_tail=verdict["log_tail"])
            record["failure"]["class"] = record["failure"].pop("class_")
            retry = verdict["status"] in RETRYABLE and attempt < self.max_attempts
            record["final"] = not retry
        self.campaign.save_record(record)
        self.rebuild_manifest()
        self.campaign.log(f"end   {job.job_id} attempt {attempt}: {verdict['status']}"
                          + (f" [{verdict['kind']}] {verdict['reason']}" if verdict["reason"] else "")
                          + (f" RouteDS={record['result'].get('RouteDS')}" if record.get("result") else ""))
        return record

    def worker(self, worker):
        while not self.stop.is_set():
            try:
                job = self.queue.get(timeout=1)
            except queue.Empty:
                if self.queue_closed:
                    return
                continue
            try:
                record = self.run_job(job, worker)
            except Exception as error:                      # the runner itself failed; keep the campaign alive
                self.campaign.log(f"error {job.job_id} on gpu {worker}: {error!r}")
                record = None
            finally:
                self.queue.task_done()
            if record is None:
                continue
            if record["status"] == "complete":
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                if self.args.max_consecutive_failures and self.consecutive_failures >= self.args.max_consecutive_failures:
                    self.campaign.log(f"abort: {self.consecutive_failures} consecutive failures")
                    self.stop.set()
                if record["status"] in RETRYABLE and not record["final"]:
                    self.queue.put(job)

    def terminate_inflight(self):
        """Stop every running simulator with its planner/Fixer workers, then mark the jobs interrupted.

        execute() runs in the per-GPU threads, which never see the main thread's KeyboardInterrupt,
        and each simulator is in its own session -- so the runner has to stop them itself."""
        from .launch import terminate_tree
        stoppers = []
        for job_id, proc in list(self.inflight.items()):
            self.campaign.log(f"interrupt {job_id}")
            stoppers.append(threading.Thread(target=terminate_tree, args=(proc,), daemon=True))
        for t in stoppers:
            t.start()
        for t in stoppers:
            t.join()
        for record in self.campaign.records():
            if record["status"] == "running" and record["attempts"] and record["attempts"][-1].get("runner_pid") == os.getpid():
                info = record["attempts"][-1]
                info.update(status="infra_fail", failure_class="infra", kind="interrupted", reason="runner interrupted",
                            ended_at=utcnow())
                record.update(status="infra_fail", final=record["attempt"] >= self.max_attempts,
                              failure=dict(**{"class": "infra"}, kind="interrupted", reason="runner interrupted", log=info["log"], log_tail=""))
                self.campaign.save_record(record)

    # -- entry -----------------------------------------------------------------
    def status_table(self):
        records = {r["job_id"]: r for r in self.campaign.records()}
        lines = []
        for job in self.jobs:
            r = records.get(job.job_id)
            if r is None:
                lines.append(f"{job.job_id:28s} pending")
                continue
            extra = ""
            if r.get("result"):
                extra = f" RouteDS={r['result'].get('RouteDS')}"
            elif r.get("failure"):
                extra = f" [{r['failure'].get('kind')}] {r['failure'].get('reason')}"
            lines.append(f"{job.job_id:28s} {r['status']:13s} attempt {r['attempt']}{extra}")
        return "\n".join(lines)

    def run(self):
        c = self.campaign
        c.dir.mkdir(parents=True, exist_ok=True)
        c.jobs_dir.mkdir(exist_ok=True)
        c.runs_dir.mkdir(exist_ok=True)
        lock_file = (c.dir / ".eval.lock").open("w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another eval is running on {c.dir}", file=sys.stderr)
            return 2
        manifest_path = c.dir / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous.get("campaign", {}).get("fingerprint") not in (None, self.fingerprint):
                print(f"{c.dir} was started with a different agent config, checkpoint, scene set or run options "
                      f"(fingerprint {previous['campaign']['fingerprint'][:12]} vs {self.fingerprint[:12]}); "
                      "choose another --output", file=sys.stderr)
                return 2
        self.write_inputs()
        todo, skipped = [], {}
        for job in self.jobs:
            record = c.load_record(job)
            run, note = needs_run(record, self.args.retry_infra, self.args.retry_failed)
            if run:
                todo.append(job)
            else:
                skipped[note] = skipped.get(note, 0) + 1
        self.rebuild_manifest(created_at=utcnow())
        c.log(f"campaign {c.dir}: {len(self.jobs)} jobs, {len(todo)} to run, skipped {skipped or 'none'}; "
              f"gpus {self.gpus} x {self.args.runs_per_gpu} slots {self.slots}; timeout {self.timeout}s; "
              f"retry_infra {self.args.retry_infra}")
        for job in todo:
            self.queue.put(job)
        self.queue_closed = True
        def interrupt(signum, frame):              # SIGTERM stops the campaign like Ctrl-C
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        threads = [threading.Thread(target=self.worker, args=(w,), name=f"gpu{w}", daemon=True) for w in self.workers]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.5)
        except KeyboardInterrupt:
            c.log("interrupted; stopping workers")
            self.stop.set()
            self.terminate_inflight()
            self.rebuild_manifest()
            return 130
        manifest = self.rebuild_manifest()
        summary = manifest["summary"]
        c.log(f"done: {summary['complete']} complete, {summary['model_fail']} model_fail, "
              f"{summary['infra_fail']} infra_fail, {summary['timeout']} timeout, {summary['input_missing']} input_missing "
              f"(exhausted {summary['exhausted']}); mean DS/den100 {summary['mean_driving_score_den100']}")
        print(f"campaign: {c.dir}\nnext: python OdysseyBenchmark/tools/merge_results.py -f {c.dir}")
        ok = all(r["status"] in ("complete", "model_fail") for r in c.records()) and not self.stop.is_set()
        return 0 if ok else 1


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m odyssey_runtime eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="agent config (OdysseyBenchmark/agents/<model>.yaml or your own)")
    parser.add_argument("--output", help="campaign directory (default experiments/simulation/eval_<model>)")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices, one worker each")
    parser.add_argument("--runs-per-gpu", type=int, default=1, help="simulations running at once on each GPU (default 1)")
    parser.add_argument("--map-slots", help="comma-separated nuPlan map directories, one per run at a time "
                        "(default: derived from NUPLAN_MAPS_ROOT, one per GPU; required with --runs-per-gpu > 1)")
    parser.add_argument("--scenes", default="all",
                        help="all (the benchmark) | mini (the published 30-scene set) | debug | names/codes, comma-separated | @file")
    parser.add_argument("--react", default="nr,r", help="traffic modes to run: nr (log replay), r (reactive), or both")
    parser.add_argument("--max-steps", type=max_steps_arg,
                        help="stop each episode after this many 0.1 s steps (>= 20, multiple of 5); default: the scene's full horizon")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--record", action="store_true", help="keep model-input images and plans (large)")
    parser.add_argument("--timeout", type=float, help="seconds per attempt (default 5400, or 1800 with --max-steps)")
    parser.add_argument("--retry-infra", type=int, default=2, help="automatic re-runs after an infrastructure failure or timeout")
    parser.add_argument("--retry-failed", action="store_true", help="also re-run jobs that failed because of the model")
    parser.add_argument("--keep-outputs", action="store_true", help="do not delete images/frames of successful runs")
    parser.add_argument("--preflight", choices=["files", "full"], default="files",
                        help="files: check paths and scene files (default); full: also compose every job's run (about 10 s for 200 jobs)")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="stop the campaign after this many failed jobs in a row (default 5)")
    parser.add_argument("--scenes-root", help="the odyssey-scenes download (default ODYSSEY_SCENES_ROOT)")
    parser.add_argument("--dry", action="store_true", help="print the job table and the first job's command; run nothing")
    parser.add_argument("--status", action="store_true", help="print the campaign's job table; run nothing")
    return parser.parse_args(argv)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]          # the repository root
    if str(root / "OdysseyTrafficAgent") not in sys.path:
        sys.path.insert(0, str(root / "OdysseyTrafficAgent"))      # build_run imports the signal timetable
    args = parse_args(argv)
    if not is_agent_config(args.agent):
        print("--agent must be an agent config (.yaml)", file=sys.stderr)
        return 2
    try:
        ev = Evaluator(args, root)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.status:
        print(ev.status_table())
        return 0
    problems = ev.preflight()
    if problems:
        print("preflight failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    if args.dry:
        from .launch import build_run, dry_view
        print(f"campaign: {ev.campaign.dir}\ngpus/slots: {ev.slots}\njobs ({len(ev.jobs)}):")
        for job in ev.jobs:
            print(f"  {job.job_id}")
        job = ev.jobs[0]
        spec = build_run(root, ev.agent, job.scene, job.react, args.max_steps, ev.gpus[0],
                         ev.campaign.dir / "runs" / job.job_id / "attempt_01", record=args.record, seed=args.seed,
                         environ=ev.run_environ(ev.workers[0]))
        print(json.dumps(dry_view(spec), indent=2))
        return 0
    return ev.run()
