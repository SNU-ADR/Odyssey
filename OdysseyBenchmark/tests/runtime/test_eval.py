"""The batch runner: job planning, map slots, failure classes, resume, records, the launcher's execute()."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from odyssey_runtime import eval as ev
from odyssey_runtime.launch import execute, sweep_workers, worker_processes

ROOT = Path(__file__).resolve().parents[3]


def rows(numbers=(1, 11, 75, 2)):
    return [dict(scene=f"odyssey_scene{n:03d}", token=f"tok{n}") for n in numbers]


def test_plan_jobs_all_debug_and_explicit():
    table = rows()
    jobs = ev.plan_jobs(table, "all", ["nr", "r"])
    assert [j.job_id for j in jobs][:3] == ["odyssey_scene001_nr", "odyssey_scene001_r", "odyssey_scene002_nr"]
    assert len(jobs) == 8 and jobs[0].token == "tok1"
    assert jobs[0].as_dict() == dict(job_id="odyssey_scene001_nr", scene="odyssey_scene001", react="nr", token="tok1")
    debug = ev.plan_jobs(table, "debug", ["nr"])                    # react is fixed per debug scene
    assert [(j.scene, j.react) for j in debug] == [("odyssey_scene011", "nr"), ("odyssey_scene075", "r"),
                                                   ("odyssey_scene002", "nr")]
    chosen = ev.plan_jobs(table, "scene075, 001", ["r"])
    assert [j.job_id for j in chosen] == ["odyssey_scene075_r", "odyssey_scene001_r"]
    with pytest.raises(ValueError, match="unknown scene"):
        ev.plan_jobs(table, ["x001"], ["nr"])                                   # only published names select a scene
    with pytest.raises(ValueError, match="debug scene"):
        ev.plan_jobs(rows((1,)), "debug", ["nr"])


def test_plan_jobs_mini_is_the_published_thirty_scenes():
    table = [dict(scene=n, token=f"t{i}") for i, n in enumerate(ev.MINI_SET + ("odyssey_scene001",), start=1)]
    jobs = ev.plan_jobs(table, "mini", ["nr", "r"])
    assert len(jobs) == 60 and len(ev.MINI_SET) == len(set(ev.MINI_SET)) == 30
    assert [j.scene for j in jobs][::2] == list(ev.MINI_SET)
    assert {j.react for j in jobs} == {"nr", "r"}
    with pytest.raises(ValueError, match="unknown scene"):           # the set needs every one of its scenes
        ev.plan_jobs(table[:5], "mini", ["nr"])


def test_scene_list_file(tmp_path):
    listing = tmp_path / "scenes.txt"
    listing.write_text("# a comment\nodyssey_scene002\n001\n")
    jobs = ev.plan_jobs(rows(), f"@{listing}", ["nr"])
    assert [j.scene for j in jobs] == ["odyssey_scene002", "odyssey_scene001"]


def slots(tmp_path, n=4):
    for i in range(n):
        d = tmp_path / f"final_gpu{i}"
        d.mkdir()
        (d / "nuplan-maps-v1.0.json").write_text("{}")
    return tmp_path


def test_slots_derive_from_map_root_index(tmp_path):
    base = slots(tmp_path)
    environ = {"NUPLAN_MAPS_ROOT": str(base / "final_gpu1")}      # any slot names the family; the GPU index picks one
    assert ev.assign_slots(["0", "2"], None, environ) == {"0": str(base / "final_gpu0"), "2": str(base / "final_gpu2")}
    with pytest.raises(ValueError, match="duplicate workers"):
        ev.assign_slots(["0", "0"], f"{base / 'final_gpu2'},{base / 'final_gpu3'}", environ)
    with pytest.raises(ValueError, match="not a nuPlan map directory"):
        ev.assign_slots(["7"], None, environ)
    with pytest.raises(ValueError, match="cannot derive"):
        ev.assign_slots(["0"], None, {"NUPLAN_MAPS_ROOT": str(base / "maps")})
    explicit = ev.assign_slots(["0", "1"], f"{base / 'final_gpu2'},{base / 'final_gpu3'}", environ)
    assert explicit == {"0": str(base / "final_gpu2"), "1": str(base / "final_gpu3")}
    with pytest.raises(ValueError, match="distinct"):
        ev.assign_slots(["0", "1"], f"{base / 'final_gpu2'},{base / 'final_gpu2'}", environ)
    with pytest.raises(ValueError, match="lists 1 slots for 2"):
        ev.assign_slots(["0", "1"], str(base / "final_gpu2"), environ)


def test_several_runs_per_gpu_get_their_own_workers_and_map_slots(tmp_path):
    base = slots(tmp_path, 4)
    environ = {"NUPLAN_MAPS_ROOT": str(base / "final_gpu0")}
    workers = ev.plan_workers(["0", "1"], 2)
    assert workers == ["0", "1", "0.2", "1.2"] and [ev.gpu_of(w) for w in workers] == ["0", "1", "0", "1"]
    assert ev.plan_workers(["2", "3"]) == ["2", "3"]
    with pytest.raises(ValueError, match="duplicate GPUs"):
        ev.plan_workers(["0", "0"])
    with pytest.raises(ValueError, match="at least 1"):
        ev.plan_workers(["0"], 0)
    with pytest.raises(ValueError, match="several runs per GPU need one map directory each"):
        ev.assign_slots(workers, None, environ)
    listed = ",".join(str(base / f"final_gpu{i}") for i in range(4))
    assert ev.assign_slots(workers, listed, environ) == {w: str(base / f"final_gpu{i}") for i, w in enumerate(workers)}


def test_run_environ_sets_slot_interpreter_and_scenes():
    env = ev.run_environ({"PATH": "/bin", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "0"}, "/maps/final_gpu2", "/envs/py", "/scenes")
    assert env["NUPLAN_MAPS_ROOT"] == "/maps/final_gpu2"
    assert env["ODYSSEY_PLANNER_PY"] == "/envs/py" and env["ODYSSEY_SCENES_ROOT"] == "/scenes"
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "0" and env["PATH"] == "/bin"


# ------------------------------------------------------------------ classifier
def make_run(tmp_path, *, exit_code=0, flag=True, csv="ok", report=None, sim_log="", planner_log="", name="run"):
    run = tmp_path / name
    (run / "odyssey_output").mkdir(parents=True)
    (run / "runtime").mkdir()
    if exit_code is not None:
        (run / "exit_code.txt").write_text(f"{exit_code}\n")
    if flag:
        (run / "odyssey_output/simulation_completed.flag").write_text("")
    header = "scene,react,steps,term_reason,RouteDS,RC,P_SD,scoring_error,tl_set,plc_rule\n"
    if csv == "ok":
        (run / "odyssey_output/routeds_NR.csv").write_text(
            header + "odyssey_scene001,nr,1828,destination_arrival,11.52943,1.0,1.0,,tlc_timetable,plc\n")
    elif csv == "error":
        (run / "odyssey_output/routeds_NR.csv").write_text(
            header + "odyssey_scene001,nr,1828,destination_arrival,,,,KeyError: rc,tlc_timetable,plc\n")
    elif csv == "unscored":
        (run / "odyssey_output/routeds_NR.csv").write_text(
            header + "odyssey_scene001,nr,12,stopped,,,,,,\n")
    if report is not None:
        (run / "odyssey_output/runner_report.json").write_text(json.dumps([dict(
            scenario_name="tok1", succeeded=False, duration_s=12.5, error_message=report)]))
    (run / "simulation.log").write_text(sim_log)
    (run / "runtime/planner_worker.log").write_text(planner_log)
    return run


PLANNER_TB = ('Traceback (most recent call last):\n  File "/x/odyssey_runtime/planner.py", line 203, in handle\n'
              '    plan = self.planner.infer(list(self.history))\n  File "/x/OdysseyBenchmark/odyssey_bridge/planners/base.py", line 240\n'
              'RuntimeError: Traceback (most recent call last):\n  File "/repo/navsim/agents/m.py", line 5\nKeyError: \'foo\'\n')
SIM_TB = ('Traceback (most recent call last):\n  File "/x/OdysseyTrafficAgent/odyssey/envs/base_env.py", line 200\n'
          'ZeroDivisionError: division by zero\n')


@pytest.mark.parametrize("kw, expect", [
    (dict(), ("complete", None)),
    (dict(csv="error"), ("infra_fail", "scoring_error")),
    (dict(csv="unscored"), ("infra_fail", "scoring_error")),
    (dict(csv=None), ("infra_fail", "scoring_error")),
    (dict(exit_code=1, flag=False, report=PLANNER_TB), ("model_fail", "planner_exception")),
    (dict(exit_code=1, flag=False, report="RuntimeError: Traceback ...\nTimeoutError: worker response deadline expired\n"), ("infra_fail", "planner_timeout")),
    (dict(exit_code=1, flag=False, report="ValueError: planner output differs from declared finite trajectory shape\n"), ("model_fail", "bad_plan")),
    (dict(exit_code=1, flag=False, report="ValueError: planner output shape [6, 3] differs from the declared output.shape [8, 3]\n"), ("model_fail", "bad_plan")),
    (dict(exit_code=1, flag=False, report="ValueError: planner output contains non-finite values\n"), ("model_fail", "bad_plan")),
    (dict(exit_code=1, flag=False, report="ValueError: status: declared in feature_shapes, but the model has no such feature (it has ['x'])\n"), ("model_fail", "profile_mismatch")),
    (dict(exit_code=1, flag=False, report="ValueError: x: model feature shape [1, 2] differs from profile [1, 3]\n"), ("model_fail", "profile_mismatch")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="ValueError: native planner PLAN_DT differs from profile: the adapter emits poses every 0.5 s, output.plan_dt declares 0.1 s\n"), ("model_fail", "profile_mismatch")),
    (dict(exit_code=1, flag=False, report=SIM_TB), ("infra_fail", "simulator")),
    (dict(exit_code=1, flag=False, report="RuntimeError: Fixer checkpoint identity mismatch\n"), ("infra_fail", "fixer")),
    (dict(exit_code=1, flag=False), ("infra_fail", "no_report")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="Error executing job\nRuntimeError: worker startup failed: {'ready': False, 'error': 'size mismatch for layer'}; log=x\n"), ("model_fail", "planner_startup")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="RuntimeError: worker startup failed: {...}\n", planner_log="torch.OutOfMemoryError: CUDA out of memory\n"), ("infra_fail", "planner_startup_env")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="RuntimeError: worker exited or control connection failed\n", planner_log="Killed\n"), ("infra_fail", "planner_died")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="RuntimeError: worker exited or control connection failed\n", planner_log="ValueError: x\n"), ("model_fail", "planner_died")),
    (dict(exit_code=1, flag=False, csv=None, sim_log="ValueError: CAM_F0: native sensor history [3] differs from profile [2]\n"), ("model_fail", "profile_mismatch")),
    (dict(exit_code=124, flag=False, csv=None), ("timeout", "wall_clock")),
    (dict(exit_code=None, flag=False, csv=None), ("infra_fail", "interrupted")),
])
def test_classifier_decision_table(tmp_path, kw, expect):
    run = make_run(tmp_path, **kw)
    outcome = None
    code = kw.get("exit_code", 0)
    if code is not None:
        outcome = dict(returncode=(0 if code in (0, 1) and kw.get("report") is not None or code == 0 else (None if code == 124 else code)),
                       timed_out=code == 124, completed=kw.get("flag", True))
        if code == 1 and kw.get("report") is None and kw.get("sim_log"):
            outcome["returncode"] = 1
        if code == 1 and kw.get("report") is None and not kw.get("sim_log"):
            outcome["returncode"] = 0
    verdict = ev.classify(run, outcome, model_paths=("/repo",))
    assert (verdict["status"], verdict["kind"]) == expect, verdict


def test_signal_death_is_infrastructure(tmp_path):
    run = make_run(tmp_path, exit_code=-9, flag=False, csv=None)
    verdict = ev.classify(run, dict(returncode=-9, timed_out=False, completed=False))
    assert (verdict["status"], verdict["kind"]) == ("infra_fail", "signal") and "signal 9" in verdict["reason"]


def test_input_error_is_final_and_named(tmp_path):
    verdict = ev.classify(tmp_path / "none", None, input_error=FileNotFoundError("route.npz missing"))
    assert verdict["status"] == "input_missing" and "route.npz" in verdict["reason"]


def test_result_reads_the_result_row(tmp_path):
    run = make_run(tmp_path)
    result = ev.result_from(run)
    assert result["RouteDS"] == "11.52943" and result["term_reason"] == "destination_arrival" and result["scene"] == "odyssey_scene001"
    assert result["csv"].endswith("routeds_NR.csv")


def test_prune_removes_only_heavy_outputs(tmp_path):
    run = make_run(tmp_path)
    for rel in ("odyssey_output/sensor_blobs/odyssey_scene001", "frames", "plan_traj", "runtime/model_audit"):
        (run / rel).mkdir(parents=True)
    removed = ev.prune(run)
    assert set(removed) == {"odyssey_output/sensor_blobs", "frames", "plan_traj", "runtime/model_audit"}
    assert (run / "odyssey_output/routeds_NR.csv").exists()
    assert (run / "simulation.log").exists() and not (run / "frames").exists()


# ------------------------------------------------------------------ resume
def record(status, attempt=1, pid=None):
    rec = dict(status=status, attempt=attempt, attempts=[dict(runner_pid=pid)] if pid is not None else [])
    return rec


def test_resume_rules():
    assert ev.needs_run(record("pending", 0), 2, False) == (True, "pending")
    assert ev.needs_run(record("complete"), 2, False) == (False, "complete")
    assert ev.needs_run(record("model_fail"), 2, False) == (False, "model_fail")
    assert ev.needs_run(record("model_fail"), 2, True) == (True, "retry-failed")
    assert ev.needs_run(record("infra_fail", 1), 2, False) == (True, "retry infra_fail")
    assert ev.needs_run(record("timeout", 3), 2, False) == (False, "timeout (retries exhausted)")
    assert ev.needs_run(record("timeout", 3), 2, True) == (True, "retry-failed")
    assert ev.needs_run(record("running", 1, pid=os.getpid()), 2, False) == (False, "running elsewhere")
    assert ev.needs_run(record("running", 1, pid=2 ** 22 + 12345), 2, False) == (True, "interrupted")


def test_atomic_json_write_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / "j.json"
    ev.write_json_atomic(path, {"a": 1})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        ev.write_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_summary_counts_and_fixed_denominator():
    recs = [dict(job_id="a_nr", react="nr", status="complete", attempt=1, max_attempts=3, result=dict(RouteDS="50")),
            dict(job_id="b_nr", react="nr", status="model_fail", attempt=1, max_attempts=3, result=None),
            dict(job_id="a_r", react="r", status="infra_fail", attempt=3, max_attempts=3, result=None)]
    s = ev.summarize(recs, ["nr", "r"])
    assert s["complete"] == 1 and s["model_fail"] == 1 and s["exhausted"] == 1
    assert s["mean_driving_score_den100"] == {"nr": 25.0, "r": None}


# ------------------------------------------------------------------ execute()
def spec_for(tmp_path, command):
    out = tmp_path / "run"
    out.mkdir()
    return dict(command=command, env=dict(os.environ, ODYSSEY_ROOT=str(ROOT)), output=str(out))


def test_execute_success_requires_the_flag(tmp_path):
    spec = spec_for(tmp_path, [sys.executable, "-c", "print('hi')"])
    result = execute(spec)
    assert result["status"] == 1 and result["returncode"] == 0 and not result["completed"]
    assert (tmp_path / "run/exit_code.txt").read_text() == "1\n"
    assert (tmp_path / "run/simulation.log").read_text() == "hi\n"
    other = tmp_path / "ok"
    other.mkdir()
    spec = spec_for(other, [sys.executable, "-c",
                            "import os, pathlib; p = pathlib.Path(os.environ['OUT'], 'odyssey_output/simulation_completed.flag'); "
                            "p.parent.mkdir(parents=True); p.write_text('')"])
    spec["env"]["OUT"] = spec["output"]
    result = execute(spec)
    assert result["status"] == 0 and result["completed"] and result["duration_s"] >= 0
    assert (other / "run/exit_code.txt").read_text() == "0\n"


def test_execute_timeout_kills_the_process_group(tmp_path):
    # A duration no other process on a shared host uses, so pgrep finds only this test's sleeps.
    duration = f"60.{os.getpid()}"
    spec = spec_for(tmp_path, ["bash", "-c", f"sleep {duration} & sleep {duration}"])
    start = time.time()
    result = execute(spec, timeout=1, grace_s=1)
    assert result["status"] == 124 and result["timed_out"] and time.time() - start < 15
    assert (tmp_path / "run/exit_code.txt").read_text() == "124\n"
    time.sleep(0.5)
    alive = subprocess.run(["pgrep", "-f", f"sleep {duration}"], capture_output=True, text=True).stdout.split()
    assert not alive, alive



def test_a_finished_run_stuck_in_teardown_is_stopped_and_counts_as_finished(tmp_path, monkeypatch):
    from odyssey_runtime import launch
    monkeypatch.setattr(launch, "TEARDOWN_GRACE_S", 1)
    monkeypatch.setattr(launch, "POLL_S", 0.2)
    spec = spec_for(tmp_path, [sys.executable, "-c",
                               "import os, pathlib, time; o = pathlib.Path(os.environ['OUT'], 'odyssey_output'); "
                               "o.mkdir(parents=True); (o / 'simulation_completed.flag').write_text(''); "
                               "(o / 'runner_report.json').write_text('[]'); time.sleep(600)"])
    spec["env"]["OUT"] = spec["output"]
    start = time.time()
    result = execute(spec, timeout=300, grace_s=1)
    assert result["stopped_after_finish"] and result["completed"] and not result["timed_out"]
    assert result["status"] == 0 and result["returncode"] < 0 and time.time() - start < 30
    assert (tmp_path / "run/exit_code.txt").read_text() == "0\n"


def test_a_run_stopped_after_it_finished_is_complete(tmp_path):
    run = make_run(tmp_path)
    verdict = ev.classify(run, dict(returncode=-15, timed_out=False, completed=True, stopped_after_finish=True))
    assert verdict["status"] == "complete"
    verdict = ev.classify(run, dict(returncode=-15, timed_out=False, completed=True))
    assert (verdict["status"], verdict["kind"]) == ("infra_fail", "signal")

def test_a_run_in_a_worker_thread_is_stopped_from_outside(tmp_path):
    """The batch runner executes runs in per-GPU threads; on Ctrl-C/SIGTERM the main thread
    stops them through the process handed to on_start (the thread never sees the interrupt)."""
    import threading
    from odyssey_runtime.launch import terminate_tree
    spec = spec_for(tmp_path, ["bash", "-c", "sleep 61 & sleep 61"])
    started, result = {}, {}
    thread = threading.Thread(target=lambda: result.update(execute(spec, on_start=lambda proc: started.update(proc=proc))))
    thread.start()
    for _ in range(100):
        if "proc" in started:
            break
        time.sleep(0.05)
    time.sleep(0.3)
    terminate_tree(started["proc"], grace_s=1)
    thread.join(timeout=15)
    assert not thread.is_alive() and result["status"] not in (0, 124)
    time.sleep(0.5)
    alive = subprocess.run(["pgrep", "-f", "sleep 61"], capture_output=True, text=True).stdout.split()
    assert not alive, alive


def test_worker_sweep_matches_only_this_parent(monkeypatch):
    table = [(11, ["python", "-m", "odyssey_runtime.worker", "5", "6", "64", "odyssey_runtime.planner:PlannerWorker", "100"]),
             (12, ["python", "-m", "odyssey_runtime.worker", "5", "6", "64", "odyssey_runtime.restorer:FixerWorker", "100"]),
             (13, ["python", "-m", "odyssey_runtime.worker", "5", "6", "64", "odyssey_runtime.planner:PlannerWorker", "200"]),
             (14, ["python", "run_simulation.py", "100"])]
    assert worker_processes(100, scan=lambda: table) == [11, 12]
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    remaining = {"n": 0}

    def scan():
        remaining["n"] += 1
        return table if remaining["n"] == 1 else []

    assert sweep_workers(100, grace_s=1, scan=scan) == []
    assert [pid for pid, _ in killed] == [11, 12]


def test_eval_help_runs_without_a_profile():
    out = subprocess.run([sys.executable, "-m", "odyssey_runtime", "eval", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "--agent" in out.stdout and "--retry-infra" in out.stdout


def test_applied_rules_come_from_the_launch_command(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert ev.applied_rules(run) is None
    (run / "launch.json").write_text(json.dumps(dict(command=[
        "python", "run_simulation.py", "spawn_ego_tight_ahead_gate=false", "tlc_timetable_set=tlc_timetable", "tl_set=tlc_timetable"])))
    assert ev.applied_rules(run) == dict(gate_off=True, tlc_timetable_set="tlc_timetable", tl_control_path=False)


def test_result_row_of_a_run_written_before_the_output_folder_was_flattened(tmp_path):
    run = tmp_path / "old"
    (run / "odyssey_output/openscene_format").mkdir(parents=True)
    (run / "odyssey_output/openscene_format/routeds_R.csv").write_text("scene,react,RouteDS\nodyssey_scene001,r,3.0\n")
    row, path = ev.read_scene_row(run)
    assert row["RouteDS"] == "3.0" and path == "odyssey_output/openscene_format/routeds_R.csv"
