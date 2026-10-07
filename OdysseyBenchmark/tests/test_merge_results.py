"""OdysseyBenchmark/tools/merge_results.py: fixed-denominator scoring, status rules, column union, pooled PLCA/PLCS."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("merge_results", ROOT / "OdysseyBenchmark/tools/merge_results.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

SCENES = ("odyssey_scene001", "odyssey_scene002", "odyssey_scene003")


def make_scenes_csv(tmp, names=SCENES):
    path = tmp / "scenes.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "token"])
        for i, name in enumerate(names, 1):
            w.writerow([name, f"tok{i}"])
    return path


def make_run(campaign, scene, react, *, ds="40.5", scoring_error="", exit_code=0, flag=True, csv_columns=None,
             record_status=None, reason="", report=None, plc_rule="plc", attempt=1, plc=None):
    run = campaign / "runs" / f"{scene}_{react}" / f"attempt_{attempt:02d}"
    (run / "odyssey_output").mkdir(parents=True)
    (run / "launch.json").write_text(json.dumps(dict(scene=scene, react=react, profile=dict(planner_id="my_model"))))
    if exit_code is not None:
        (run / "exit_code.txt").write_text(f"{exit_code}\n")
    if flag:
        (run / "odyssey_output/simulation_completed.flag").write_text("")
    if csv_columns is not None:
        row = dict(scene=scene, react=react, steps="1828", term_reason="destination_arrival",
                   RouteDS=ds, RC="1.0", P_SD="1", P_col="0.6", P_off="0.5", P_TL="1.0", P_PLC="0.343", PLCA="0.5", PLCS="0.25",
                   Eff="50.0", Comf="1", collision_count="1", tl_violation_count="0", plc_stops="4", plc_reached="2",
                   plc_pass="1", plc_late="0", plc_fail="1", tl_set="tlc_timetable", plc_rule=plc_rule,
                   scoring_error=scoring_error)
        row.update(plc or {})
        row.update({c: "x" for c in csv_columns if c not in row})
        columns = list(row)
        with (run / f"odyssey_output/routeds_{react.upper()}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerow(row)
    if report is not None:
        (run / "odyssey_output/runner_report.json").write_text(json.dumps([dict(scenario_name="t", succeeded=False, duration_s=3.0, error_message=report)]))
    (run / "simulation.log").write_text("Step 1\nRuntimeError: boom\n")
    if record_status is not None:
        (campaign / "jobs").mkdir(exist_ok=True)
        rec = dict(schema="odyssey_eval_job/1", job_id=f"{scene}_{react}", scene=scene, react=react, status=record_status,
                   final=True, attempt=attempt, run_dir=str(run.relative_to(campaign)),
                   attempts=[dict(exit_code=exit_code, flag=flag, duration_s=12.0, gpu="1")],
                   failure=None if record_status == "complete" else dict(**{"class": "model" if record_status == "model_fail" else "infra"}, kind="k", reason=reason))
        (campaign / "jobs" / f"{scene}_{react}.json").write_text(json.dumps(rec))
    return run


def campaign_dir(tmp):
    c = tmp / "campaign"
    (c / "runs").mkdir(parents=True)
    (c / "manifest.json").write_text(json.dumps(dict(model=dict(name="my_model", checkpoint=dict(bytes=5, head_1mb_sha256="deadbeef")),
                                                     benchmark=dict(reacts=["nr", "r"]), environment=dict(host="h", code=dict(commit="c0ffee", dirty=False)))))
    return c


def run_tool(tmp, campaign, *extra):
    return mr.main(["-f", str(campaign), "--scenes", str(tmp / "scenes.csv"), *extra])


def summary_rows(campaign):
    return {r["react"]: r for r in csv.DictReader((campaign / "summary.csv").open())}


def test_mixed_campaign_blocks_final_then_passes(tmp_path, capsys):
    make_scenes_csv(tmp_path)
    c = campaign_dir(tmp_path)
    make_run(c, "odyssey_scene001", "nr", csv_columns=[], record_status="complete")
    make_run(c, "odyssey_scene002", "nr", exit_code=1, flag=False, record_status="model_fail", reason="KeyError: foo")
    make_run(c, "odyssey_scene003", "nr", exit_code=1, flag=False, record_status="infra_fail", reason="CUDA")
    make_run(c, "odyssey_scene001", "r", csv_columns=[], scoring_error="KeyError: rc", record_status="complete")   # record/csv disagree
    make_run(c, "odyssey_scene002", "r", csv_columns=[], ds="", record_status="complete")                         # unscored
    # odyssey_scene003 r missing entirely
    assert run_tool(tmp_path, c) == 1
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    s = summary_rows(c)
    assert s["nr"]["final"] == "False" and s["nr"]["RouteDS"] == ""          # blocked: metrics hidden
    assert (s["nr"]["n_complete"], s["nr"]["n_model_fail"], s["nr"]["n_infra_fail"]) == ("1", "1", "1")
    assert (s["r"]["n_scoring_error"], s["r"]["n_unscored"], s["r"]["n_missing"]) == ("1", "1", "1")
    payload = json.loads((c / "summary.json").read_text())
    actions = {(j["scene"], j["react"]): j["action"] for j in payload["jobs_to_rerun"]}
    assert actions == {("odyssey_scene003", "nr"): "rerun", ("odyssey_scene001", "r"): "rerun", ("odyssey_scene002", "r"): "investigate", ("odyssey_scene003", "r"): "run"}
    runs = {(r["scene"], r["react"]): r for r in csv.DictReader((c / "runs.csv").open())}
    assert runs[("odyssey_scene002", "nr")]["RouteDS_final"] == "0.0" and runs[("odyssey_scene002", "nr")]["RC"] == ""
    assert runs[("odyssey_scene001", "nr")]["RouteDS_final"] == "40.5" and runs[("odyssey_scene001", "nr")]["record_source"] == "record.json"
    assert runs[("odyssey_scene001", "nr")]["scene"] == "odyssey_scene001" and runs[("odyssey_scene001", "nr")]["PLCS"] == "0.25"

    assert run_tool(tmp_path, c, "--allow-incomplete") == 0
    s = summary_rows(c)
    assert s["nr"]["RouteDS"] == str(round((40.5 + 0 + 0) / 3, 4)) and s["nr"]["RouteDS_scored"] == "40.5"

    # fix the blockers: scene003 nr complete on attempt 2, r jobs complete
    make_run(c, "odyssey_scene003", "nr", csv_columns=[], record_status="complete", attempt=2)
    for scene in SCENES:
        import shutil
        shutil.rmtree(c / "runs" / f"{scene}_r", ignore_errors=True)
        (c / "jobs" / f"{scene}_r.json").unlink(missing_ok=True)
        make_run(c, scene, "r", csv_columns=[], record_status="complete")
    assert run_tool(tmp_path, c) == 0
    s = summary_rows(c)
    assert s["nr"]["final"] == "True" and s["nr"]["RouteDS"] == "27.0" and s["r"]["RouteDS"] == "40.5"
    assert (s["nr"]["SDC"], s["nr"]["PLCA"], s["nr"]["PLCS"], s["nr"]["Eff."], s["nr"]["P_PLC"]) == ("100.0", "50.0", "25.0", "50.0", "0.343")
    assert s["all"]["RouteDS"] == "33.75"
    merged = json.loads((c / "merged.json").read_text())
    assert merged["driving score"] == {"nr": 27.0, "r": 40.5} and merged["eval num"] == {"nr": 3, "r": 3}
    assert merged["_checkpoint"]["records"][0]["route_id"] == "odyssey_scene001_nr"


def test_fallback_without_records_and_unclassified_mapping(tmp_path):
    make_scenes_csv(tmp_path, ("odyssey_scene001", "odyssey_scene002"))
    c = campaign_dir(tmp_path)
    make_run(c, "odyssey_scene001", "nr", csv_columns=[])
    make_run(c, "odyssey_scene002", "nr", exit_code=1, flag=False, report="Traceback\nKeyError: 'foo'")
    assert run_tool(tmp_path, c, "--react", "nr") == 1
    runs = {r["scene"]: r for r in csv.DictReader((c / "runs.csv").open())}
    assert runs["odyssey_scene001"]["status"] == "complete" and runs["odyssey_scene001"]["record_source"] == "fallback"
    assert runs["odyssey_scene002"]["status"] == "failed_unclassified" and runs["odyssey_scene002"]["reason"] == "KeyError: 'foo'"
    assert run_tool(tmp_path, c, "--react", "nr", "--classify-unclassified", "model_fail") == 0
    s = summary_rows(c)
    assert s["nr"]["final"] == "True" and s["nr"]["RouteDS"] == "20.25"


def test_runs_csv_is_the_union_of_per_run_columns(tmp_path):
    make_scenes_csv(tmp_path, ("odyssey_scene001", "odyssey_scene002"))
    c = campaign_dir(tmp_path)
    make_run(c, "odyssey_scene001", "nr", csv_columns=["only_in_first"], record_status="complete")
    make_run(c, "odyssey_scene002", "nr", csv_columns=["only_in_second", "another"], record_status="complete")
    run_tool(tmp_path, c, "--react", "nr")
    header = next(csv.reader((c / "runs.csv").open()))
    assert {"only_in_first", "only_in_second", "another"} <= set(header)
    rows = {r["scene"]: r for r in csv.DictReader((c / "runs.csv").open())}
    assert rows["odyssey_scene001"]["only_in_second"] == "" and rows["odyssey_scene002"]["another"] == "x"


def test_mixed_rules_block_final(tmp_path):
    make_scenes_csv(tmp_path, ("odyssey_scene001", "odyssey_scene002"))
    c = campaign_dir(tmp_path)
    make_run(c, "odyssey_scene001", "nr", csv_columns=[], record_status="complete", plc_rule="plc")
    make_run(c, "odyssey_scene002", "nr", csv_columns=[], record_status="complete", plc_rule="other")
    assert run_tool(tmp_path, c, "--react", "nr") == 1
    s = summary_rows(c)
    assert s["nr"]["plc_rule"] == "" and s["nr"]["final"] == "False"


def test_plca_plcs_pool_the_stop_lines_of_the_sheet(tmp_path):
    names = ("odyssey_scene001", "odyssey_scene002", "odyssey_scene003", "odyssey_scene004")
    make_scenes_csv(tmp_path, names)
    c = campaign_dir(tmp_path)
    # (stops, reached, pass, late, fail, per-run PLCA, per-run PLCS)
    counts = [("1", "1", "1", "0", "0", "1.0", "1.0"),          # one clean pass
              ("4", "3", "2", "1", "1", "0.5", "0.375"),        # credit 2 - 0.5 = 1.5
              ("2", "0", "0", "0", "0", "", "0.0"),             # never reached its stop lines
              ("0", "0", "0", "0", "0", "", "")]                # no evaluation stop line: not counted
    for name, (stops, reached, passed, late, fail, plca, plcs) in zip(names, counts):
        make_run(c, name, "nr", csv_columns=[], record_status="complete",
                 plc=dict(plc_stops=stops, plc_reached=reached, plc_pass=passed, plc_late=late, plc_fail=fail, PLCA=plca, PLCS=plcs))
    assert run_tool(tmp_path, c, "--react", "nr") == 0
    s = summary_rows(c)["nr"]
    # credit 2.5 over 4 stop lines reached and 7 on the routes (in percent); the per-run means would be 75 and 45.83
    assert (s["PLCA"], s["PLCS"], s["n_PLC"]) == ("62.5", "35.71", "3")


def test_two_scene_rows_in_one_result_file_is_an_error(tmp_path):
    run = tmp_path / "run"
    (run / "odyssey_output").mkdir(parents=True)
    (run / "odyssey_output/routeds_NR.csv").write_text("scene,RouteDS\na,1\nb,2\n")
    with pytest.raises(ValueError, match="one scene per run"):
        mr.read_scene_row(run)
    (run / "odyssey_output/routeds_NR.csv").write_text("scene,RouteDS\na,1\n")
    row, path = mr.read_scene_row(run)
    assert row["scene"] == "a" and path.endswith("routeds_NR.csv")
