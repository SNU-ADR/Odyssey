#!/usr/bin/env python3
"""Merge a campaign's runs into the benchmark table.

    python OdysseyBenchmark/tools/merge_results.py -f experiments/simulation/eval_my_model [--allow-incomplete]

Reads the records the batch runner wrote (``jobs/*.json``, ``manifest.json``) and, when a run has
no record, the run directory itself (its ``routeds_{NR|R}.csv``). Writes next to them:

    runs.csv       one row per expected scene x react, with the per-run result columns
    summary.csv    one row per react (and an optional 'all' row): the benchmark columns
    summary.json   the same with raw keys, plus the jobs that still need attention
    merged.json    Bench2Drive-like: "driving score" per react and one record per route

Scoring rules (fixed by the benchmark):
  * a job the MODEL failed counts with RouteDS 0, the other metrics blank;
  * the denominator is always the full scene set;
  * an infrastructure failure, timeout, missing run, scoring error or unscored episode BLOCKS the
    final score: the table is still printed, the exit code is 1 and the job is listed with what
    to do (rerun / install / investigate). ``--allow-incomplete`` prints a provisional score instead.
Aggregation: means over scenes, except Eff. (median) and PLCA/PLCS (pooled over the sheet's stop
lines). SDC, PLCA, PLCS and Comf. are reported in percent, as in the paper's table. Standard library only.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

#: (result-file column, sheet label, aggregation). The first six are the paper's table; the rest
#: are RouteDS's components (Appendix C.3 / Table 11). 'pooled': see pooled_plc.
SET_COLS = [("RouteDS", "RouteDS", "mean"), ("P_SD", "SDC", "mean"), ("PLCA", "PLCA", "pooled"),
            ("PLCS", "PLCS", "pooled"), ("Eff", "Eff.", "median"), ("Comf", "Comf.", "mean"),
            ("RC", "RC", "mean"), ("P_SD", "P_SD", "mean"), ("P_col", "P_col", "mean"),
            ("P_off", "P_off", "mean"), ("P_TL", "P_TL", "mean"), ("P_PLC", "P_PLC", "mean")]
LABELS = [label for _, label, _ in SET_COLS]
#: Sheet columns given in percent, as in the paper's table (the RouteDS components stay fractions).
PERCENT_LABELS = ("SDC", "PLCA", "PLCS", "Comf.")
OK_STATUSES = ("complete", "model_fail")
ACTION = dict(timeout="rerun", infra_fail="rerun", input_missing="install the scene", scoring_error="rerun",
              unscored="investigate", failed_unclassified="classify (see --classify-unclassified)", missing="run",
              aborted="rerun", running="wait or rerun")
RESULT_COLUMNS = ["scene", "react", "steps", "term_reason", "RouteDS", "RC", "P_SD", "P_col", "P_off",
                  "P_TL", "P_PLC", "PLCA", "PLCS", "Eff", "Comf", "collision_count", "tl_violation_count",
                  "plc_stops", "plc_reached", "plc_pass", "plc_late", "plc_fail", "tl_set", "plc_rule",
                  "scoring_error"]
LEAD_COLUMNS = ["sheet", "agent", "scene", "react", "status", "reason", "action", "RouteDS_final"] + \
    [c for c in RESULT_COLUMNS if c not in ("scene", "react")] + \
    ["exit_code", "flag", "duration_s", "attempts", "gpu", "run_dir", "csv_path", "record_source"]
DIRNAME = re.compile(r"^(?:(?P<agent>.+?)_)?(?P<scene>odyssey_scene\d{3})_(?P<react>nr|r)$")


def fnum(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_scene_row(run_dir):
    """The run's result row (odyssey_output/routeds_{NR|R}.csv; older runs: openscene_format/); one scene per run."""
    out = Path(run_dir, "odyssey_output")
    for path in sorted(out.glob("routeds_*.csv")) or sorted(out.glob("openscene_format/routeds_*.csv")):
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) > 1:
            raise ValueError(f"{path}: {len(rows)} scene rows; one scene per run is required")
        if rows:
            return rows[0], str(path.relative_to(run_dir))
    return None, None


def read_runner_report(run_dir):
    reports = sorted(Path(run_dir, "odyssey_output").glob("runner_report*.json"))
    data = read_json(reports[-1]) if reports else None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def scene_of(text):
    """The published scene name in ``text`` (odyssey_sceneNNN), or None."""
    m = re.search(r"odyssey_scene\d{3}", text or "")
    return m.group(0) if m else None


# ----------------------------------------------------------------------------- expected set
def expected_jobs(scenes_csv, manifest, reacts):
    """-> {(scene, react): {scene, token}}: the campaign's planned jobs, or a scenes.csv x reacts."""
    if manifest and manifest.get("benchmark", {}).get("job_table") and not scenes_csv:
        return {(t["scene"], t["react"]): dict(scene=t["scene"], token=t.get("token"))
                for t in manifest["benchmark"]["job_table"] if t["react"] in reacts}
    if not (scenes_csv and Path(scenes_csv).is_file()):
        raise SystemExit("the expected scene set is unknown: pass --scenes <scenes.csv> or set ODYSSEY_SCENES_ROOT")
    sys.path.insert(0, str(ROOT))
    from odyssey_runtime import scenes
    table = scenes.release_table(Path(scenes_csv).parent)
    return {(t["scene"], react): dict(scene=t["scene"], token=t.get("token")) for t in table for react in reacts}


# ----------------------------------------------------------------------------- per-run rows
def discover_runs(campaign, runs_dir):
    """Run directories: the runner's runs/<job>/attempt_NN (latest attempt), or a flat layout."""
    found = {}
    base = runs_dir or (campaign / "runs")
    if base.is_dir():
        for job_dir in sorted(base.iterdir()):
            if not job_dir.is_dir():
                continue
            attempts = sorted(p for p in job_dir.glob("attempt_*") if p.is_dir())
            found[job_dir.name] = attempts[-1] if attempts else job_dir
    if not found:
        base = runs_dir or campaign
        for p in sorted(base.iterdir()):
            if p.is_dir() and (p / "launch.json").exists():
                found[p.name] = p
    return found


def identify(run_dir, record, launch):
    launch = launch or {}
    scene = scene_of((record or {}).get("scene")) or scene_of(launch.get("scene_name")) or scene_of(launch.get("scene"))
    react = (record or {}).get("react") or launch.get("react")
    agent = (launch.get("agent") or {}).get("name") or (launch.get("profile") or {}).get("planner_id")
    m = DIRNAME.match(run_dir.name) or DIRNAME.match(run_dir.parent.name)
    if m:
        scene = scene or scene_of(m.group("scene"))
        react = react or m.group("react")
        agent = agent or m.group("agent")
    return scene, react, agent


def derive_status(run_dir, record, row, classify_unclassified):
    """-> (status, reason). The record is trusted except where the result file contradicts 'complete'."""
    exit_code = None
    try:
        exit_code = int((run_dir / "exit_code.txt").read_text().strip())
    except (OSError, ValueError):
        pass
    flag = (run_dir / "odyssey_output/simulation_completed.flag").is_file()
    if record:
        status = record.get("status")
        reason = (record.get("failure") or {}).get("reason") or ""
        if status == "complete":
            if row is None or row.get("scoring_error"):
                return ("scoring_error", f"record/result disagree: {row.get('scoring_error') if row else 'result file missing'}")
            if fnum(row.get("RouteDS")) is None:
                return ("unscored", "record/result disagree: no RouteDS")
        if status == "infra_fail" and (record.get("failure") or {}).get("kind") == "scoring_error":
            return ("scoring_error", reason)
        return status, reason
    if exit_code is None:
        return ("aborted", "no exit_code.txt") if (run_dir / "launch.json").exists() else ("missing", "no run")
    if exit_code == 0 and flag:
        if row is None:
            return "infra_fail", "result file missing"
        if row.get("scoring_error"):
            return "scoring_error", row["scoring_error"]
        if fnum(row.get("RouteDS")) is None:
            return "unscored", f"no dense scoring inputs (term_reason={row.get('term_reason')})"
        return "complete", ""
    if exit_code == 124:
        return "timeout", "wall-clock timeout"
    report = read_runner_report(run_dir)
    reason = (report or {}).get("error_message") or ""
    if not reason:
        try:
            lines = [l for l in (run_dir / "simulation.log").read_text(errors="replace").splitlines() if l.strip()]
            reason = lines[-1] if lines else ""
        except OSError:
            pass
    reason = (re.findall(r"^\w+(?:Error|Exception)\b.*$", reason, re.M) or [reason.strip()[-300:]])[-1]
    if classify_unclassified == "model_fail":
        return "model_fail", reason
    return "failed_unclassified", reason


def job_row(key, expected, run_dir, record, agent_default, classify_unclassified):
    scene, react = key
    base = dict(scene=scene, react=react, agent=agent_default, record_source="none",
                status="missing", reason="", action=ACTION["missing"], RouteDS_final=None)
    if run_dir is None:
        return base
    launch = read_json(run_dir / "launch.json") or {}
    row, csv_path = read_scene_row(run_dir)
    status, reason = derive_status(run_dir, record, row, classify_unclassified)
    _, _, agent = identify(run_dir, record, launch)
    attempts = record.get("attempts", []) if record else []
    last = attempts[-1] if attempts else {}
    out = dict(base, agent=agent or agent_default, status=status, reason=reason, action=ACTION.get(status, ""),
               record_source="record.json" if record else "fallback", run_dir=str(run_dir), csv_path=csv_path,
               exit_code=last.get("exit_code"), flag=last.get("flag"), duration_s=last.get("duration_s"),
               attempts=len(attempts) or None, gpu=last.get("gpu"))
    if out["exit_code"] is None:
        try:
            out["exit_code"] = int((run_dir / "exit_code.txt").read_text().strip())
        except (OSError, ValueError):
            pass
        out["flag"] = (run_dir / "odyssey_output/simulation_completed.flag").is_file()
    if out["duration_s"] is None:
        report = read_runner_report(run_dir)
        out["duration_s"] = (report or {}).get("duration_s")
    if row is not None:
        for k, v in row.items():
            if k not in ("scene", "react"):
                out[k] = v
        if row.get("scene"):
            out["scene"] = row["scene"]
    out["RouteDS_final"] = (fnum(row.get("RouteDS")) if status == "complete" and row else
                            (0.0 if status == "model_fail" else None))
    return out


# ----------------------------------------------------------------------------- aggregation
def pooled_plc(rows):
    """(PLCA, PLCS, runs counted) over a sheet: the total credit (1 per clean pass, 0.5 per late one)
    over every stop line reached, and over every evaluation stop line, of all its runs. Not a mean of
    the per-run ratios, which would weigh a route with one stop line like a route with ten."""
    credit = reached = stops = runs = 0
    for r in rows:
        n_stops, n_reached, n_pass, n_late = (fnum(r.get(k)) for k in ("plc_stops", "plc_reached", "plc_pass", "plc_late"))
        if not n_stops or None in (n_reached, n_pass, n_late):
            continue
        credit += n_pass - 0.5 * n_late
        reached += n_reached
        stops += n_stops
        runs += 1
    return (round(credit / reached, 4) if reached else None), (round(credit / stops, 4) if stops else None), runs


def aggregate(rows, react):
    expected = [r for r in rows if r["react"] == react]
    complete = [r for r in expected if r["status"] == "complete"]
    counts = {s: sum(1 for r in expected if r["status"] == s) for s in
              ("model_fail", "infra_fail", "timeout", "input_missing", "scoring_error", "unscored",
               "failed_unclassified", "missing", "aborted", "running")}
    out = dict(react=react, n_expected=len(expected), n_complete=len(complete), **{f"n_{k}": v for k, v in counts.items()})
    blockers = [r for r in expected if r["status"] not in OK_STATUSES]
    rules = sorted({(r.get("tl_set"), r.get("plc_rule")) for r in complete}, key=str)
    out["tl_set"], out["plc_rule"] = (rules[0] if len(rules) == 1 else (None, None))
    out["final"] = not blockers and len(rules) <= 1 and bool(expected)
    final_scores = [r["RouteDS_final"] for r in expected if r["RouteDS_final"] is not None]
    out["RouteDS"] = round(sum(final_scores) / len(expected), 4) if expected else None
    scored = [fnum(r.get("RouteDS")) for r in complete]
    scored = [x for x in scored if x is not None]
    out["RouteDS_scored"] = round(sum(scored) / len(scored), 4) if scored else None
    out["n_scored"] = len(scored)
    for key, label, how in SET_COLS[1:]:
        values = [fnum(r.get(key)) for r in complete]
        values = [v for v in values if v is not None]
        if how == "median":
            out[label] = round(statistics.median(values), 4) if values else None
            out["n_Eff"] = len(values)
        elif how == "mean":
            out[label] = round(sum(values) / len(values), 4) if values else None
    out["PLCA"], out["PLCS"], out["n_PLC"] = pooled_plc(complete)
    for label in PERCENT_LABELS:
        if out[label] is not None:
            out[label] = round(out[label] * 100, 2)
    reasons = {}
    for r in expected:
        if r.get("term_reason"):
            reasons[r["term_reason"]] = reasons.get(r["term_reason"], 0) + 1
    out["term_reasons"] = json.dumps(reasons, sort_keys=True)
    out["blockers"] = [dict(scene=r["scene"], react=react, status=r["status"],
                            reason=r.get("reason"), action=r.get("action"), run_dir=r.get("run_dir")) for r in blockers]
    out["model_failures"] = [dict(scene=r["scene"], react=react, reason=r.get("reason"), run_dir=r.get("run_dir"))
                             for r in expected if r["status"] == "model_fail"]
    return out


def cross_check_routeds(row):
    """RouteDS recomputed from its components (paper eq. 2); None when a term is missing."""
    terms = [fnum(row.get(k)) for k in ("RC", "P_SD", "P_col", "P_off", "P_TL", "P_PLC")]
    if any(t is None for t in terms):
        return None
    product = 1.0
    for t in terms:
        product *= t
    return max(product, 0.0) * 100


# ----------------------------------------------------------------------------- outputs
def write_runs_csv(path, rows):
    extra = []
    for r in rows:
        for k in r:
            if k not in LEAD_COLUMNS and k not in extra:
                extra.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEAD_COLUMNS + extra, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def write_summary(campaign, summaries, meta, allow_incomplete):
    columns = ["sheet", "agent", "react", "final", "benchmark_set", "scene_set", "n_expected", "n_complete", "n_model_fail",
               "n_infra_fail", "n_timeout", "n_input_missing", "n_scoring_error", "n_unscored", "n_failed_unclassified",
               "n_missing", "RouteDS", "RouteDS_scored", "n_scored", "SDC", "PLCA", "PLCS", "n_PLC", "Eff.", "n_Eff",
               "Comf.", "RC", "P_SD", "P_col", "P_off", "P_TL", "P_PLC", "term_reasons",
               "tl_set", "plc_rule", "code_commit", "code_dirty", "agent_name", "agent_config_sha256", "checkpoint_id", "host"]
    rows = []
    for s in summaries:
        shown = dict(s, **meta)
        if not (s["final"] or allow_incomplete):
            for label in LABELS + ["RouteDS_scored"]:
                shown[label] = None
        rows.append({k: shown.get(k) for k in columns})
    with (campaign / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    payload = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), meta=meta,
                   columns={label: key for key, label, _ in SET_COLS},
                   sheets=[dict(s, metrics_shown=bool(s["final"] or allow_incomplete)) for s in summaries],
                   jobs_to_rerun=[b for s in summaries for b in s["blockers"]])
    (campaign / "summary.json").write_text(json.dumps(payload, indent=1))
    return rows


def write_merged(campaign, rows, summaries, meta, allow_incomplete):
    summaries = [s for s in summaries if s["react"] != "all"]      # the combined row is not a benchmark number
    records = []
    metrics = RESULT_COLUMNS[RESULT_COLUMNS.index("RouteDS"):RESULT_COLUMNS.index("Comf") + 1]
    for r in sorted(rows, key=lambda r: (r["react"], r["scene"])):
        scores = {k: fnum(r.get(k)) for k in metrics if r.get(k) not in (None, "")}
        if r["status"] == "complete":
            scores["score_composed"] = fnum(r.get("RouteDS"))
        elif r["status"] == "model_fail":
            scores["score_composed"] = 0.0
        records.append(dict(route_id=f"{r['scene']}_{r['react']}", scene=r["scene"],
                            react=r["react"], status=r["status"], reason=r.get("reason") or None,
                            term_reason=r.get("term_reason"), scores=scores, duration_s=fnum(r.get("duration_s")),
                            attempts=r.get("attempts"), gpu=r.get("gpu"), run_dir=r.get("run_dir")))
    merged = {"campaign": str(campaign), "agent": meta.get("agent_name"), "code_commit": meta.get("code_commit"),
              "complete": {s["react"]: s["final"] for s in summaries},
              "driving score": {s["react"]: (s["RouteDS"] if s["final"] or allow_incomplete else None) for s in summaries},
              "eval num": {s["react"]: s["n_complete"] + s["n_model_fail"] for s in summaries},
              "expected num": {s["react"]: s["n_expected"] for s in summaries},
              "_checkpoint": {"records": records}}
    (campaign / "merged.json").write_text(json.dumps(merged, indent=2))


def render_console(summaries, meta, allow_incomplete, quiet):
    lines = [f"agent {meta.get('agent_name')}  commit {meta.get('code_commit')}{' (dirty)' if meta.get('code_dirty') else ''}  "
             f"host {meta.get('host')}"]
    lines.append(f"{'sheet':22s} {'n':>7s} {'RouteDS':>8s} {'SDC':>6s} {'PLCA':>6s} {'PLCS':>6s} {'Eff.':>7s} {'Comf.':>6s}  status")
    for s in summaries:
        n = f"{s['n_complete'] + s['n_model_fail']}/{s['n_expected']}"
        show = s["final"] or allow_incomplete

        def cell(label, width=6, digits=2):
            v = s.get(label)
            return f"{v:>{width}.{digits}f}" if show and isinstance(v, (int, float)) else f"{'-':>{width}s}"
        status = "FINAL" if s["final"] else "INCOMPLETE" + ("" if not s["blockers"] else f" ({len(s['blockers'])} to fix)")
        if s.get("optional"):
            status += " (mean of nr and r; optional, not a benchmark sheet)"
        if s["final"] and not s.get("benchmark_set", True):
            status += " (mini set, not the benchmark)" if s.get("scene_set") == "mini" else " (partial scene set, not the benchmark)"
        # An incomplete sheet shows RouteDS over the jobs finished so far (marked *): the fixed
        # denominator would count every job still to run as 0.
        provisional = show and not s["final"] and s.get("react") != "all"
        ds = (f"{s['RouteDS_scored']:>7.2f}*" if provisional and isinstance(s.get("RouteDS_scored"), (int, float))
              else cell("RouteDS", 8))
        lines.append(f"{s['sheet']:22s} {n:>7s} {ds} {cell('SDC', 6, 1)} {cell('PLCA', 6, 1)} {cell('PLCS', 6, 1)} "
                     f"{cell('Eff.', 7, 1)} {cell('Comf.', 6, 1)}  {status}")
    if allow_incomplete and any(not s["final"] for s in summaries):
        lines.append("  * provisional: mean over the finished jobs only (summary.csv: RouteDS_scored); "
                     "RouteDS with the fixed denominator is in summary.csv")
    if not quiet:
        for s in summaries:
            if s.get("react") != "all" and s.get("RC") is not None and (s["final"] or allow_incomplete):
                lines.append(f"  {s['sheet']} components: RC {s['RC']}  P_SD {s['P_SD']}  P_col {s['P_col']}  "
                             f"P_off {s['P_off']}  P_TL {s['P_TL']}  P_PLC {s['P_PLC']}")
            if s["model_failures"]:
                lines.append(f"\n{s['sheet']}: {len(s['model_failures'])} model failure(s) scored 0:")
                for f in s["model_failures"][:20]:
                    lines.append(f"  {f['scene']}  {f['reason']}")
            if s["blockers"]:
                lines.append(f"\n{s['sheet']}: {len(s['blockers'])} job(s) block the final score:")
                for b in s["blockers"][:20]:
                    lines.append(f"  {b['scene']:18s} {b['status']:20s} {b['action']:12s} {b.get('reason') or ''}")
    return "\n".join(lines)


def meta_from(campaign, manifest, agent_default):
    code = manifest.get("environment", {}).get("code", {}) if manifest else {}
    if not code.get("commit"):
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip())
            code = dict(commit=commit or "unknown", dirty=dirty)
        except (OSError, subprocess.SubprocessError):
            code = dict(commit="unknown", dirty=None)
    model = manifest.get("model", {}) if manifest else {}
    ckpt = model.get("checkpoint", {})
    return dict(code_commit=code.get("commit"), code_dirty=code.get("dirty"),
                agent_name=model.get("name") or agent_default or "unknown",
                agent_config_sha256=(model.get("agent_config") or {}).get("sha256"),
                checkpoint_id=(f"{ckpt.get('bytes')}:{str(ckpt.get('head_1mb_sha256'))[:12]}" if ckpt.get("head_1mb_sha256") else ckpt.get("path")),
                host=(manifest.get("environment", {}).get("host") if manifest else None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", required=True, help="campaign directory (manifest.json, jobs/, runs/) or a flat folder of runs")
    parser.add_argument("--runs-dir", help="run directories, if not <campaign>/runs")
    parser.add_argument("--scenes", help="scenes.csv of the expected set (default: the campaign's planned jobs, else $ODYSSEY_SCENES_ROOT/scenes.csv)")
    parser.add_argument("--react", help="nr,r (default: the manifest's, else both)")
    parser.add_argument("--agent", help="agent/model name for the sheet (default: the manifest's)")
    parser.add_argument("--allow-incomplete", action="store_true", help="print provisional metrics and exit 0 even when jobs block the score")
    parser.add_argument("--classify-unclassified", choices=["infra_fail", "model_fail"], default="infra_fail",
                        help="how a failed run without a runner record is counted")
    parser.add_argument("--out", help="write outputs here instead of the campaign directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    campaign = Path(args.folder)
    manifest = read_json(campaign / "manifest.json")
    reacts = [r for r in re.split(r"[,\s]+", args.react or (manifest or {}).get("benchmark", {}).get("reacts") and
                                  ",".join(manifest["benchmark"]["reacts"]) or "nr,r") if r]
    # Expected set: the campaign's planned jobs (manifest); for a flat folder of runs, a scenes.csv.
    scenes_csv = args.scenes
    if scenes_csv is None and not (manifest or {}).get("benchmark", {}).get("job_table"):
        scenes_csv = (Path(os.environ["ODYSSEY_SCENES_ROOT"]) / "scenes.csv" if os.environ.get("ODYSSEY_SCENES_ROOT") else None)
        if scenes_csv is None and (campaign / "inputs/scenes.csv").is_file():
            scenes_csv = campaign / "inputs/scenes.csv"
    expected = expected_jobs(scenes_csv, manifest, reacts)
    agent_default = args.agent or ((manifest or {}).get("model") or {}).get("name")

    records = {}
    for path in (campaign / "jobs").glob("*.json") if (campaign / "jobs").is_dir() else []:
        rec = read_json(path)
        if rec and scene_of(rec.get("scene")) and rec.get("react"):
            records[(scene_of(rec["scene"]), rec["react"])] = rec
    run_dirs = discover_runs(campaign, Path(args.runs_dir) if args.runs_dir else None)
    by_key = {}
    for name, run_dir in run_dirs.items():
        launch = read_json(run_dir / "launch.json")
        scene, react, agent = identify(run_dir, None, launch)
        if args.agent and agent and agent != args.agent and name.startswith(agent + "_"):
            continue
        if scene and react:
            by_key.setdefault((scene, react), run_dir)
    for key, rec in records.items():
        if rec.get("run_dir"):
            by_key[key] = campaign / rec["run_dir"]

    rows = [job_row(key, exp, by_key.get(key), records.get(key), agent_default, args.classify_unclassified)
            for key, exp in sorted(expected.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    for r in rows:
        r["sheet"] = f"{r['agent']}_{r['react']}"
        if r["status"] == "complete":
            recomputed = cross_check_routeds(r)
            ds = fnum(r.get("RouteDS"))
            # The result file carries each term rounded to 5 decimals, so the product can differ in the 4th.
            if recomputed is not None and ds is not None and abs(recomputed - ds) > 5e-3:
                print(f"warning: {r['scene']} {r['react']}: RouteDS {ds} but its components give {recomputed:.5f}", file=sys.stderr)

    summaries = []
    scene_set = (manifest or {}).get("benchmark", {}).get("scene_set") or ("custom" if manifest else "unknown")
    for react in reacts:
        s = aggregate(rows, react)
        s["sheet"] = f"{agent_default or 'model'}_{react}"
        s["agent"] = agent_default
        s["benchmark_set"] = s["n_expected"] == 100       # the published benchmark is the full 100-scene set
        s["scene_set"] = scene_set
        if not s["benchmark_set"] and s["n_expected"]:
            what = f"the {s['n_expected']}-scene mini set" if scene_set == "mini" else f"{s['n_expected']} scene(s)"
            print(f"note: {react}: this campaign covers {what}; the benchmark number needs all 100", file=sys.stderr)
        summaries.append(s)
    if len(summaries) == 2 and all(s["final"] for s in summaries):
        both = dict(summaries[0])
        both.update(react="all", sheet=f"{agent_default}_all", final=True, optional=True,
                    RouteDS=round((summaries[0]["RouteDS"] + summaries[1]["RouteDS"]) / 2, 4), blockers=[], model_failures=[])
        for label in LABELS[1:] + ["RouteDS_scored"]:
            both[label] = None
        both["n_expected"] = sum(s["n_expected"] for s in summaries)
        both["n_complete"] = sum(s["n_complete"] for s in summaries)
        summaries.append(both)

    meta = meta_from(campaign, manifest, agent_default)
    out = Path(args.out) if args.out else campaign
    out.mkdir(parents=True, exist_ok=True)
    write_runs_csv(out / "runs.csv", rows)
    write_summary(out, summaries, meta, args.allow_incomplete)
    write_merged(out, rows, summaries, meta, args.allow_incomplete)
    print(render_console(summaries, meta, args.allow_incomplete, args.quiet))
    final = all(s["final"] for s in summaries if s["react"] != "all")
    todo = sum(len(s["blockers"]) for s in summaries)
    print(f"\n{'FINAL' if final else f'INCOMPLETE: {todo} job(s) to fix'}  ->  {out / 'summary.csv'}, {out / 'runs.csv'}, {out / 'merged.json'}")
    return 0 if final or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
