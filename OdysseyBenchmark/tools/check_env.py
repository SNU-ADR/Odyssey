#!/usr/bin/env python
"""Check the host before evaluating: interpreters (with CUDA), nvdiffrast, nvcc, OdysseyZoo, scenes, weights, map slots, GPUs, Fixer weights.

    python OdysseyBenchmark/tools/check_env.py [--agent OdysseyBenchmark/agents/ltf_sdroute.yaml] [--gpus 0,1,2,3] [--md5] [--json]

Runs under any Python 3 with PyYAML; it executes the configured interpreters to probe them.
Exit code 1 when any check fails.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]          # the repository root
sys.path.insert(0, str(ROOT / "OdysseyBenchmark"))

from odyssey_runtime import scenes                                   # noqa: E402
from odyssey_runtime.agent_config import load as load_agent          # noqa: E402
from odyssey_runtime.launch import (EnvironmentRefused,  # noqa: E402
                                    check_benchmark_environment, gpu_visibility_error)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append(dict(check=name, ok=bool(ok), detail=str(detail)))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r["ok"]]


def probe_python(path):
    """(ok, detail): the interpreter runs, imports torch, and torch sees a CUDA device."""
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False, "not an executable file"
    try:
        out = subprocess.run([path, "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if out.returncode:
        return False, (out.stderr.strip().splitlines() or ["import failed"])[-1]
    version, torch_version, cuda = (out.stdout.split() + ["", "", ""])[:3]
    if cuda != "True":
        hidden = " (CUDA_VISIBLE_DEVICES is empty)" if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else ""
        return False, f"python {version}, torch {torch_version}: torch sees no CUDA device{hidden}; run this on the GPU host"
    return True, f"python {version}, torch {torch_version}, CUDA available"


def probe_import(path, module):
    """(ok, detail): `module` imports in that interpreter, with the launcher's third_party path."""
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return False, "not an executable file"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(ROOT / "third_party/nvdiffrast"), os.environ.get("PYTHONPATH", "")]))
    try:
        out = subprocess.run([path, "-c", f"import {module}"], capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        return False, str(error)
    if out.returncode:
        return False, (out.stderr.strip().splitlines() or ["import failed"])[-1]
    return True, f"import {module}"


def probe_nvcc():
    """(ok, detail): the CUDA 12.8 compiler the simulator's extensions need at run time."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return False, "nvcc not on PATH; put $CUDA_HOME/bin (CUDA 12.8) on PATH, or gsplat disables itself"
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{nvcc}: {error}"
    match = re.search(r"release (\d+\.\d+)", out)
    release = match.group(1) if match else "?"
    return release == "12.8", f"{nvcc} (CUDA {release})" + ("" if release == "12.8" else "; the simulator is built against 12.8")


def gpu_table():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    gpus = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus[parts[0]] = dict(name=parts[1], memory=parts[2], compute_cap=parts[3])
    return gpus


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default=str(ROOT / "OdysseyBenchmark/agents/ltf_sdroute.yaml"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--md5", action="store_true", help="verify the scene download against MD5SUMS (slow)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ
    r = Report()

    try:
        check_benchmark_environment(env)
        r.add("no ODYSSEY_* variables the launcher refuses", True)
    except EnvironmentRefused as error:
        r.add("no ODYSSEY_* variables the launcher refuses", False, " ".join(str(error).split()))
    ok, detail = probe_nvcc()
    r.add("nvcc", ok, detail)

    for var in ("ODYSSEY_SIM_PY", "ODYSSEY_FIXER_PY"):
        ok, detail = probe_python(env.get(var, ""))
        r.add(f"{var} ({env.get(var, 'unset')})", ok, detail)
    ok, detail = probe_import(env.get("ODYSSEY_SIM_PY", ""), "nvdiffrast.torch")
    r.add("nvdiffrast in ODYSSEY_SIM_PY", ok, detail if ok else f"{detail}; see docs/installation.md")

    zoo = Path(env.get("ODYSSEY_ZOO_ROOT", ROOT / "OdysseyZoo"))
    r.add(f"OdysseyZoo at {zoo}", (zoo / "sdroute/graph.py").is_file(), "sdroute/graph.py" if (zoo / "sdroute/graph.py").is_file() else "missing sdroute/graph.py")

    agent = None
    try:
        agent = load_agent(args.agent)
        r.add(f"agent config {args.agent}", True, agent.model.name)
    except (OSError, ValueError) as error:
        r.add(f"agent config {args.agent}", False, error)
    if agent is not None:
        problems = agent.verify_paths()
        r.add("model paths (python, repo, agent yaml, checkpoint, adapter)", not problems, "; ".join(problems))
        ok, detail = probe_python(agent.model.python)
        r.add(f"model.python ({agent.model.python})", ok, detail)

    scenes_root = env.get("ODYSSEY_SCENES_ROOT")
    rows = []
    if not scenes_root:
        r.add("ODYSSEY_SCENES_ROOT", False, "unset; point it at the odyssey-scenes download")
    else:
        root = Path(scenes_root)
        try:
            rows = scenes.release_table(root)
            missing = {}
            for row in rows:
                lost = [f for f in scenes.RELEASE_FILES if not (root / row["scene"] / f).is_file()]
                if lost:
                    missing[row["scene"]] = lost
            r.add(f"scenes at {root}: {len(rows)} in scenes.csv", len(rows) == 100 and not missing,
                  f"{len(rows)} scenes; incomplete: {missing}" if missing else f"{len(rows)} scenes, all files present")
            if args.md5 and (root / "MD5SUMS").is_file():
                out = subprocess.run(["md5sum", "-c", "--quiet", "MD5SUMS"], cwd=root, capture_output=True, text=True)
                r.add("MD5SUMS", out.returncode == 0, out.stdout.strip()[-400:] or "all files match")
        except (OSError, ValueError) as error:
            r.add(f"scenes at {root}", False, error)

    gpus = [g for g in re.split(r"[,\s]+", args.gpus) if g]
    table = gpu_table()
    for g in gpus:
        r.add(f"GPU {g}", g in table, table.get(g, "not visible to nvidia-smi"))
    visibility = gpu_visibility_error(gpus, env)
    if visibility:
        r.add("GPUs inside CUDA_VISIBLE_DEVICES", False, visibility)
    base = os.path.realpath(env["NUPLAN_MAPS_ROOT"]) if env.get("NUPLAN_MAPS_ROOT") else ""
    m = re.fullmatch(r"(.*?)(\d+)", base)
    slots = env.get("ODYSSEY_MAP_SLOTS")
    slot_list = [s for s in re.split(r"[,\s]+", slots) if s] if slots else ([m.group(1) + g for g in gpus] if m else [])
    if not slot_list:
        r.add("map slots", False, f"cannot derive one slot per GPU from NUPLAN_MAPS_ROOT={base!r}; set ODYSSEY_MAP_SLOTS"
              if base else "NUPLAN_MAPS_ROOT is unset; point it at a map copy (OdysseyBenchmark/scripts/make_map_slots.sh)")
    else:
        for g, slot in zip(gpus, slot_list):
            r.add(f"map slot for GPU {g}: {slot}", os.path.isfile(os.path.join(slot, "nuplan-maps-v1.0.json")),
                  "" if os.path.isfile(os.path.join(slot, "nuplan-maps-v1.0.json")) else "nuplan-maps-v1.0.json missing")
        r.add("map slots distinct", len({os.path.realpath(s) for s in slot_list[:len(gpus)]}) == len(gpus))

    fixer_root = Path(env.get("ODYSSEY_FIXER_ROOT", ROOT / "OdysseyRenderer/fixer"))
    weights = fixer_root / "models/finetuned/h1_b16_e1_model_11001.pkl"
    r.add("Fixer weights h1_b16_e1_model_11001.pkl", weights.is_file(), "" if weights.is_file() else f"missing {weights}")

    free = shutil.disk_usage(ROOT).free / 1e9
    r.add(f"free disk under {ROOT}: {free:.0f} GB", free >= 20, "" if free >= 20 else "20 GB recommended per campaign")

    if args.json:
        print(json.dumps(dict(checks=r.rows, gpus=table, map_slots=dict(zip(gpus, slot_list)), scenes=len(rows)), indent=1))
    else:
        for row in r.rows:
            print(f"[{'ok' if row['ok'] else 'FAIL'}] {row['check']}" + (f": {row['detail']}" if row["detail"] else ""))
        print("all checks passed" if not r.failed else f"{len(r.failed)} check(s) failed")
    return 0 if not r.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
