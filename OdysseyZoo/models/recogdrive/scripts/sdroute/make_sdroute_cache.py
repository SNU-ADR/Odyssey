#!/usr/bin/env python3
"""Build <out>/<log>/<token>/sdroute_target.gz for the ReCogDrive SD-route arm.

Runs with this repo's navsim on PYTHONPATH and nothing else (see run_sdroute_cache.sh). The route has
to be the tensor the other SD-route models train on; this repo's PDM helpers yield it bit for bit.

The route itself is built by the sdroute wrapper (route_target.py), loaded by file path, which is what LTF /
DiffusionDrive / DrivoR / SafeDrive call -- same frame, same arguments.
"""
import argparse
import gzip
import importlib.util
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

SCENE_FILTER_KEYS = ("num_history_frames", "num_future_frames", "frame_interval", "has_route", "max_scenes")
_WRAPPER = None


def _default_wrapper() -> str:
    """The SD-route wrapper of the neighbouring LTF tree, OdysseyZoo/models/navsim."""
    return str(Path(__file__).resolve().parents[3] / "navsim/navsim/agents/sdroute/route_target.py")


def _load_wrapper(path: str):
    # Cached per process: the route builder builds the per-city SD graph behind an lru_cache,
    # and reloading the module for every task would rebuild it each time.
    global _WRAPPER
    if _WRAPPER is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"SD-route wrapper not found: {path}")
        spec = importlib.util.spec_from_file_location("_sdroute_route_target", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _WRAPPER = module.build_sdroute_target
    return _WRAPPER


def _dump(path: Path, data) -> None:
    # Same format navsim's Dataset writes: gzip pickle, compresslevel 1.
    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wb", compresslevel=1) as f:
        pickle.dump(data, f)
    os.replace(tmp, path)


def _run_logs(payload):
    logs, cfg = payload
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    build_sdroute_target = _load_wrapper(cfg["wrapper"])
    scene_filter = SceneFilter(**cfg["scene_filter"])
    scene_filter.log_names = list(logs)
    scene_filter.tokens = cfg["tokens"]
    loader = SceneLoader(
        data_path=Path(cfg["data_path"]),
        sensor_blobs_path=Path(cfg["sensor_blobs_path"]),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    out_root = Path(cfg["out"])
    written = skipped = empty = 0
    t0 = time.time()
    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        meta = scene.scene_metadata
        token_dir = out_root / meta.log_name / meta.initial_token
        target = token_dir / "sdroute_target.gz"
        if target.exists() and not cfg["overwrite"]:
            skipped += 1
            continue
        frame_idx = meta.num_history_frames - 1
        route, mask = build_sdroute_target(scene, frame_idx)
        if not bool(mask.any()):
            empty += 1
        token_dir.mkdir(parents=True, exist_ok=True)
        _dump(target, {"route_centerline": route, "route_centerline_mask": mask})
        written += 1
    return {"logs": len(logs), "tokens": len(loader.tokens), "written": written,
            "skipped": skipped, "empty": empty, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="cache root; files land at <out>/<log>/<token>/sdroute_target.gz")
    ap.add_argument("--split-yaml", default="navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
                    help="scene filter yaml of this repo (its token list is what the training cache uses)")
    ap.add_argument("--wrapper", default=_default_wrapper(),
                    help="path to the sdroute wrapper route_target.py (default: the neighbouring LTF tree's)")
    ap.add_argument("--data-path", default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/navsim_logs/trainval")
    ap.add_argument("--sensor-blobs-path", default=os.environ.get("OPENSCENE_DATA_ROOT", "") + "/sensor_blobs/trainval")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--num-logs", type=int, default=0, help="0 = all logs of the split")
    ap.add_argument("--log-stride", type=int, default=1, help="take every Nth log (with --num-logs, a spread-out subset)")
    ap.add_argument("--log-offset", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import navsim
    print(f"navsim from: {navsim.__file__}", flush=True)

    split = yaml.load(open(args.split_yaml), Loader=yaml.CSafeLoader)
    scene_filter = {k: split[k] for k in SCENE_FILTER_KEYS if k in split}
    tokens = list(split["tokens"])
    logs = list(split["log_names"])
    available = {p.stem for p in Path(args.data_path).glob("*.pkl")}
    logs = [log for log in logs if log in available]
    logs = logs[args.log_offset::args.log_stride]
    if args.num_logs:
        logs = logs[: args.num_logs]
    print(f"logs: {len(logs)}  tokens in split: {len(tokens)}  workers: {args.workers}  out: {args.out}", flush=True)

    cfg = {
        "wrapper": args.wrapper,
        "scene_filter": scene_filter,
        "tokens": tokens,
        "data_path": args.data_path,
        "sensor_blobs_path": args.sensor_blobs_path,
        "out": args.out,
        "overwrite": args.overwrite,
    }
    totals = {"tokens": 0, "written": 0, "skipped": 0, "empty": 0}
    t0 = time.time()
    if args.workers <= 1:
        results = [_run_logs((logs, cfg))]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_logs, ([log], cfg)) for log in logs]
            for done, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if done % 25 == 0 or done == len(futures):
                    written = sum(x["written"] for x in results)
                    dt = time.time() - t0
                    eta = (len(futures) - done) * dt / done
                    print(f"  {done}/{len(futures)} logs, {written} written, {dt:.0f}s elapsed, "
                          f"~{eta / 60:.0f} min left", flush=True)
    for r in results:
        for k in totals:
            totals[k] += r[k]
    dt = time.time() - t0
    rate = totals["written"] / dt if dt > 0 else 0.0
    print(f"DONE tokens={totals['tokens']} written={totals['written']} skipped={totals['skipped']} "
          f"empty_route={totals['empty']} ({100.0 * totals['empty'] / max(totals['written'], 1):.2f}%) "
          f"in {dt:.0f}s ({rate:.1f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
