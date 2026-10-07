"""Add the SD route to an existing SafeDrive feature cache, without rebuilding it.

`SafeDrive_TargetBuilder` emits `route_centerline` / `route_centerline_mask` only when
`use_sdroute` is set, and its `get_unique_name()` is unchanged, so a cache built before the
route existed keeps the same filename and still passes the Dataset's validity check. Re-running
run_dataset_caching.py would therefore work but would redo everything -- image paths, BEV
semantic rasters, agent targets -- for a key that costs a map lookup. This script rewrites only
the target pickle, adding the two route keys. cache_safedrive.sh runs it as step 3.

    # from the repository root
    python navsim/planning/script/run_sdroute_backfill.py \
        agent=safedrive_sdroute_agent \
        experiment_name=caching/safedrive_sdroute_backfill \
        train_test_split=navtrain \
        cache_path=$PWD/exp/safedrive_train_cache \
        worker.threads_per_node=32 \
        +apply=true                     # without this it is a dry run

The route builder is navsim/agents/sdroute/route_target.py; the import fails loudly if it is absent.
"""
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.safedrive.safedrive_features import SafeDrive_TargetBuilder
from navsim.agents.sdroute.route_target import build_sdroute_target
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import (
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

ROUTE_KEYS = ("route_centerline", "route_centerline_mask")


def backfill_logs(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """Rewrite one worker's share of target pickles. Returns per-log counters."""
    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    apply: bool = bool(args[0]["apply"])

    # Only the config is instantiated, not the agent: building SafeDrive_Model per worker would
    # cost minutes and a few GB for a filename and a map lookup.
    config = instantiate(cfg.agent.config)
    assert getattr(config, "use_sdroute", False), (
        "pass an SD-route agent config; with use_sdroute=False the target builder emits no route"
    )
    target_name = SafeDrive_TargetBuilder(config=config).get_unique_name() + ".gz"

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),  # the route needs the map and ego poses only
    )

    cache_path = Path(cfg.cache_path)
    counts = {"written": 0, "already": 0, "uncached": 0}
    for token in scene_loader.tokens:
        scene = scene_loader.get_scene_from_token(token)
        meta = scene.scene_metadata
        target_file = cache_path / meta.log_name / meta.initial_token / target_name
        if not target_file.is_file():
            # This token was never cached; backfill does not create new cache entries.
            counts["uncached"] += 1
            continue

        target_dict = load_feature_target_from_pickle(target_file)
        if all(k in target_dict for k in ROUTE_KEYS):
            counts["already"] += 1
            continue

        frame_idx = meta.num_history_frames - 1
        route, route_mask = build_sdroute_target(scene, frame_idx)
        target_dict["route_centerline"] = route
        target_dict["route_centerline_mask"] = route_mask

        if apply:
            # The target pickle is expensive to rebuild (BEV semantic rasters), and this is the
            # only copy, so never write over it in place: a crash mid-write would lose the token.
            tmp = target_file.with_suffix(".gz.tmp")
            dump_feature_target_to_pickle(tmp, target_dict)
            os.replace(tmp, target_file)
        counts["written"] += 1

    return [counts]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for the SD-route cache backfill."""
    pl.seed_everything(0, workers=True)
    apply = bool(cfg.get("apply", False))
    if not apply:
        logger.warning("DRY RUN: routes are built and counted but nothing is written. Add +apply=true.")

    cache_path = Path(cfg.cache_path)
    assert cache_path.is_dir(), f"cache_path {cache_path} does not exist; nothing to backfill"

    worker: WorkerPool = Sequential() if "debug" in list(cfg.keys()) else instantiate(cfg.worker)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(f"Backfilling SD route into {len(scene_loader)} cached scenes at {cache_path}")

    data_points = [
        {"cfg": cfg, "log_file": log_file, "tokens": tokens_list, "apply": apply}
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    results = worker_map(worker, backfill_logs, data_points)

    totals = {"written": 0, "already": 0, "uncached": 0}
    for counts in results:
        for k in totals:
            totals[k] += counts[k]
    verb = "wrote" if apply else "would write"
    logger.info(
        f"{verb} {totals['written']}, already had the route {totals['already']}, "
        f"not in the cache {totals['uncached']}"
    )


if __name__ == "__main__":
    main()
