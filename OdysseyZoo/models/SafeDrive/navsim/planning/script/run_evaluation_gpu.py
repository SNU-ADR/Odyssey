from pathlib import Path
import torch
import logging
import pickle

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from typing import List, Tuple, Dict, Any
import pandas as pd

from navsim.common.dataclasses import SensorConfig
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.run_pdm_score import run_pdm_score

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.planning.training.dataset import Dataset
from navsim.planning.script.run_training import custom_collate_fn


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def build_test_dataset(cfg: DictConfig, agent: AbstractAgent) -> Dataset:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: test dataset
    """
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = list(set(scene_filter.tokens) & set(metric_cache_loader.tokens))

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    test_data = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        return_token=True,
    )
    return test_data


def _to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to device, descending into list/tuple/dict.

    With second_lidar the collate returns a list of tensors, not a tensor, so a plain
    torch.is_tensor check would leave the lidar on CPU and spconv would assert.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def save_pred(cfg: DictConfig, dataloader: DataLoader, agent: AbstractAgent) -> str:
    """
    Saves predictions of the agent to a file.
    :param cfg: omegaconf dictionary
    :return: path to the saved predictions
    """
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    pred_save_path = Path(cfg.output_dir) / "predictions.pkl"

    if not pred_save_path.parent.exists():
        pred_save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving predictions to {pred_save_path}")

    agent.eval()
    pred_traj_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for features, targets in tqdm(dataloader, desc="Saving predictions", unit="batch"):
        features = _to_device(features, device)
        if targets is not None:
            targets = _to_device(targets, device)
        with torch.no_grad():
            try:
                pred = agent.forward(features, targets)
            except TypeError:


                pred = agent.forward(features)
            pred_traj = pred['trajectory'].detach().cpu().numpy()
            pred_traj_dict.update({token: pred_traj[i] for i, token in enumerate(features['token'])})

    with open(pred_save_path, 'wb') as f:
        pickle.dump(pred_traj_dict, f)

    return str(pred_save_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    worker = build_worker(cfg)

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    data_points = []
    log_files_all = []
    tokens_all = []
    for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items():
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        })
        log_files_all.append(log_file)
        tokens_all.extend(tokens_list)

    if getattr(cfg, "score_cache_path", None):
        logger.info(f"Using score cache path: {cfg.score_cache_path}")
        agent_config = {
            "_target_": "navsim.agents.load_score_cache_agent.LoadScoreCacheAgent",
            "_convert_": "all",
            "score_cache_path": cfg.score_cache_path,
            "imi_test_weight": cfg.get("imi_test_weight", 1.0),
            "NC_test_weight": cfg.get("NC_test_weight", 0.0),
            "DAC_test_weight": cfg.get("DAC_test_weight", 0.0),
            "EP_test_weight": cfg.get("EP_test_weight", 0.0),
            "TTC_test_weight": cfg.get("TTC_test_weight", 0.0),
            "C_test_weight": cfg.get("C_test_weight", 0.0),
            "pdm_score_test_weight": cfg.get("pdm_score_test_weight", 0.0),
            "DDC_test_weight": cfg.get("DDC_test_weight", 0.0),
            "TLC_test_weight": cfg.get("TLC_test_weight", 0.0),
            "LK_test_weight": cfg.get("LK_test_weight", 0.0),
            "HC_test_weight": cfg.get("HC_test_weight", 0.0),
            "PwNC_test_weight": cfg.get("PwNC_test_weight", 0.0),
            "TwDAC_test_weight": cfg.get("TwDAC_test_weight", 0.0),
        }
        cfg.agent = OmegaConf.create(agent_config)
        save_csv = getattr(cfg, "save_csv", None) or "result_score_cache.csv"
    elif cfg.pred_traj_path is None:
        assert getattr(cfg, "checkpoint_path") is not None, "checkpoint_path must be provided in the config"
        agent: AbstractAgent = instantiate(cfg.agent, checkpoint_path=cfg.checkpoint_path)
        agent.initialize()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.to(device)

        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_files_all
        scene_filter.tokens = list(set(tokens_all) & set(metric_cache_loader.tokens))
        scene_loader_inference = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=agent.get_sensor_config(),
        )
        dataset = Dataset(
            scene_loader=scene_loader_inference,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            return_token=True,
        )
        dataloader = DataLoader(
            dataset,
            **cfg.dataloader.params,
            shuffle=False,
            collate_fn=custom_collate_fn if getattr(cfg, "second_lidar", False) else None,
        )
        pl.seed_everything(cfg.seed, workers=True)
        logger.info(f"Global Seed set to {cfg.seed}")

        pred_traj_path = save_pred(cfg, dataloader, agent)
        save_csv = "result.csv"
        cfg.agent = OmegaConf.create({
            "_target_": "navsim.agents.load_prediction_agent.LoadPredictionAgent",
            "_convert_": "all",
            "pred_traj_path": pred_traj_path
        })
    else:
        logger.info(f"Using provided prediction trajectory path: {cfg.pred_traj_path}")
        pred_traj_path = cfg.pred_traj_path
        if pred_traj_path == 'predictions.pkl':
            save_csv = 'result.csv'
        else:
            save_csv = pred_traj_path.split('/')[-1].replace(".pkl", ".csv").replace('predictions', 'result')
        cfg.agent = OmegaConf.create({
            "_target_": "navsim.agents.load_prediction_agent.LoadPredictionAgent",
            "_convert_": "all",
            "pred_traj_path": pred_traj_path
        })

    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    pdm_score_df.to_csv(save_path /save_csv)

    num_successful = int(pdm_score_df["valid"].sum())
    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_successful}.
            Number of failed scenarios: {len(pdm_score_df) - num_successful}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / save_csv}.
        """
    )


if __name__ == "__main__":
    main()
