from typing import Tuple
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)


def _cfg_int(cfg_node, key, env_name, hardcoded_default):
    """int from cfg_node[key], or hardcoded_default when unset; a non-empty env var wins over both.

    cfg.checkpointing keeps the checkpoint policy with the rest of the run config. The env vars
    (CKPT_EVERY_EPOCH, PERIODIC_CKPT_STEPS, ...) override it from the shell.
    """
    raw = os.environ.get(env_name)
    if raw not in (None, ""):
        return int(raw)
    if cfg_node is None:
        return int(hardcoded_default)
    value = cfg_node.get(key, None)
    return int(hardcoded_default if value is None else value)


CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"



def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'last_hidden_state': last_hidden_state,
    }
    targets = {
        'trajectory': trajectory
    }
    # SD route (agent.use_sdroute): cached as a target, read by the planner as conditioning.
    if 'route_centerline' in targets_list[0]:
        targets['route_centerline'] = torch.stack([t['route_centerline'] for t in targets_list], dim=0).cpu()
        targets['route_centerline_mask'] = torch.stack([t['route_centerline_mask'] for t in targets_list], dim=0).cpu()

    return features, targets, tokens_list


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    # gloo (PDM_DIST_BACKEND=gloo) when several ranks share one GPU: NCCL
    # refuses two ranks on the same device. Lightning finds this group already up and reuses it.
    dist.init_process_group(
        backend=os.getenv('PDM_DIST_BACKEND', 'nccl'),
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # If rich is installed Lightning may pick RichProgressBar, which prints nothing when stdout is not
    # a terminal; TQDM keeps progress visible in log files.
    #
    # Checkpoint callbacks: the monitored one (val/loss_epoch) writes nothing when validation is off
    # (limit_val_batches=0); the periodic ones in cfg.checkpointing write either way:
    #   every_n_epochs: N      -- every Nth epoch, kept, named periodic-epoch=<n>.
    #   every_n_train_steps: N -- every N optimizer steps, named periodic-step=<n>.
    # step_save_top_k defaults to -1 here (keep every step file): RL mid-epoch checkpoints are
    # scored, not just kept as spares.
    ck = cfg.get("checkpointing", None)
    every_epoch = _cfg_int(ck, "every_n_epochs", "CKPT_EVERY_EPOCH", 0)
    periodic_steps = _cfg_int(ck, "every_n_train_steps", "PERIODIC_CKPT_STEPS", 0)
    step_keep = _cfg_int(ck, "step_save_top_k", "PERIODIC_CKPT_KEEP", -1)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=(ck.get("monitor", "val/loss_epoch") if ck is not None else "val/loss_epoch"),
            mode='min',
            save_top_k=_cfg_int(ck, "save_top_k", "CKPT_SAVE_TOP_K", 5),
            every_n_epochs=1,
        ),
        pl.callbacks.TQDMProgressBar(refresh_rate=1),
    ]
    if every_epoch > 0:
        callbacks.insert(1, pl.callbacks.ModelCheckpoint(
            monitor=None, save_top_k=-1, every_n_epochs=every_epoch, filename="periodic-{epoch}"
        ))
    if periodic_steps > 0:
        callbacks.insert(1, pl.callbacks.ModelCheckpoint(
            monitor=None, save_top_k=step_keep, every_n_train_steps=periodic_steps,
            filename="periodic-{step}"
        ))
    if every_epoch <= 0 and periodic_steps <= 0:
        logger.warning(
            "Only the val/loss_epoch-monitored callback will write checkpoints, and it writes none "
            "when validation is off (limit_val_batches=0). Set checkpointing.every_n_epochs=1 "
            "(and/or checkpointing.every_n_train_steps)."
        )
    # W&B logging, opt-in: set WANDB_PROJECT or wandb.project. Without it Lightning keeps its
    # default TensorBoard logger. WANDB_MODE=offline logs locally (wandb sync later).
    pl_logger = True
    wb = cfg.get("wandb", None)
    # WANDB_PROJECT set but empty disables W&B even when wandb.project is set.
    if "WANDB_PROJECT" in os.environ:
        wandb_project = os.environ["WANDB_PROJECT"]
    else:
        wandb_project = (wb.get("project", "") if wb is not None else "") or ""
    if wandb_project:
        from pytorch_lightning.loggers import WandbLogger

        pl_logger = WandbLogger(
            project=wandb_project,
            name=(os.environ.get("WANDB_NAME")
                  or (wb.get("name", None) if wb is not None else None)
                  or cfg.experiment_name),
            save_dir=str(cfg.output_dir),
            config=OmegaConf.to_container(cfg.agent, resolve=True),
        )

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks, logger=pl_logger)

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
