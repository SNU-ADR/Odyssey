from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.agent_lightning_module_ema import AgentLightningModule_EMA, ModelCheckpointEMA

from pytorch_lightning import loggers
from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch
import collections
import collections.abc
import io
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from navsim.agents.safedrive.safedrive_features import CameraPaths
import pickle

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)


class SaveBeforeValidation(Callback):
    """Write <dirpath>/pre_validation.ckpt as validation starts inside fit.

    Lightning 2.2 runs validation before on_train_epoch_end and ModelCheckpoint writes only after
    it, so a run that dies in validation loses the whole epoch. One rolling file, overwritten each
    time; its own name, because a name ModelCheckpoint also uses would make it version the next
    save to -v1. Resume an interrupted phase from this file (+resume_from=True +ckpt_path=<this file>).
    """

    def __init__(self, dirpath):
        self.path = dirpath.rstrip("/") + "/pre_validation.ckpt"

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        trainer.save_checkpoint(self.path)   # every rank calls it; rank 0 writes, then a barrier

try:
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE
    WANDB_AVAILABLE = bool(_WANDB_AVAILABLE)   # lightning rejects wandb < 0.12.10
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


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

class _DropLidarFeature:
    """Collate wrapper for camera-only models (agent config use_lidar=False).

    A feature cache built with a lidar config still carries lidar_feature (.pcd paths), and the custom
    collate loads every one of them although the camera-only model never reads it: wasted I/O, and a
    FileNotFoundError when only the camera blobs are on disk. Drop the key before collating. A class,
    not a lambda, so dataloader workers can pickle it.
    """

    def __init__(self, collate):
        self.collate = collate

    def __call__(self, batch):
        return self.collate([({k: v for k, v in f.items() if k != "lidar_feature"}, t) for f, t in batch])


def custom_collate_fn(batch):
    elem = batch[0]

    # path-only camera cache (cache_camera_images=False): decode + undistort + resize here, in the
    # dataloader worker, so the GPU process only ever sees the (B, F, N, 3, 256, 512) uint8 tensor
    if isinstance(elem, CameraPaths):
        return torch.stack([e.to_tensor() for e in batch], dim=0)

    if isinstance(elem, Path):
        result = []
        for p in batch:
            try:
                with open(Path('dataset/sensor_blobs/trainval') / p, "rb") as fp:
                    lidar_pc_byte = io.BytesIO(fp.read())
            except:
                with open(Path('dataset/sensor_blobs/test') / p, "rb") as fp:
                    lidar_pc_byte = io.BytesIO(fp.read())
            lidar_tensor = torch.tensor(
                LidarPointCloud.from_buffer(lidar_pc_byte, "pcd").points.T
            )
            result.append(lidar_tensor[:, :5])
        return result


    if isinstance(elem, torch.Tensor):
        try:
            return torch.stack(batch, dim=0)
        except RuntimeError:
            return batch


    elif isinstance(elem, collections.abc.Mapping):
        TOKEN_KEYS = {
            "agent_tokens",
            "agent_tokens_vehicle",
            "agent_tokens_pedestrian",
        }
        out = {}
        for key in elem:
            vals = [d[key] for d in batch]

            if key in TOKEN_KEYS:
                out[key] = vals
            else:
                out[key] = custom_collate_fn(vals)
        return out


    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(custom_collate_fn(samples) for samples in zip(*batch)))


    elif isinstance(elem, tuple):
        return tuple(custom_collate_fn(samples) for samples in zip(*batch))


    elif isinstance(elem, collections.abc.Sequence) and not isinstance(elem, (str, bytes)):
        it = iter(batch)
        length = len(next(it))
        if not all(len(x) == length for x in it):
            return batch
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]


    else:
        return batch

from functools import lru_cache
import collections

@lru_cache(maxsize=20000)
def load_safety_by_token(token: str):
    full_path = Path("dataset/extra_data/planning_vb/formatted_pdm_score_8448") / f"formatted_pdm_score_8448_{token}.pkl"
    with open(full_path, "rb") as f:
        gt = pickle.load(f)['trajectory_scores']

    def as_tensor(x):
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)  # zero-copy
        return torch.tensor(x)         # fallback
    return (
        as_tensor(gt['no_at_fault_collisions']),
        as_tensor(gt['drivable_area_compliance']),
        as_tensor(gt['driving_direction_compliance']),
        as_tensor(gt['ego_progress']),
        as_tensor(gt['time_to_collision_within_bound']),
        as_tensor(gt['comfort']),
        as_tensor(gt['score']),
        gt['collision_tokens'],
        gt['drivable_area_compliance_timestep']
    )

@lru_cache(maxsize=10000)
def load_pcd_points(abs_path: str):

    try:
        p = Path('dataset/sensor_blobs/trainval') / abs_path
        with open(p, "rb") as fp:
            lidar_pc_byte = io.BytesIO(fp.read())
    except:
        p = Path('dataset/sensor_blobs/test') / abs_path
        with open(p, "rb") as fp:
            lidar_pc_byte = io.BytesIO(fp.read())
    pts = LidarPointCloud.from_buffer(lidar_pc_byte, "pcd").points.T
    import numpy as np
    return torch.from_numpy(np.ascontiguousarray(pts[:, :5]))

def custom_collate_fn_w_safety_gt(batch):
    elem = batch[0]

    if isinstance(elem, CameraPaths):
        return torch.stack([e.to_tensor() for e in batch], dim=0)

    if isinstance(elem, Path):
        return [load_pcd_points(str(p)) for p in batch]

    if isinstance(elem, torch.Tensor):
        try:
            return torch.stack(batch, dim=0)
        except RuntimeError:
            return batch

    elif isinstance(elem, collections.abc.Mapping):
        out = {}
        for key in elem:
            values = [d[key] for d in batch]
            if key == "scene_frame_token":
                out[key] = values
                NC, DAC, EP, TTC, DDC, C, SCORE = [], [], [], [], [], [], []
                collision_tokens_list, dac_time_list = [], []
                for tok in values:
                    n,d,e,t,dd,c,s,coll,dact = load_safety_by_token(tok)
                    NC.append(n); DAC.append(d); EP.append(e)
                    TTC.append(t); DDC.append(dd); C.append(c); SCORE.append(s)
                    collision_tokens_list.append(coll); dac_time_list.append(dact)
                out['safety_scores_dict'] = {
                    'NC_list': torch.stack(NC),
                    'DAC_list': torch.stack(DAC),
                    'EP_list': torch.stack(EP),
                    'TTC_list': torch.stack(TTC),
                    'DDC_list': torch.stack(DDC),
                    'C_list': torch.stack(C),
                    'pdm_score_list': torch.stack(SCORE),
                    'collision_tokens_list': collision_tokens_list,
                    'dac_time_list': dac_time_list,
                }
            else:
                out[key] = custom_collate_fn_w_safety_gt(values)
        return out

    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(custom_collate_fn_w_safety_gt(samples) for samples in zip(*batch)))

    elif isinstance(elem, tuple):
        return tuple(custom_collate_fn_w_safety_gt(samples) for samples in zip(*batch))

    elif isinstance(elem, collections.abc.Sequence) and not isinstance(elem, (str, bytes)):
        it = iter(batch)
        length = len(next(it))
        if not all(len(x) == length for x in it):
            return batch
        transposed = zip(*batch)
        return [custom_collate_fn_w_safety_gt(samples) for samples in transposed]

    else:
        return batch


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    if "debug" in list(cfg.keys()):
        if cfg.debug:
            cfg.train_logs = OmegaConf.create(['2021.06.09.12.51.31_veh-35_03869_04221'])
            cfg.val_logs = OmegaConf.create(['2021.06.09.12.51.31_veh-35_03869_04221'])

    logger.info("Building Lightning Module")

    check_unused_parameter=False

    if "check_unused_parameter" in list(cfg.keys()):
        if cfg.check_unused_parameter:
            check_unused_parameter = True

    if getattr(cfg, "ema", False):
        lightning_module = AgentLightningModule_EMA(
            agent=agent,
            check_unused_parameter=check_unused_parameter
        )
    else:
        lightning_module = AgentLightningModule(
            agent=agent,
            check_unused_parameter=check_unused_parameter
        )

    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path:
        lightning_module = AgentLightningModule.load_from_checkpoint(
            ckpt_path,
            agent=agent,
            check_unused_parameter=check_unused_parameter,
        )

    if getattr(cfg, "safety_scoring_ray", False):
        agent.safety_scoring_ray = cfg.safety_scoring_ray
        agent._config.safety_scoring_ray = cfg.safety_scoring_ray

    if hasattr(agent, "set_ray"):
        agent.set_ray()

    if getattr(cfg, "agent_name", False):
        lightning_module.agent_name = cfg.agent_name

    collate_fn = None
    if getattr(cfg, "second_lidar", False):
        if getattr(cfg, "collate_fn_w_safety_gt", False):
            collate_fn = custom_collate_fn_w_safety_gt
        else:
            collate_fn = custom_collate_fn
    if not getattr(getattr(agent, "_config", None), "use_lidar", True):
        collate_fn = _DropLidarFeature(collate_fn or torch.utils.data.default_collate)
    if getattr(cfg, "test_traj_save", False) == False:
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
        train_shuffle = True
        if "train_shuffle" in list(cfg.keys()):
            train_shuffle = False
        train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=train_shuffle, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False, collate_fn=collate_fn)
        logger.info("Num training samples: %d", len(train_data))
        logger.info("Num validation samples: %d", len(val_data))


    if getattr(cfg, "test_traj_save", False) :
        tb_logger = loggers.TensorBoardLogger(save_dir=cfg.output_dir,name="lightning_logs_test",version="")
        exp_name_suffix = "test"
    elif getattr(cfg, "validate_only", False):
        tb_logger = loggers.TensorBoardLogger(save_dir=cfg.output_dir,name="lightning_logs_val",version="")
        exp_name_suffix = "val"
    else:
        tb_logger = loggers.TensorBoardLogger(save_dir=cfg.output_dir,name="lightning_logs",version="")
        exp_name_suffix = None

    loggers_ = [tb_logger]
    # opt-in: +use_wandb=True on the command line, and only for real training runs
    if getattr(cfg, "use_wandb", False) and WANDB_AVAILABLE and exp_name_suffix is None:
        project = getattr(cfg, "wandb_project", "navsim")
        wandb_run_name = Path(cfg.experiment_name).name
        wandb_logger = WandbLogger(project=project, name=wandb_run_name, save_dir=cfg.output_dir)
        loggers_.append(wandb_logger)
        wandb_logger.experiment.define_metric("trainer/global_step")
        wandb_logger.experiment.define_metric("*", step_metric="trainer/global_step")
        logger.info(f"WandB logging enabled: project={project}, run={wandb_run_name}")


    strategy = cfg.trainer.params.strategy
    if cfg.trainer.params.strategy == 'ddp' and "ddp_find_unused_parameters" in list(cfg.keys()):
        if cfg.ddp_find_unused_parameters:
            strategy = DDPStrategy(find_unused_parameters=True)
    del cfg.trainer.params.strategy

    if "every_n_epochs" in list(cfg.keys()):
        every_n_epochs=cfg.every_n_epochs
    else:
        every_n_epochs=1

    if getattr(cfg, "ema", False):
        checkpoint_callback = ModelCheckpointEMA(
            dirpath = cfg.output_dir + '/lightning_logs/checkpoints/',
            filename='{epoch}-{step}',
            save_top_k=-1,
            save_last=True,          # phase 2 -> phase 3 hands over through last.ckpt
            every_n_epochs=every_n_epochs
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath = cfg.output_dir + '/lightning_logs/checkpoints/',
            filename='{epoch}-{step}',
            save_top_k=-1,
            save_last=True,          # phase 2 -> phase 3 hands over through last.ckpt
            every_n_epochs=every_n_epochs
        )

    training_callbacks = agent.get_training_callbacks()
    training_callbacks.append(checkpoint_callback)
    if getattr(cfg, "save_before_validation", False):
        training_callbacks.append(SaveBeforeValidation(cfg.output_dir + '/lightning_logs/checkpoints/'))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, strategy=strategy, logger=loggers_, callbacks=training_callbacks)
    if getattr(cfg, "validate_only", False) or getattr(cfg, "test_traj_save", False):
        test_traj_save = False
        if getattr(cfg, "test_traj_save", False):
            test_traj_save = True
            from navsim.common.dataclasses import SensorConfig
            from navsim.common.dataloader import MetricCacheLoader
            scene_loader = SceneLoader(
                sensor_blobs_path=None,
                data_path=Path(cfg.navsim_log_path),
                scene_filter=instantiate(cfg.train_test_split.scene_filter),
                sensor_config=SensorConfig.build_no_sensors(),
            )
            metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
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
            val_dataloader = DataLoader(
                dataset,
                **cfg.dataloader.params,
                shuffle=False,
                collate_fn=custom_collate_fn if getattr(cfg, "second_lidar", False) else None,
            )

        ckpt_path = getattr(cfg, "ckpt_path", None)
        if ckpt_path:
            lightning_module = AgentLightningModule.load_from_checkpoint(
                ckpt_path,
                agent=agent,
                check_unused_parameter=check_unused_parameter,
            )

        if getattr(cfg, "agent_name", False):
            lightning_module.agent_name = cfg.agent_name

        if getattr(cfg, "test_save_name", False):
            test_save_name = f"{cfg.test_save_name}"
            lightning_module.scoring_test_save_name = f"{cfg.test_save_name}.pkl"
        else:
            test_save_name = "predictions"
            lightning_module.scoring_test_save_name = "predictions.pkl"

        if "imi_test_weight" in list(cfg.keys()):
            agent._safedrive_model.imi_test_weight = float(cfg.imi_test_weight)
        if getattr(cfg, "NC_test_weight", False):
            agent._safedrive_model.NC_test_weight = float(cfg.NC_test_weight)
        if getattr(cfg, "DAC_test_weight", False):
            agent._safedrive_model.DAC_test_weight = float(cfg.DAC_test_weight)
        if getattr(cfg, "EP_test_weight", False):
            agent._safedrive_model.EP_test_weight = float(cfg.EP_test_weight)
        if getattr(cfg, "TTC_test_weight", False):
            agent._safedrive_model.TTC_test_weight = float(cfg.TTC_test_weight)
        if getattr(cfg, "W_test_weight", False):
            agent._safedrive_model.W_test_weight = float(cfg.W_test_weight)
        if getattr(cfg, "C_test_weight", False):
            agent._safedrive_model.C_test_weight = float(cfg.C_test_weight)
        if getattr(cfg, "pdm_score_test_weight", False):
            agent._safedrive_model.pdm_score_test_weight = float(cfg.pdm_score_test_weight)
        if getattr(cfg, "DDC_test_weight", False):
            agent._safedrive_model.DDC_test_weight = float(cfg.DDC_test_weight)
        if getattr(cfg, "TLC_test_weight", False):
            agent._safedrive_model.TLC_test_weight = float(cfg.TLC_test_weight)
        if getattr(cfg, "LK_test_weight", False):
            agent._safedrive_model.LK_test_weight = float(cfg.LK_test_weight)
        if getattr(cfg, "HC_test_weight", False):
            agent._safedrive_model.HC_test_weight = float(cfg.HC_test_weight)
        if getattr(cfg, "scoring_test", False):
            agent._safedrive_model.scoring_test = cfg.scoring_test
            imi = agent._safedrive_model.imi_test_weight
            NC = agent._safedrive_model.NC_test_weight
            DAC = agent._safedrive_model.DAC_test_weight
            EP = agent._safedrive_model.EP_test_weight
            TTC = agent._safedrive_model.TTC_test_weight
            W = agent._safedrive_model.W_test_weight
            C = agent._safedrive_model.C_test_weight
            pdm = agent._safedrive_model.pdm_score_test_weight
            DDC = agent._safedrive_model.DDC_test_weight
            TLC = agent._safedrive_model.TLC_test_weight
            LK = agent._safedrive_model.LK_test_weight
            HC = agent._safedrive_model.HC_test_weight
            lightning_module.scoring_test_save_name = f'{test_save_name}_IMI{imi}_NC{NC}_DAC{DAC}_EP{EP}_TTC{TTC}_W{W}_C{C}_PDM{pdm}_DDC{DDC}_TLC{TLC}_LK{LK}_HC{HC}.pkl'








        if getattr(cfg, "pair_NC_scoring", False):
            agent._safedrive_model.pair_NC_scoring = True
            agent._safedrive_model.PwNC_test_weight = 1.0
            if getattr(cfg, "PwNC_test_weight", False):
                agent._safedrive_model.PwNC_test_weight = float(cfg.PwNC_test_weight)
            lightning_module.scoring_test_save_name = lightning_module.scoring_test_save_name.replace('.pkl', f'_PwNC{agent._safedrive_model.PwNC_test_weight}.pkl')


        if getattr(cfg, "twdac_scoring", False):
            agent._safedrive_model.twdac_scoring = True
            agent._safedrive_model.TwDAC_test_weight = 1.0
            if getattr(cfg, "TwDAC_test_weight", False): # 1
                agent._safedrive_model.TwDAC_test_weight = float(cfg.TwDAC_test_weight)
                lightning_module.scoring_test_save_name = lightning_module.scoring_test_save_name.replace('.pkl', f'_TwDAC{agent._safedrive_model.TwDAC_test_weight}.pkl')
            if getattr(cfg, "TwDAC_bevseg_pred", False): # 6
                agent._safedrive_model.TwDAC_bevseg_pred = True
                lightning_module.scoring_test_save_name = lightning_module.scoring_test_save_name.replace('.pkl', '_BEVSegPred.pkl')
            if getattr(cfg, "TwDAC_bbox_margin", False): # 8
                agent._safedrive_model.TwDAC_bbox_margin = (float(cfg.TwDAC_bbox_margin[0]), float(cfg.TwDAC_bbox_margin[1]))
                lightning_module.scoring_test_save_name = lightning_module.scoring_test_save_name.replace('.pkl', f'_TwDACMargin{cfg.TwDAC_bbox_margin[0]}_{cfg.TwDAC_bbox_margin[1]}.pkl')


        if getattr(cfg, "vis_save_name", False):
            agent._safedrive_model._config.vis_save_name = cfg.vis_save_name

        if getattr(cfg, "pkl_save_name", False):
            lightning_module.scoring_test_save_name = lightning_module.scoring_test_save_name.replace('.pkl', f'{cfg.pkl_save_name}.pkl')


        if getattr(cfg, "debug", False):
            lightning_module.debug_mode = True
        if getattr(cfg, "visualize", False):
            lightning_module.agent._safedrive_model.visualize = True
        agent._safedrive_model._config.agent_name = cfg.agent_name

        lightning_module.test_traj_save = test_traj_save
        trainer.validate(model=lightning_module, dataloaders=val_dataloader)
        return

    if getattr(cfg, "visualize", False):
        lightning_module.agent._safedrive_model.visualize = True
    logger.info("Starting Training")
    if getattr(cfg, "resume_from", False):
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
        )
    else:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )


if __name__ == "__main__":
    main()