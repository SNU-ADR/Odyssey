# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging
from typing import Dict
from omegaconf import DictConfig

from odyssey.envs.build_env import build_env

logger = logging.getLogger(__name__)


def build_envs(cfg: DictConfig, scene_dict: Dict[str, dict]):
    """
    Build the simulation environment: one env, in this process, for all scenes.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :param scene_dict: Dict[str, dict]. Dictionary of scenes.
    :return A list holding the env.
    """
    logger.info('Building environments...')
    logger.info('Building simulations from %d scenes...', len(scene_dict))
    envs = [build_env(cfg, name='main_thread', data=scene_dict)]
    logger.info('Building environments...DONE!')
    return envs
