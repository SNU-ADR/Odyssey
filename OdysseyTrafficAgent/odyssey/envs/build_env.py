# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from omegaconf import DictConfig, OmegaConf

from odyssey.envs.base_env import BaseEnv


def build_env(config, name, data):
    """
    Return the navigation class for target object.
    """
    # Resolve OmegaConf interpolations (like ${now:...}) before passing to env, so the env holds
    # plain values that no longer depend on Hydra's resolvers.
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
        config = OmegaConf.create(config)

    if config["env"] == "base_env":
        return BaseEnv(config, name, data)
    else:
        raise NotImplementedError(f'The assigned env {config["env"]} is not'
                                  f'implemented.')
