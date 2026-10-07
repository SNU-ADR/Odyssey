from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from pathlib import PosixPath
from pathlib import Path
import io
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        if hasattr(getattr(self, "_safedrive_model", None), "swnet_ins_reg_branch"):
            for key, value in features.items():
                if isinstance(value, list):
                    if isinstance(value[0], list):
                        matric_list = []
                        for v_ in value:
                            matric_list.append([v.unsqueeze(0)  for v in v_])
                        features[key] = matric_list
                    else:
                        if isinstance(value[0], PosixPath):

                            lidar_feature = []
                            for p in value:
                                with open(Path('dataset/sensor_blobs/test') / p, "rb") as fp:
                                    lidar_pc_byte = io.BytesIO(fp.read())
                                lidar_tensor = torch.tensor(
                                    LidarPointCloud.from_buffer(lidar_pc_byte, "pcd").points.T
                                )
                                lidar_feature.append(lidar_tensor[:, :5])
                            features[key] = [lidar_feature]
                        else:
                            features[key] = [v.unsqueeze(0) for v in value]
                elif hasattr(value, "to_tensor"):          # CameraPaths (cache_camera_images=False)
                    features[key] = value.to_tensor().unsqueeze(0)
                else:
                    features[key] = value.unsqueeze(0)
            with torch.no_grad():
                predictions = self.forward(features)
                poses = predictions["trajectory"].squeeze(0).numpy()

        else:
            # add batch dimension
            features = {k: v.unsqueeze(0) for k, v in features.items()}

            # forward pass
            with torch.no_grad():
                predictions = self.forward(features)
                poses = predictions["trajectory"].squeeze(0).numpy()
        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []
