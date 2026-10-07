from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.sdroute.route_target import build_sdroute_target
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        # The SD-route arm needs the Scene at inference to rebuild the route (see compute_trajectory);
        # the scorers pass it only when requires_scene is True.
        super().__init__(
            trajectory_sampling,
            requires_scene=config.use_sdroute,
        )

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(self._trajectory_sampling, config)
        # The decoder reads the route out of `targets`, so the training loop must pass them.
        self.forward_takes_targets = config.use_sdroute

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Transfuser only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass.

        `targets` carries the route the SD-route module reads. At inference it comes from
        compute_trajectory() (open loop) or from the closed-loop simulator; when it is absent
        the module leaves the query untouched rather than inventing a route.
        """
        return self._transfuser_model(features, targets)

    def compute_trajectory(self, agent_input, scene=None):
        """Open-loop inference; the SD-route arm also rebuilds the route from the Scene.

        The base implementation calls forward(features) without targets, so the route branch
        would be skipped without any error. This override builds the route with the same
        builder and frame index as TransfuserTargetBuilder, so the model sees at eval the
        tensor it was trained on.

        `scene` is passed because requires_scene is True for the SD-route arm (see __init__);
        baseline agents fall through to the base implementation.
        """
        if not self.forward_takes_targets:
            return super().compute_trajectory(agent_input)
        if scene is None:
            raise RuntimeError(
                "TransfuserAgent: the route arm needs the Scene to build route_centerline, but "
                "compute_trajectory() was called without one. The caller must pass the Scene "
                "(run_pdm_score.py and run_pdm_score_one_stage.py do this when agent.requires_scene is True)."
            )

        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # Same as TransfuserTargetBuilder.compute_targets; keep the two in step.
        frame_idx = scene.scene_metadata.num_history_frames - 1
        targets: Dict[str, torch.Tensor] = {}
        route, route_mask = build_sdroute_target(scene, frame_idx)
        targets["route_centerline"] = route.unsqueeze(0)
        targets["route_centerline_mask"] = route_mask.unsqueeze(0)

        with torch.no_grad():
            predictions = self.forward(features, targets)
            poses = predictions["trajectory"].squeeze(0).numpy()

        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
