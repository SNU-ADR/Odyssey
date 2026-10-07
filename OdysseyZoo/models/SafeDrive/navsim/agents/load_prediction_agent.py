import pickle

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig


class LoadPredictionAgent(AbstractAgent):
    """Privileged agent interface of human operator."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        pred_traj_path: str = None,
    ):
        """
        Initializes the human agent object.
        :param trajectory_sampling: trajectory sampling specification
        """
        self._trajectory_sampling = trajectory_sampling
        with open(pred_traj_path, 'rb') as file:
            self._pred_dict = pickle.load(file)

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        token = scene.frames[scene.scene_metadata.num_history_frames-1].token

        assert token in self._pred_dict, f"Token {token} not found in prediction dictionary."
        poses = self._pred_dict[token]
        return Trajectory(poses)
