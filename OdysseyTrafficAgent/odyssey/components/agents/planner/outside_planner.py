# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.common.dataclasses import Trajectory
from odyssey.engine.engine_utils import get_engine

class OutsidePlanner:
    def __init__(self, agent):
        self.agent = agent
        self._traj_info = self.agent.object_track

    def reset(self, agent):
        self.agent = agent
        self._traj_info = self.agent.object_track

    def get_trajectory(self, step: int) -> Trajectory:
        return self.agent.client.get_trajectory(step)

    @property
    def engine(self):
        return get_engine()
