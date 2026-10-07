# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.policy.base_policy import BasePolicy
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.common.dataclasses import Trajectory
from odyssey.engine.engine_utils import get_engine


class TrajectoryPolicy(BasePolicy):

    def __init__(self, agent, random_seed=None, config=None):
        super(TrajectoryPolicy, self).__init__(agent=agent, random_seed=random_seed, config=config)

        self._traj_info = self.agent.object_track
        self._valid_indicator = self._traj_info[SD.VALID] == 1
        self._traj_length = len(self._valid_indicator)

    @property
    def is_current_step_valid(self):

        index = max(int(self.traj_step), 0)
        return bool(index < self._traj_length and self._valid_indicator[index])

    def act(self, *args, **kwargs):
        """
        Return a set of waypoints.
        """

        index = max(int(self.traj_step), 0)

        if not self.is_current_step_valid:
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        return Trajectory(
            waypoints=self._traj_info[SD.POSITION][index:index + 5, :2],
            velocities=self._traj_info['velocity'][index:index + 5],
            headings=self._traj_info[SD.HEADING][index:index + 5],
            angular_velocities=self._traj_info["angular_velocity"][index:index + 5],
            wp_dt=get_engine().sim_dt,   # scene log rows
        )
