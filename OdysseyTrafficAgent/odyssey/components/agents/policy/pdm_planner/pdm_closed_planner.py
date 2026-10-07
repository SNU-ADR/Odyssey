# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import gc
import logging
import warnings
from typing import List, Optional, Type

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2

from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

from odyssey.components.agents.policy.pdm_planner.abstract_pdm_closed_planner import (
    AbstractPDMClosedPlanner,
)
from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from odyssey.components.agents.policy.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PDMClosedPlanner(AbstractPDMClosedPlanner):
    """PDM-Closed planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        lane_keeping_weight: float = 0.0,
    ):
        """
        Constructor for PDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """
        super(PDMClosedPlanner, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
            lane_keeping_weight,
        )

    def initialize(self, initialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization["map_api"]
        self._route_is_gt_matched = bool(initialization.get("route_is_gt_matched", False))
        reference_centerline = initialization.get("reference_centerline")
        if reference_centerline is None:
            self._reference_centerline_discrete_path = None
        else:
            reference_centerline = np.asarray(reference_centerline, dtype=np.float64)
            self._reference_centerline_discrete_path = [
                StateSE2(float(x), float(y), float(heading))
                for x, y, heading in reference_centerline
            ]
        self._load_route_dicts(initialization["route_roadblock_dict_ids"])
        
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input, as_trajectory: bool = False
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api, ego_state, self._map_radius
        )

        trajectory = self._get_closed_loop_trajectory(current_input, as_trajectory=as_trajectory)

        self._iteration += 1
        return trajectory
