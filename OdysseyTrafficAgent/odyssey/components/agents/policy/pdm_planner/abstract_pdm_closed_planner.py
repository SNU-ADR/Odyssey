# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from typing import List, Optional
import logging

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from odyssey.components.agents.policy.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from odyssey.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from odyssey.components.agents.policy.pdm_planner.proposal.pdm_generator import PDMGenerator
from odyssey.components.agents.policy.pdm_planner.proposal.pdm_proposal import PDMProposalManager
from odyssey.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
from odyssey.components.agents.policy.pdm_planner.simulation.pdm_simulator import PDMSimulator
from odyssey.components.agents.policy.pdm_planner.utils.pdm_emergency_brake import PDMEmergencyBrake
from odyssey.components.agents.policy.pdm_planner.utils.pdm_geometry_utils import parallel_discrete_path
from odyssey.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath


class AbstractPDMClosedPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

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
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(AbstractPDMClosedPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, map_radius
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(
            proposal_sampling, lane_keeping_weight=lane_keeping_weight
        )
        self._emergency_brake = PDMEmergencyBrake(trajectory_sampling)

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None
        self.logger = logging.getLogger("PDMPlanner")

    def _update_proposal_manager(self, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        current_lane = self._get_starting_lane(ego_state)

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(current_lane)

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(current_lane.speed_limit_mps)

    def _get_proposal_paths(
        self, current_lane: LaneGraphEdgeMapObject
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)

        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
        as_trajectory: bool = False,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :param as_trajectory: True returns an InterpolatedTrajectory (unrolled to the full
            trajectory horizon); False (default) returns the winning proposal's raw state
            array, which is what dense_reward_manager consumes. See the branch at the end.
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state

        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        # 5. Score proposals
        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        # Two callers want two DIFFERENT things out of this, so `as_trajectory` picks:
        #
        #   as_trajectory=False (default, unchanged): the raw state ndarray of the winning
        #     proposal. dense_reward_manager relies on this -- it np.concatenates the result
        #     with other proposal arrays (dense_reward_manager.py:293), which an object would
        #     break. Every pre-existing caller keeps its exact current behaviour.
        #
        #   as_trajectory=True: a real InterpolatedTrajectory, via the generator's own
        #     unroll. This is what the method's `-> InterpolatedTrajectory` annotation always
        #     claimed and what odyssey_to_pdm_utils.convert_to_trajectory needs (it reads
        #     `._trajectory`); with the raw array that caller died on
        #       AttributeError: 'numpy.ndarray' object has no attribute '_trajectory'
        #     which is why ego_policy=pdm_policy had never run. It is also the only correct
        #     plan for driving: the simulated array spans just the PROPOSAL horizon (40
        #     poses), while generate_trajectory unrolls the winner to the full trajectory
        #     horizon (50) and wraps it as EgoStates -- the upstream nuPlan PDM-Closed path.
        best_idx = int(np.argmax(proposal_scores))
        # Expose the winning proposal for simulator diagnostics. This is observational only;
        # both return branches below still consume the same argmax as before.
        self._last_best_idx = best_idx
        self._last_proposal_scores = proposal_scores.copy()
        if as_trajectory:
            return self._generator.generate_trajectory(best_idx)
        return simulated_proposals_array[best_idx]
