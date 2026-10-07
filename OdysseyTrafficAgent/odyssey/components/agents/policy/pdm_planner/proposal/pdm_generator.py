# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import copy
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import transform
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import distance, line_locate_point
from shapely.geometry import Point, Polygon
from shapely.geometry.base import CAP_STYLE

from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from odyssey.components.agents.policy.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_array_representation import (
    state_array_to_ego_states,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_enums import (
    LeadingAgentIndex,
    StateIDMIndex,
    StateIndex,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_geometry_utils import (
    normalize_angle,
)


class PDMGenerator:
    """Class to generate proposals in PDM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        leading_agent_update_rate: int = 2,
    ):
        """
        Constructor of PDMGenerator
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param leading_agent_update_rate: sample update-rate of leading agent state, defaults to 2
        """
        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "PDMGenerator: Proposals and Trajectory must have equal interval length!"

        # trajectory config
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._sample_interval: float = trajectory_sampling.interval_length

        # generation config
        self._leading_agent_update: int = leading_agent_update_rate

        # lazy loaded
        self._state_array: Optional[npt.NDArray[np.float64]] = None
        self._state_idm_array: Optional[npt.NDArray[np.float64]] = None
        self._leading_agent_array: Optional[npt.NDArray[np.float64]] = None
        self._leading_agent_tokens: Optional[npt.NDArray[np.object_]] = None

        self._proposal_manager: Optional[PDMProposalManager] = None
        self._observation: Optional[PDMObservation] = None

        self._initial_ego_state: Optional[EgoState] = None
        self._vehicle_parameters: Optional[VehicleParameters] = None

        # caches
        self._driving_corridor_cache: Optional[Dict[int, Polygon]] = None
        self._time_point_list: Optional[List[TimePoint]] = None
        # The same proposal rails are reused across planner iterations.  Keep an ordered
        # progress cursor per lateral rail so a self-overlapping route (e.g. a legitimate
        # P-turn) cannot jump to a later branch under spatial-only projection.
        self._progress_manager_id: Optional[int] = None
        self._last_ego_progress_by_lateral: Dict[int, float] = {}

    def generate_proposals(
        self,
        initial_ego_state: EgoState,
        observation: PDMObservation,
        proposal_manager: PDMProposalManager,
    ) -> npt.NDArray[np.float64]:
        """
        Generates proposals by unrolling IDM policies vor varying paths,
        and saving the proposal states in array representation.
        :param initial_ego_state: state of ego-vehicle at t=0
        :param observation: PDMObservation class
        :param proposal_manager: PDMProposalManager class
        :return: unrolled proposal states in array representation
        """
        self._reset(initial_ego_state, observation, proposal_manager)
        self._initialize_time_points()

        # unroll proposals per path, to interpolate along batch-dim
        lateral_batch_dict = self._get_lateral_batch_dict()

        for lateral_idx, lateral_batch_idcs in lateral_batch_dict.items():
            self._initialize_states(lateral_batch_idcs)
            for time_idx in range(1, self._proposal_sampling.num_poses + 1, 1):
                self._update_leading_agents(lateral_batch_idcs, time_idx)
                self._update_idm_states(lateral_batch_idcs, time_idx)
                self._update_states_se2(lateral_batch_idcs, time_idx)

        return self._state_array

    def generate_trajectory(
        self,
        proposal_idx: int,
    ) -> InterpolatedTrajectory:
        """
        Complete unrolling of final trajectory to number of trajectory samples.
        :param proposal_idx: index of best-scored proposal
        :return: InterpolatedTrajectory class
        """
        assert (
            len(self._time_point_list) == self._proposal_sampling.num_poses + 1
        ), "PDMGenerator: Proposals must be generated first!"

        lateral_batch_idcs = [proposal_idx]
        current_time_point = copy.deepcopy(self._time_point_list[-1])

        for time_idx in range(
            self._proposal_sampling.num_poses + 1,
            self._trajectory_sampling.num_poses + 1,
            1,
        ):
            current_time_point += TimePoint(int(self._sample_interval * 1e6))
            self._time_point_list.append(current_time_point)

            self._update_leading_agents(lateral_batch_idcs, time_idx)
            self._update_idm_states(lateral_batch_idcs, time_idx)
            self._update_states_se2(lateral_batch_idcs, time_idx)

        # convert array representation to list of EgoState class
        ego_states: List[EgoState] = state_array_to_ego_states(
            self._state_array[proposal_idx],
            self._time_point_list,
            self._vehicle_parameters,
        )
        return InterpolatedTrajectory(ego_states)

    def _reset(
        self,
        initial_ego_state: EgoState,
        observation: PDMObservation,
        proposal_manager: PDMProposalManager,
    ) -> None:
        """
        Re-initializes several class attributes for unrolling in new iteration
        :param initial_ego_state: ego-vehicle state at t=0
        :param observation: PDMObservation class
        :param proposal_manager: PDMProposalManager class
        """

        # lazy loading
        self._proposal_manager: PDMProposalManager = proposal_manager
        self._observation: PDMObservation = observation

        manager_id = id(proposal_manager)
        if manager_id != self._progress_manager_id:
            self._progress_manager_id = manager_id
            self._last_ego_progress_by_lateral = {}

        self._initial_ego_state = initial_ego_state
        self._vehicle_parameters = initial_ego_state.car_footprint.vehicle_parameters

        # reset proposal state arrays
        self._state_array: npt.NDArray[np.float64] = np.zeros(
            (
                len(self._proposal_manager),
                self._trajectory_sampling.num_poses + 1,
                StateIndex.size(),
            ),
            dtype=np.float64,
        )  # x, y, heading
        self._state_idm_array: npt.NDArray[np.float64] = np.zeros(
            (len(self._proposal_manager), self._trajectory_sampling.num_poses + 1, 2),
            dtype=np.float64,
        )  # progress, velocity
        self._leading_agent_array: npt.NDArray[np.float64] = np.zeros(
            (len(self._proposal_manager), self._trajectory_sampling.num_poses + 1, 3),
            dtype=np.float64,
        )  # progress, velocity, rear-length
        self._leading_agent_tokens = np.full(
            (len(self._proposal_manager), self._trajectory_sampling.num_poses + 1),
            "",
            dtype=object,
        )

        # reset caches
        self._driving_corridor_cache: Dict[int, Polygon] = {}

        self._time_point_list: List[TimePoint] = []
        self._updated: bool = True

    def _initialize_time_points(self) -> None:
        """Initializes a list of TimePoint objects for proposal horizon."""
        current_time_point = copy.deepcopy(self._initial_ego_state.time_point)
        self._time_point_list = [current_time_point]
        for time_idx in range(1, self._proposal_sampling.num_poses + 1, 1):
            current_time_point += TimePoint(int(self._sample_interval * 1e6))
            self._time_point_list.append(copy.deepcopy(current_time_point))

    def _initialize_states(self, lateral_batch_idcs: List[int]) -> None:
        """
        Initializes all state arrays for ego, IDM, and leading agent at t=0
        :param lateral_batch_idcs: list of proposal indices, sharing a path.
        """

        # all initial states are identical for shared lateral_idx
        # thus states are created for lateral_batch_idcs[0] and repeated
        dummy_proposal_idx = lateral_batch_idcs[0]

        proposal = self._proposal_manager[dummy_proposal_idx]
        ego_position = Point(*self._initial_ego_state.rear_axle.point.array)
        previous_progress = self._last_ego_progress_by_lateral.get(
            proposal.lateral_idx
        )
        ego_progress = proposal.path.project_near(
            ego_position,
            reference_progress=previous_progress,
            heading=float(self._initial_ego_state.rear_axle.heading),
        )
        self._last_ego_progress_by_lateral[proposal.lateral_idx] = ego_progress
        ego_velocity = self._initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.x

        self._state_idm_array[
            lateral_batch_idcs, 0, StateIDMIndex.PROGRESS
        ] = ego_progress
        self._state_idm_array[
            lateral_batch_idcs, 0, StateIDMIndex.VELOCITY
        ] = ego_velocity

        state_array = self._proposal_manager[dummy_proposal_idx].path.interpolate(
            [ego_progress], as_array=True
        )[0]
        self._state_array[lateral_batch_idcs, 0, StateIndex.STATE_SE2] = state_array

    def _update_states_se2(self, lateral_batch_idcs: List[int], time_idx: int) -> None:
        """
        Updates state array for ego, at current time-step.
        :param lateral_batch_idcs: list of proposal indices, sharing a path.
        :param time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"
        dummy_proposal_idx = lateral_batch_idcs[0]
        current_progress = self._state_idm_array[
            lateral_batch_idcs, time_idx, StateIDMIndex.PROGRESS
        ]
        states_se2_array: npt.NDArray[np.float64] = self._proposal_manager[
            dummy_proposal_idx
        ].path.interpolate(current_progress, as_array=True)
        self._state_array[
            lateral_batch_idcs, time_idx, StateIndex.STATE_SE2
        ] = states_se2_array

    def _update_idm_states(self, lateral_batch_idcs: List[int], time_idx: int) -> None:
        """
        Updates idm state array, by propagating policy for one step.
        :param lateral_batch_idcs: list of proposal indices, sharing a path.
        :param time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"
        longitudinal_idcs = [
            self._proposal_manager[proposal_idx].longitudinal_idx
            for proposal_idx in lateral_batch_idcs
        ]
        next_idm_states = self._proposal_manager.longitudinal_policies.propagate(
            self._state_idm_array[lateral_batch_idcs, time_idx - 1],
            self._leading_agent_array[lateral_batch_idcs, time_idx],
            longitudinal_idcs,
            self._sample_interval,
        )
        self._state_idm_array[lateral_batch_idcs, time_idx] = next_idm_states

    def _update_leading_agents(
        self, lateral_batch_idcs: List[int], time_idx: int
    ) -> None:
        """
        Update leading agent state array by searching for agents/obstacles in driving corridor.
        :param lateral_idx: index indicating the path of proposals
        :param lateral_batch_idcs: list of proposal indices, sharing a path.
        :param time_idx: index of unrolling iteration (for proposal/trajectory samples)
        """
        assert time_idx > 0, "PDMGenerator: call _initialize_states first!"

        # update leading agent state at first call or at update rate (runtime)
        update_leading_agent: bool = (time_idx % self._leading_agent_update) == 0

        if not update_leading_agent:
            self._leading_agent_array[
                lateral_batch_idcs, time_idx
            ] = self._leading_agent_array[lateral_batch_idcs, time_idx - 1]

        else:
            dummy_proposal_idx = lateral_batch_idcs[0]

            leading_agent_array = np.zeros(len(LeadingAgentIndex), dtype=np.float64)
            intersecting_objects: List[str] = self._get_intersecting_objects(
                lateral_batch_idcs, time_idx
            )

            # collect all leading vehicles ones for all proposals (run-time)
            object_progress_dict: Dict[str, float] = {}
            eligible_objects = [
                object_token
                for object_token in intersecting_objects
                if object_token not in self._observation.collided_track_ids
            ]
            observation = self._observation[time_idx]
            if eligible_objects:
                centroids = np.asarray(
                    [observation[object_token].centroid for object_token in eligible_objects],
                    dtype=object,
                )
                object_progresses = line_locate_point(
                    self._proposal_manager[dummy_proposal_idx].linestring,
                    centroids,
                )
                object_progress_dict = dict(zip(eligible_objects, object_progresses))

            # select leading agent for each proposal individually
            for proposal_idx in lateral_batch_idcs:
                current_ego_progress = self._state_idm_array[
                    proposal_idx, time_idx - 1, StateIDMIndex.PROGRESS
                ]

                # filter all objects ahead
                agents_ahead: Dict[str, float] = {
                    agent: progress
                    for agent, progress in object_progress_dict.items()
                    if progress > current_ego_progress
                }

                if len(agents_ahead) > 0:  # red light, object or agent ahead
                    current_state_se2 = StateSE2(
                        *self._state_array[
                            proposal_idx, time_idx - 1, StateIndex.STATE_SE2
                        ]
                    )
                    ego_polygon: Polygon = CarFootprint.build_from_rear_axle(
                        current_state_se2, self._vehicle_parameters
                    ).oriented_box.geometry

                    ahead_tokens = list(agents_ahead)
                    relative_distances = distance(
                        ego_polygon,
                        np.asarray(
                            [observation[agent] for agent in ahead_tokens],
                            dtype=object,
                        ),
                    )

                    argmin = np.argmin(relative_distances)
                    nearest_agent = ahead_tokens[argmin]
                    self._leading_agent_tokens[proposal_idx, time_idx] = nearest_agent

                    # add rel. distance for red light, object or agent
                    relative_distance = (
                        current_ego_progress + relative_distances[argmin]
                    )
                    leading_agent_array[LeadingAgentIndex.PROGRESS] = relative_distance

                    # calculate projected velocity if not red light
                    if self._observation.red_light_token not in nearest_agent:
                        leading_agent_array[
                            LeadingAgentIndex.VELOCITY
                        ] = self._get_leading_agent_velocity(
                            current_state_se2.heading,
                            self._observation.unique_objects[nearest_agent],
                        )

                else:  # nothing ahead, free driving
                    path_length = self._proposal_manager[proposal_idx].linestring.length
                    path_rear = self._vehicle_parameters.length / 2

                    leading_agent_array[LeadingAgentIndex.PROGRESS] = path_length
                    leading_agent_array[LeadingAgentIndex.LENGTH_REAR] = path_rear
                    self._leading_agent_tokens[
                        proposal_idx, time_idx
                    ] = "__path_end__"

                self._leading_agent_array[proposal_idx, time_idx] = leading_agent_array

    @staticmethod
    def _get_leading_agent_velocity(ego_heading: float, agent: SceneObject) -> float:
        """
        Calculates velocity of leading vehicle projected to ego's heading.
        :param ego_heading: heading angle [rad]
        :param agent: SceneObject class
        :return: projected velocity [m/s]
        """

        if isinstance(agent, Agent):  # dynamic object
            relative_heading = normalize_angle(agent.center.heading - ego_heading)
            projected_velocity = transform(
                StateSE2(agent.velocity.magnitude(), 0, 0),
                StateSE2(0, 0, relative_heading).as_matrix(),
            ).x
        else:  # static object
            projected_velocity = 0.0

        return projected_velocity

    def _get_intersecting_objects(
        self, lateral_batch_idcs: List[int], time_idx: int
    ) -> List[str]:
        """
        Returns and caches all intersecting objects for the proposals path and time-step.
        :param lateral_batch_idcs: list of proposal indices, sharing a path
        :param time_idx: index indicating the path of proposals
        :return: list of object tokens
        """
        dummy_proposal_idx = lateral_batch_idcs[0]
        driving_corridor: Polygon = self._get_driving_corridor(dummy_proposal_idx)
        return self._observation[time_idx].intersects(driving_corridor)

    def _get_driving_corridor(self, proposal_idx: int) -> Polygon:
        """
        Creates and caches driving corridor of ego-vehicle for each proposal path.
        :param proposal_idx: index of a proposal
        :return: linestring of max trajectory distance and ego's width
        """
        lateral_idx = self._proposal_manager[proposal_idx].lateral_idx

        if lateral_idx not in self._driving_corridor_cache.keys():
            ego_distance = self._state_idm_array[
                proposal_idx, 0, StateIDMIndex.PROGRESS
            ]
            trajectory_distance = (
                ego_distance
                + abs(self._proposal_manager.max_target_velocity)
                * self._trajectory_sampling.num_poses
                * self._sample_interval
            )
            linestring_ahead = self._proposal_manager[proposal_idx].path.substring(
                ego_distance, trajectory_distance
            )
            expanded_path = linestring_ahead.buffer(
                self._vehicle_parameters.width / 2, cap_style=CAP_STYLE.square
            )

            self._driving_corridor_cache[lateral_idx] = expanded_path

        return self._driving_corridor_cache[lateral_idx]

    def _get_lateral_batch_dict(self) -> Dict[int, List[int]]:
        """
        Creates a dictionary for lateral paths and their proposal indices.
        :return: dictionary of lateral and proposal indices
        """
        lateral_batch_dict: Dict[int, List[int]] = {}

        for proposal_idx in range(len(self._proposal_manager)):
            lateral_idx = self._proposal_manager[proposal_idx].lateral_idx

            if lateral_idx not in lateral_batch_dict.keys():
                lateral_batch_dict[lateral_idx] = [proposal_idx]
            else:
                lateral_batch_dict[lateral_idx].append(proposal_idx)

        return lateral_batch_dict
