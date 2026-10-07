from types import SimpleNamespace

import numpy as np
from unittest.mock import Mock, patch
from shapely.affinity import rotate, translate
from shapely.geometry import Point, box
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.observation.idm.idm_states import (
    IDMAgentState,
    IDMLeadAgentState,
)

from odyssey.components.agents.policy.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from odyssey.components.agents.policy.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from odyssey.components.agents.policy.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    WEIGHTED_METRICS_WEIGHTS,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex,
)
from odyssey.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from odyssey.components.agents.policy.nuplan_idm_policy import (
    _IDMPolicyWithEmergencyBrake,
    NuPlanIDMBatch,
)
from odyssey.components.agents.policy.pdm_planner.reference_config import (
    ACCEL_MAX,
    FALLBACK_TARGET_VELOCITY,
    LATERAL_OFFSETS,
    LQR_Q_LATERAL,
    LQR_Q_LONGITUDINAL,
    LQR_R_LATERAL,
    LQR_R_LONGITUDINAL,
    MAP_RADIUS,
    MIN_GAP_TO_LEAD_AGENT,
    PROPOSAL_NUM_POSES,
    SAMPLE_INTERVAL,
    SPEED_LIMIT_FRACTIONS,
    TRAJECTORY_NUM_POSES,
    build_reference_idm_policy,
)
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (
    OdysseyToNuPlanConverter,
)
from odyssey.manager import base_manager as base_manager_module
from odyssey.manager.agent_manager import BaseAgentManager, _nuplan_idm_controls_vehicle
from odyssey.manager.metric_manager import MetricManager
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.components.agents.policy.nuplan_idm_policy import (
    NuPlanIDMBatch,
    OPEN_LOOP_DETECTION_TYPES,
)
from odyssey.utils.nuplan_map_utils import route_roadblock_ids
from odyssey.utils.cadence import (
    source_frame_at_elapsed_time,
    synchronized_gt_warmup_steps,
)
from odyssey.components.agents.policy.pdm_policy import (
    _resolve_pdm_lateral_offsets,
)


def test_pdm_closed_configuration_matches_tuplan_garage_reference():
    assert (TRAJECTORY_NUM_POSES, PROPOSAL_NUM_POSES, SAMPLE_INTERVAL) == (80, 40, 0.1)
    assert SPEED_LIMIT_FRACTIONS == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert LATERAL_OFFSETS == (-1.0, 1.0)
    assert FALLBACK_TARGET_VELOCITY == 15.0
    assert MIN_GAP_TO_LEAD_AGENT == 1.0
    assert ACCEL_MAX == 1.5
    assert MAP_RADIUS == 50.0
    assert LQR_Q_LONGITUDINAL == (10.0,)
    assert LQR_R_LONGITUDINAL == (1.0,)
    assert LQR_Q_LATERAL == (1.0, 10.0, 0.0)
    assert LQR_R_LATERAL == (1.0,)

    policy = build_reference_idm_policy()
    assert np.allclose(policy._speed_limit_fractions, SPEED_LIMIT_FRACTIONS)
    assert np.allclose(policy._fallback_target_velocities, FALLBACK_TARGET_VELOCITY)
    assert np.allclose(policy._min_gap_to_lead_agent, MIN_GAP_TO_LEAD_AGENT)
    assert np.allclose(policy._accel_max, ACCEL_MAX)


def test_pdm_speed_scale_changes_only_the_ego_proposal_targets():
    policy = build_reference_idm_policy(speed_scale=0.75)

    assert np.allclose(
        policy._speed_limit_fractions,
        np.asarray(SPEED_LIMIT_FRACTIONS) * 0.75,
    )
    assert np.allclose(
        policy._fallback_target_velocities,
        FALLBACK_TARGET_VELOCITY * 0.75,
    )
    assert np.allclose(policy._min_gap_to_lead_agent, MIN_GAP_TO_LEAD_AGENT)
    assert np.allclose(policy._accel_max, ACCEL_MAX)


def test_pdm_lateral_offsets_are_configurable_without_changing_reference():
    assert _resolve_pdm_lateral_offsets({}) == list(LATERAL_OFFSETS)
    assert _resolve_pdm_lateral_offsets({"pdm_lateral_offsets": [-0.5, 0.5]}) == [
        -0.5,
        0.5,
    ]
    assert _resolve_pdm_lateral_offsets({"pdm_lateral_offsets": []}) == []
    assert _resolve_pdm_lateral_offsets({"pdm_lateral_offsets": None}) == []


def test_pdm_lateral_offsets_reject_non_finite_values():
    with np.testing.assert_raises_regex(ValueError, "finite distances"):
        _resolve_pdm_lateral_offsets({"pdm_lateral_offsets": [np.nan]})


def test_policy_lane_preference_does_not_mutate_reference_score_weights():
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    reference = PDMScorer(sampling)
    selector = PDMScorer(sampling, lane_keeping_weight=1.0)

    assert reference._weighted_metric_weights[
        WeightedMetricIndex.LANE_KEEPING
    ] == 0.0
    assert selector._weighted_metric_weights[
        WeightedMetricIndex.LANE_KEEPING
    ] == 1.0
    assert WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.LANE_KEEPING] == 0.0


def test_policy_lane_preference_rejects_invalid_weight():
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    with np.testing.assert_raises_regex(ValueError, "finite and non-negative"):
        PDMScorer(sampling, lane_keeping_weight=-1.0)


def test_nuplan_idm_emergency_limit_does_not_change_normal_gap_parameter():
    stock = IDMPolicy(10.0, 1.0, 1.5, 1.0, 2.0)
    emergency = _IDMPolicyWithEmergencyBrake(
        10.0, 1.0, 1.5, 1.0, 2.0, emergency_decel_max=4.905
    )
    agent = IDMAgentState(progress=0.0, velocity=10.0)
    close_lead = IDMLeadAgentState(progress=1.0, velocity=0.0, length_rear=0.0)

    # The analytical IDM request, including its desired-gap term, remains stock nuPlan.
    assert stock.idm_params == emergency.idm_params
    # Only the hard lower clamp differs once stock IDM asks for emergency braking.
    assert stock.solve_forward_euler_idm_policy(agent, close_lead, 0.1).velocity == 9.8
    assert np.isclose(
        emergency.solve_forward_euler_idm_policy(agent, close_lead, 0.1).velocity,
        10.0 - 0.4905,
    )


def test_source_ended_agent_gets_one_egress_tail_only_at_true_map_dead_end():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._source_end_route_extension_enabled = True
    batch._source_end_route_extension_length_m = 20.0
    batch._source_end_route_extension_trigger_m = 20.0
    batch._source_end_extended_tokens = set()
    batch._source_end_route_extensions = 0

    class _Agent:
        def __init__(self, outgoing_edges=()):
            self.end_segment = SimpleNamespace(outgoing_edges=outgoing_edges)
            self._path = create_path_from_se2(
                [
                    StateSE2(0.0, 0.0, 0.0),
                    StateSE2(5.0, 0.0, 0.0),
                    StateSE2(10.0, 0.0, 0.0),
                ]
            )
            self._state = SimpleNamespace(progress=8.0)
            self._requires_state_update = False

        def get_progress_to_go(self):
            return self._path.get_end_progress() - self._state.progress

        def get_path_to_go(self):
            return [StateSE2(8.0, 0.0, 0.0), StateSE2(10.0, 0.0, 0.0)]

    dead_end = _Agent()
    assert batch._extend_source_end_dead_end_path("dead-end", dead_end)
    assert np.isclose(dead_end._path.get_end_progress(), 22.0)
    assert dead_end._state.progress == 0.0
    assert dead_end._requires_state_update
    assert not batch._extend_source_end_dead_end_path("dead-end", dead_end)
    assert batch._source_end_route_extensions == 1

    red_light_wait = _Agent(outgoing_edges=(SimpleNamespace(id="connector"),))
    original_end_progress = red_light_wait._path.get_end_progress()
    assert not batch._extend_source_end_dead_end_path("red", red_light_wait)
    assert np.isclose(red_light_wait._path.get_end_progress(), original_end_progress)


def test_route_exhaustion_despawns_any_agent_at_true_idm_dead_end():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._source_end_route_exhaustion_despawn_enabled = True
    batch._source_end_route_exhaustion_margin_m = 0.5
    batch._source_end_route_exhaustion_despawns = 0
    class _Agent:
        def __init__(self, remaining, outgoing=(), excluded=()):
            self.length = 4.0
            self.end_segment = SimpleNamespace(outgoing_edges=outgoing)
            self._odyssey_excluded_connector_ids = excluded
            self._remaining = remaining

        def is_active(self, _step):
            return True

        def has_valid_path(self):
            return True

        def get_progress_to_go(self):
            return self._remaining

    manager = SimpleNamespace(agents={
        "exhausted": _Agent(2.4),
        "still-observed": _Agent(1.0),
        "intermittent-gap": _Agent(1.0),
        "red-or-short-route": _Agent(1.0, outgoing=(SimpleNamespace(id="next"),)),
        "excluded-only": _Agent(
            1.0, outgoing=(SimpleNamespace(id="47421"),), excluded={"47421"}
        ),
        "far-from-end": _Agent(5.0),
    })

    assert batch._collect_dead_end_route_exhaustions(manager, 10) == {
        "exhausted", "still-observed", "intermittent-gap", "excluded-only"
    }
    assert batch._source_end_route_exhaustion_despawns == 4


def test_dead_end_queue_relief_retires_only_stalled_lead_with_follower():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._dead_end_queue_relief_enabled = True
    batch._dead_end_queue_relief_lane_ids = {"68604"}
    batch._dead_end_queue_relief_timeout_ticks = 4
    batch._dead_end_queue_relief_progress_epsilon_m = 0.5
    batch._dead_end_queue_relief_stopped_speed_mps = 0.5
    batch._dead_end_queue_relief_trigger_m = 5.0
    batch._dead_end_queue_relief_follower_distance_m = 30.0
    batch._dead_end_queue_progress = {}
    batch._dead_end_queue_retirements = 0

    class _Agent:
        def __init__(self, progress_to_go, x, velocity=0.0, lane="68604", outgoing=()):
            self._progress_to_go = progress_to_go
            self._pose = StateSE2(x, 0.0, 0.0)
            self.velocity = velocity
            self.end_segment = SimpleNamespace(id=lane, outgoing_edges=outgoing)

        def is_active(self, _step):
            return True

        def has_valid_path(self):
            return True

        def get_progress_to_go(self):
            return self._progress_to_go

        def to_se2(self):
            return self._pose

    lead = _Agent(3.0, 0.0)
    follower = _Agent(15.0, -12.0)
    manager = SimpleNamespace(agents={"lead": lead, "follower": follower})

    for step in range(4):
        assert not batch._collect_dead_end_queue_retirements(manager, step)
    assert batch._collect_dead_end_queue_retirements(manager, 4) == {"lead"}
    assert batch._dead_end_queue_retirements == 1

    # A red-light route still has an outgoing graph edge and must never enter this timer.
    red = _Agent(2.0, 0.0, outgoing=(SimpleNamespace(id="green-after-red"),))
    manager.agents = {"red": red, "follower": follower}
    assert not batch._collect_dead_end_queue_retirements(manager, 20)
    assert "red" not in batch._dead_end_queue_progress


def test_dead_end_queue_relief_needs_a_follower_and_resets_after_progress():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._dead_end_queue_relief_enabled = True
    batch._dead_end_queue_relief_lane_ids = {"68604"}
    batch._dead_end_queue_relief_timeout_ticks = 2
    batch._dead_end_queue_relief_progress_epsilon_m = 0.5
    batch._dead_end_queue_relief_stopped_speed_mps = 0.5
    batch._dead_end_queue_relief_trigger_m = 5.0
    batch._dead_end_queue_relief_follower_distance_m = 30.0
    batch._dead_end_queue_progress = {}
    batch._dead_end_queue_retirements = 0

    lead = SimpleNamespace(
        is_active=lambda _step: True,
        has_valid_path=lambda: True,
        get_progress_to_go=lambda: 2.0,
        to_se2=lambda: lead.pose,
        pose=StateSE2(0.0, 0.0, 0.0),
        velocity=0.0,
        end_segment=SimpleNamespace(id="68604", outgoing_edges=()),
    )
    manager = SimpleNamespace(agents={"lead": lead})
    assert not batch._collect_dead_end_queue_retirements(manager, 0)
    assert not batch._dead_end_queue_progress

    follower = SimpleNamespace(
        is_active=lambda _step: True,
        has_valid_path=lambda: True,
        get_progress_to_go=lambda: 10.0,
        to_se2=lambda: StateSE2(-8.0, 0.0, 0.0),
        velocity=0.0,
        end_segment=SimpleNamespace(id="68604", outgoing_edges=()),
    )
    manager.agents["follower"] = follower
    assert not batch._collect_dead_end_queue_retirements(manager, 1)
    lead.pose = StateSE2(0.6, 0.0, 0.0)
    assert not batch._collect_dead_end_queue_retirements(manager, 2)
    assert not batch._collect_dead_end_queue_retirements(manager, 3)
    assert batch._collect_dead_end_queue_retirements(manager, 4) == {"lead"}


def test_dead_end_relief_can_retire_final_vehicle_when_follower_guard_is_disabled():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._dead_end_queue_relief_enabled = True
    batch._dead_end_queue_relief_lane_ids = {"51717"}
    batch._dead_end_queue_relief_timeout_ticks = 2
    batch._dead_end_queue_relief_progress_epsilon_m = 0.5
    batch._dead_end_queue_relief_stopped_speed_mps = 0.5
    batch._dead_end_queue_relief_trigger_m = 5.0
    batch._dead_end_queue_relief_follower_distance_m = 30.0
    batch._dead_end_queue_relief_require_follower = False
    batch._dead_end_queue_progress = {}
    batch._dead_end_queue_retirements = 0

    agent = SimpleNamespace(
        is_active=lambda _step: True,
        has_valid_path=lambda: True,
        get_progress_to_go=lambda: 3.5,
        to_se2=lambda: StateSE2(0.0, 0.0, 0.0),
        velocity=0.0,
        end_segment=SimpleNamespace(id="51717", outgoing_edges=()),
    )
    manager = SimpleNamespace(agents={"last": agent})

    assert not batch._collect_dead_end_queue_retirements(manager, 0)
    assert not batch._collect_dead_end_queue_retirements(manager, 1)
    assert batch._collect_dead_end_queue_retirements(manager, 2) == {"last"}


def test_nuplan_idm_late_admission_requires_emergency_stopping_distance():
    safe = NuPlanIDMBatch._has_safe_stopping_gap

    # 13.39 m/s into a stopped queue needs about 18.3 m before vehicle footprints/margin.
    assert not safe(17.0, 13.39, 0.0, 4.905, 0.1, 1.0)
    assert safe(21.0, 13.39, 0.0, 4.905, 0.1, 1.0)
    # A new leader must also leave enough room for the already-live follower behind it.
    assert not safe(7.0, 10.31, 2.88, 4.905, 0.1, 1.0)
    # Opening or equal-speed traffic only needs the configured static minimum gap.
    assert safe(1.0, 4.0, 6.0, 4.905, 0.1, 1.0)


def _ego_state_for_spawn_clearance(speed=10.0):
    parameters = SimpleNamespace(length=5.0, width=2.0)
    return SimpleNamespace(
        center=SimpleNamespace(x=0.0, y=0.0, heading=0.0),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=speed, y=0.0)
        ),
        car_footprint=SimpleNamespace(vehicle_parameters=parameters),
    )


def _spawn_clearance_batch():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._ego_lane_spawn_clearance_enabled = True
    batch._ego_lane_spawn_reaction_time_s = 1.0
    batch._ego_lane_spawn_stopping_distance_multiplier = 3.0
    batch._ego_lane_spawn_front_min_m = 15.0
    batch._ego_lane_spawn_rear_m = 20.0
    batch._ego_lane_spawn_heading_tolerance_rad = np.deg2rad(30.0)
    batch._ego_lane_spawn_lateral_margin_m = 0.5
    batch._idm_params = {"emergency_decel_max": 4.905}
    return batch


def test_late_spawn_ego_lane_clearance_uses_ego_stopping_distance_ahead():
    batch = _spawn_clearance_batch()
    ego = _ego_state_for_spawn_clearance(speed=10.0)

    # 10 m/s requires (10 * 1 s + 10^2 / (2 * 4.905)) * 3 = 60.6 m.
    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(55.0, 0.0, 0.0, 10.0), ego
    )[0] == "ahead"
    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(70.0, 0.0, 0.0, 10.0), ego
    ) is None


def test_late_spawn_ego_lane_clearance_keeps_twenty_metre_rear_gap():
    batch = _spawn_clearance_batch()
    ego = _ego_state_for_spawn_clearance(speed=0.0)

    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(-24.0, 0.0, 0.0, 5.0), ego
    )[0] == "behind"
    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(-30.0, 0.0, 0.0, 5.0), ego
    ) is None


def test_late_spawn_ego_lane_clearance_excludes_adjacent_and_oncoming_lanes():
    batch = _spawn_clearance_batch()
    ego = _ego_state_for_spawn_clearance(speed=10.0)

    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(20.0, 4.0, 0.0, 10.0), ego
    ) is None
    assert batch._ego_lane_spawn_clearance(
        _moving_idm_agent(20.0, 0.0, np.pi, 10.0), ego
    ) is None


def test_ego_and_idm_share_one_gt_warmup_boundary():
    config = {"num_history": 16, "pdm_gt_warmup_enabled": True}
    assert synchronized_gt_warmup_steps(config) == 15
    assert synchronized_gt_warmup_steps({**config, "pdm_gt_warmup_enabled": False}) == 0

    class _Scenario:
        def __init__(self):
            self.initial_iterations = []

        def set_initial_iteration(self, step):
            self.initial_iterations.append(step)

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._step = -1
    batch._gt_warmup_steps = 15
    batch._idm_start_step = None
    batch._solve_stride = 1
    batch._scenario = _Scenario()
    published = []
    solved = []
    batch._publish_gt_warmup = published.append
    batch._solve_once = lambda step, prev, ego_step: solved.append(
        (step, prev, ego_step)
    )

    batch._advance(0)
    batch._advance(14)
    assert published == [0, 14]
    assert solved == []
    assert batch._scenario.initial_iterations == []

    batch._advance(15)
    batch._advance(16)
    assert batch._scenario.initial_iterations == [15]
    assert solved == [(15, 15, 15), (16, 15, 16)]


def _source_vehicle(x, y, speed=0.0, token="parked", heading=0.0):
    return SimpleNamespace(
        track_token=token,
        center=SimpleNamespace(
            x=x,
            y=y,
            heading=heading,
            point=SimpleNamespace(array=np.asarray([x, y])),
        ),
        velocity=SimpleNamespace(x=speed, y=0.0),
    )


def test_reactive_handoff_gate_accepts_only_near_aligned_rail_associations():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._handoff_smooth_merge_max_offset_m = 3.0
    batch._handoff_smooth_merge_max_heading_deg = 45.0
    rail_agent = _moving_idm_agent(0.0, 0.0, 0.0, 0.0)

    assert batch._handoff_merge_fallback_reason(
        _source_vehicle(3.0, 0.0, heading=np.deg2rad(45.0)), rail_agent
    ) is None
    assert batch._handoff_merge_fallback_reason(
        _source_vehicle(3.01, 0.0), rail_agent
    ).startswith("rail_offset_")
    assert batch._handoff_merge_fallback_reason(
        _source_vehicle(0.0, 0.0, heading=np.deg2rad(45.1)), rail_agent
    ).startswith("rail_heading_")


def test_handoff_merge_is_forward_monotone_and_preserves_the_full_builder_path():
    original_path = create_path_from_se2([
        StateSE2(0.0, 0.0, 0.0),
        StateSE2(5.0, 0.0, 0.0),
        StateSE2(50.0, 0.0, 0.0),
        # nuPlan's helper intentionally uses the final pose as a progress sentinel.
        StateSE2(50.1, 0.0, 0.0),
    ])
    agent = SimpleNamespace(
        _path=original_path,
        progress=0.0,
        _state=SimpleNamespace(progress=0.0),
        _requires_state_update=False,
    )

    assert NuPlanIDMBatch._install_handoff_merge_path(
        _source_vehicle(0.0, 2.8, speed=2.0, heading=0.0), agent
    )

    merged = list(agent._path.get_sampled_path())
    xy = np.asarray([(state.x, state.y) for state in merged])
    headings = np.unwrap(np.asarray([state.heading for state in merged]))
    assert np.all(np.diff(xy[:, 0]) > 0.0)
    assert np.max(np.abs(np.diff(headings))) < np.deg2rad(45.0)
    assert np.isclose(xy[0, 1], 2.8)
    assert np.isclose(xy[-1, 0], 50.0)
    assert agent._state.progress == 0.0
    assert agent._requires_state_update


def test_handoff_merge_rejects_a_backward_initial_tangent_without_mutating_agent():
    original_path = create_path_from_se2([
        StateSE2(0.0, 0.0, 0.0),
        StateSE2(10.0, 0.0, 0.0),
        StateSE2(20.0, 0.0, 0.0),
        StateSE2(20.1, 0.0, 0.0),
    ])
    agent = SimpleNamespace(
        _path=original_path,
        progress=0.0,
        _state=SimpleNamespace(progress=0.0),
        _requires_state_update=False,
    )

    installed = NuPlanIDMBatch._install_handoff_merge_path(
        _source_vehicle(0.0, 2.0, speed=1.0, heading=np.pi), agent
    )

    assert not installed
    assert agent._path is original_path
    assert agent._state.progress == 0.0
    assert not agent._requires_state_update


def test_handoff_merge_accepts_short_rail_without_duplicate_endpoint():
    original_path = create_path_from_se2([
        StateSE2(0.0, 0.0, 0.0),
        StateSE2(3.0, 0.0, 0.0),
        StateSE2(5.0, 0.0, 0.0),
        StateSE2(5.1, 0.0, 0.0),
    ])
    agent = SimpleNamespace(
        _path=original_path,
        progress=0.0,
        _state=SimpleNamespace(progress=0.0),
        _requires_state_update=False,
    )

    assert NuPlanIDMBatch._install_handoff_merge_path(
        _source_vehicle(0.0, 1.0, speed=1.0, heading=0.0), agent
    )
    merged = list(agent._path.get_sampled_path())
    xy = np.asarray([(state.x, state.y) for state in merged])
    assert np.all(np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3)
    assert agent._requires_state_update


def _straight_edge(x0, x1, heading_deg=0.0, curvature=0.0):
    heading = np.deg2rad(heading_deg)
    distances = np.linspace(0.0, x1 - x0, 20)
    return SimpleNamespace(
        outgoing_edges=[],
        baseline_path=SimpleNamespace(
            discrete_path=[
                StateSE2(x0 + d * np.cos(heading), d * np.sin(heading), heading)
                for d in distances
            ],
            get_curvature_at_arc_length=lambda _s: curvature,
        ),
    )


def _rail_end_agent(remaining_m, successors):
    """Builder agent whose route ends ``remaining_m`` ahead."""
    rail = [StateSE2(0.0, 0.0, 0.0), StateSE2(remaining_m, 0.0, 0.0)]
    lane = SimpleNamespace(outgoing_edges=successors)
    original_path = create_path_from_se2(rail + [StateSE2(remaining_m + 0.1, 0.0, 0.0)])
    agent = SimpleNamespace(
        _path=original_path,
        _route=[lane],
        progress=0.0,
        _state=SimpleNamespace(progress=0.0),
        _requires_state_update=False,
        get_path_to_go=lambda: list(rail),
    )
    agent.end_segment = lane
    return agent, original_path


def test_stationary_handoff_at_rail_end_keeps_gt_heading_and_extends_straight():
    straight = _straight_edge(0.45, 30.0)
    turn = _straight_edge(0.45, 30.0, heading_deg=-60.0, curvature=0.2)
    agent, _ = _rail_end_agent(0.45, [turn, straight])
    source = _source_vehicle(0.0, 0.26, speed=0.0, heading=np.deg2rad(1.1))

    assert NuPlanIDMBatch._install_handoff_merge_path(source, agent)

    merged = list(agent._path.get_sampled_path())
    assert np.isclose(merged[0].heading, np.deg2rad(1.1))
    assert np.isclose(merged[0].x, 0.0) and np.isclose(merged[0].y, 0.26)
    headings = np.asarray([state.heading for state in merged])
    assert np.max(np.abs(headings)) < np.deg2rad(10.0)
    assert agent._route[-1] is straight
    assert agent._path.get_end_progress() > 8.0


def test_rejected_rail_end_merge_leaves_route_and_path_untouched():
    agent, original_path = _rail_end_agent(0.45, [_straight_edge(0.45, 30.0)])
    source = _source_vehicle(0.0, 0.26, speed=0.0, heading=np.pi)

    assert not NuPlanIDMBatch._install_handoff_merge_path(source, agent)
    assert agent._path is original_path
    assert len(agent._route) == 1
    assert not agent._requires_state_update


def test_stationary_off_rail_vehicle_stays_open_loop_instead_of_being_snapped():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._parked_vehicle_fallback_enabled = True
    batch._parked_vehicle_speed_threshold_mps = 0.5
    batch._parked_vehicle_snap_distance_m = 2.0
    batch._parked_vehicle_max_track_displacement_m = 3.0
    batch._parked_vehicle_min_valid_samples = 3
    batch._source_vehicle_motion_profiles = {"parked": (20, 0.4)}
    agent = _moving_idm_agent(0.0, 0.0, 0.0, 0.0)
    agent.get_route = lambda: [SimpleNamespace(id="street")]

    assert batch._parked_vehicle_fallback_reason(
        _source_vehicle(0.0, 3.0), agent
    ) == "off_rail_snap"
    # A genuinely queued/stopped lane-centred car must remain reactive IDM traffic.
    assert batch._parked_vehicle_fallback_reason(
        _source_vehicle(0.0, 0.5), agent
    ) is None
    # Geometry, not just low speed, is required; a moving source vehicle remains reactive.
    assert batch._parked_vehicle_fallback_reason(
        _source_vehicle(0.0, 3.0, speed=1.0), agent
    ) is None


def test_lane_centred_stationary_vehicle_remains_reactive_even_on_unstructured_route():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._parked_vehicle_fallback_enabled = True
    batch._parked_vehicle_speed_threshold_mps = 0.5
    batch._parked_vehicle_snap_distance_m = 2.0
    batch._parked_vehicle_max_track_displacement_m = 3.0
    batch._parked_vehicle_min_valid_samples = 3
    batch._source_vehicle_motion_profiles = {"parked": (20, 0.4)}
    agent = _moving_idm_agent(0.0, 0.0, 0.0, 0.0)
    agent.get_route = lambda: [SimpleNamespace(id="unstructured")]

    assert batch._parked_vehicle_fallback_reason(
        _source_vehicle(0.0, 0.0), agent
    ) is None


def test_vehicle_that_moves_elsewhere_in_full_track_is_not_parked_fallback():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._parked_vehicle_fallback_enabled = True
    batch._parked_vehicle_speed_threshold_mps = 0.5
    batch._parked_vehicle_snap_distance_m = 2.0
    batch._parked_vehicle_max_track_displacement_m = 3.0
    batch._parked_vehicle_min_valid_samples = 3
    batch._source_vehicle_motion_profiles = {"parked": (20, 8.0)}
    agent = _moving_idm_agent(0.0, 0.0, 0.0, 0.0)
    agent.get_route = lambda: [SimpleNamespace(id="unstructured")]

    assert batch._parked_vehicle_fallback_reason(
        _source_vehicle(0.0, 0.0), agent
    ) is None


def test_source_vehicle_motion_profile_uses_entire_valid_track():
    scene = {
        "object_track": {
            "car": {
                "type": "VEHICLE",
                "state": {
                    "position": np.asarray([[0.0, 0.0], [5.0, 0.0], [0.1, 0.0]]),
                    "valid": np.asarray([True, True, True]),
                },
            },
            "person": {
                "type": "PEDESTRIAN",
                "state": {
                    "position": np.asarray([[0.0, 0.0], [0.0, 0.0]]),
                    "valid": np.asarray([True, True]),
                },
            },
        }
    }

    assert NuPlanIDMBatch._build_source_vehicle_motion_profiles(scene) == {
        "car": (3, 5.0)
    }


def test_source_motion_gate_keeps_only_subthreshold_tracks_open_loop():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._reactive_min_source_displacement_m = 7.5
    batch._source_vehicle_motion_profiles = {
        "stationary": (20, 0.2),
        "boundary": (20, 7.5),
        "moving": (20, 12.0),
    }

    assert batch._source_motion_open_loop_reason("stationary").startswith(
        "source_motion_0.20m_below_7.50m"
    )
    assert batch._source_motion_open_loop_reason("missing").startswith(
        "source_motion_0.00m_below_7.50m"
    )
    assert batch._source_motion_open_loop_reason("boundary") is None
    assert batch._source_motion_open_loop_reason("moving") is None


def test_only_open_loop_vehicles_are_classified_as_gt_replay():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._source_motion_open_loop_tokens = {"low_motion"}
    batch._obs = SimpleNamespace(
        extra_open_loop_vehicle_tokens={"builder_fallback"}
    )

    assert batch.is_gt_replay_vehicle("low_motion")
    assert batch.is_gt_replay_vehicle("builder_fallback")
    assert not batch.is_gt_replay_vehicle("reactive_idm")


def test_low_motion_gt_vehicle_retries_until_current_spawn_pose_is_free():
    def tracked(token, geometry, kind=TrackedObjectType.VEHICLE):
        center = geometry.centroid
        return SimpleNamespace(
            track_token=token,
            tracked_object_type=kind,
            center=StateSE2(float(center.x), float(center.y), 0.0),
            velocity=SimpleNamespace(x=0.0, y=0.0),
            box=SimpleNamespace(geometry=geometry, length=1.0, width=1.0),
        )

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._source_motion_open_loop_tokens = {"parked"}
    batch._source_motion_spawned_tokens = set()
    batch._source_motion_spawn_deferrals = 0
    batch._idm_params = {
        "emergency_decel_max": 4.905,
        "min_gap_to_lead_agent": 1.0,
    }
    batch._solve_dt = 0.1
    batch._obs = SimpleNamespace(
        extra_open_loop_vehicle_tokens=set(),
        _open_loop_detections_types=[TrackedObjectType.PEDESTRIAN],
    )
    manager = SimpleNamespace(agents={})
    ego = SimpleNamespace(car_footprint=SimpleNamespace(geometry=box(-1, -1, 1, 1)))
    occupied = SimpleNamespace(
        tracked_objects=SimpleNamespace(
            tracked_objects=[tracked("parked", box(-0.5, -0.5, 0.5, 0.5))]
        )
    )

    assert not batch._admit_source_motion_open_loop_vehicles(
        occupied, manager, ego, step=10
    )
    assert batch._source_motion_spawned_tokens == set()
    assert batch._source_motion_spawn_deferrals == 1

    free = SimpleNamespace(
        tracked_objects=SimpleNamespace(
            tracked_objects=[tracked("parked", box(5, 5, 6, 6))]
        )
    )
    assert batch._admit_source_motion_open_loop_vehicles(
        free, manager, ego, step=11
    ) == {"parked"}
    assert batch._source_motion_spawned_tokens == {"parked"}
    assert batch._obs.extra_open_loop_vehicle_tokens == {"parked"}


def test_low_motion_gt_vehicle_retries_until_reactive_follower_can_brake():
    candidate_geometry = box(4.5, -0.5, 5.5, 0.5)
    candidate = SimpleNamespace(
        track_token="parked",
        tracked_object_type=TrackedObjectType.VEHICLE,
        center=StateSE2(5.0, 0.0, 0.0),
        velocity=SimpleNamespace(x=0.0, y=0.0),
        box=SimpleNamespace(
            geometry=candidate_geometry,
            length=1.0,
            width=1.0,
        ),
    )
    follower = SimpleNamespace(
        to_se2=lambda: StateSE2(0.0, 0.0, 0.0),
        velocity=6.0,
        length=4.0,
        width=2.0,
        polygon=box(-2.0, -1.0, 2.0, 1.0),
    )
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._source_motion_open_loop_tokens = {"parked"}
    batch._source_motion_spawned_tokens = set()
    batch._source_motion_spawn_deferrals = 0
    batch._idm_params = {
        "emergency_decel_max": 4.905,
        "min_gap_to_lead_agent": 1.0,
    }
    batch._solve_dt = 0.1
    batch._obs = SimpleNamespace(
        extra_open_loop_vehicle_tokens=set(),
        _open_loop_detections_types=[TrackedObjectType.PEDESTRIAN],
    )
    manager = SimpleNamespace(agents={"follower": follower})
    ego = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(-20.0, -1.0, -16.0, 1.0))
    )
    tracks = SimpleNamespace(
        tracked_objects=SimpleNamespace(tracked_objects=[candidate])
    )

    assert not batch._admit_source_motion_open_loop_vehicles(
        tracks, manager, ego, step=10
    )
    assert batch._source_motion_spawned_tokens == set()
    assert batch._source_motion_spawn_deferrals == 1


def test_late_idm_spawn_checks_braking_gap_to_existing_gt_replay_vehicle():
    candidate = SimpleNamespace(
        to_se2=lambda: StateSE2(0.0, 0.0, 0.0),
        velocity=6.0,
        length=4.0,
        width=2.0,
    )
    replay = SimpleNamespace(
        track_token="parked",
        center=StateSE2(5.0, 0.0, 0.0),
        velocity=SimpleNamespace(x=0.0, y=0.0),
        box=SimpleNamespace(length=4.0, width=2.0),
    )
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._idm_params = {
        "emergency_decel_max": 4.905,
        "min_gap_to_lead_agent": 1.0,
    }
    batch._solve_dt = 0.1
    ego = SimpleNamespace(
        center=StateSE2(-30.0, 0.0, 0.0),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        car_footprint=SimpleNamespace(
            vehicle_parameters=SimpleNamespace(length=4.0, width=2.0)
        ),
    )

    assert batch._unsafe_longitudinal_neighbors(
        candidate,
        ego,
        SimpleNamespace(agents={}),
        [replay],
    ) == ["parked"]


def test_unstable_short_track_filter_requires_short_local_heading_flip():
    scene = {
        "object_track": {
            "corrupt": {
                "type": "VEH",
                "state": {
                    "position": np.asarray([[0.0, 0.0], [0.2, 0.0]]),
                    "heading": np.deg2rad([152.0, 30.0]),
                    "valid": np.asarray([True, True]),
                },
            },
            "stable_short": {
                "type": "VEHICLE",
                "state": {
                    "position": np.asarray([[0.0, 0.0], [0.2, 0.0]]),
                    "heading": np.deg2rad([30.0, 32.0]),
                    "valid": np.asarray([True, True]),
                },
            },
            "real_turn": {
                "type": "VEHICLE",
                "state": {
                    "position": np.asarray([[0.0, 0.0], [4.0, 0.0]]),
                    "heading": np.deg2rad([152.0, 30.0]),
                    "valid": np.asarray([True, True]),
                },
            },
        }
    }

    assert NuPlanIDMBatch._build_unstable_short_track_tokens(scene) == {"corrupt"}


def test_unstable_short_track_filter_uses_pre_upsample_source_samples():
    headings = np.zeros(11)
    headings[0:6] = np.deg2rad(np.linspace(152.0, 30.0, 6))
    positions = np.zeros((11, 2))
    positions[:6, 0] = np.linspace(0.0, 0.2, 6)
    scene = {
        "cadence": SimpleNamespace(upsample_n=5),
        "object_track": {
            "corrupt": {
                "type": "VEHICLE",
                "state": {
                    "position": positions,
                    "heading": headings,
                    "valid": np.asarray([True] * 6 + [False] * 5),
                },
            },
        },
    }

    assert NuPlanIDMBatch._build_unstable_short_track_tokens(scene) == {"corrupt"}


def _unknown_signal_fallback_batch(intersections):
    connectors = {
        connector_id: SimpleNamespace(
            parent=SimpleNamespace(
                intersection=SimpleNamespace(id=intersection_id)
            )
        )
        for connector_id, intersection_id in intersections.items()
    }
    map_api = SimpleNamespace(
        get_map_object=lambda connector_id, _layer: connectors.get(str(connector_id))
    )
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._scenario = SimpleNamespace(map_api=map_api)
    batch._unknown_signal_fallback_enabled = True
    batch._signal_connector_intersection_cache = {}
    batch._unknown_signal_fallback_step = None
    batch._unknown_signal_fallback_connector_ids = frozenset()
    batch._unknown_signal_fallback_active_intersections = frozenset()
    batch._unknown_signal_fallback_events = 0
    batch._unknown_signal_fallback_intersection_ticks = 0
    batch._unknown_signal_fallback_connector_ticks = 0
    return batch


def _warmup_dispatch_batch(mode):
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._initialization_mode = mode
    batch._gt_warmup_steps = 15
    batch._idm_start_step = None
    batch._solve_stride = 1
    batch._step = -1
    batch._scenario = SimpleNamespace(initial_iterations=[])
    batch._scenario.set_initial_iteration = batch._scenario.initial_iterations.append
    batch.solve_calls = []
    batch.warmup_calls = []
    batch._solve_once = lambda step, prev, ego_step: batch.solve_calls.append(
        (step, prev, ego_step)
    )
    batch._publish_gt_warmup = batch.warmup_calls.append
    return batch


def test_gt_merge_replays_background_gt_until_shared_handoff():
    batch = _warmup_dispatch_batch("gt_merge")

    batch._advance(0)

    assert batch.warmup_calls == [0]
    assert batch.solve_calls == []
    assert batch._idm_start_step is None


def test_centerline_snap_warmup_runs_idm_while_ego_remains_on_gt():
    batch = _warmup_dispatch_batch("centerline_snap_warmup")

    batch._advance(0)
    batch._advance(1)

    assert batch.warmup_calls == []
    assert batch._scenario.initial_iterations == [0]
    assert batch._idm_start_step == 0
    assert batch.solve_calls == [(0, 0, 0), (1, 0, 1)]


def _light(status, connector_id):
    return TrafficLightStatusData(
        status=status, lane_connector_id=connector_id, timestamp=0
    )


def test_all_unknown_physical_intersection_falls_back_to_unprotected_rules():
    batch = _unknown_signal_fallback_batch({"a": "junction", "b": "junction"})
    raw = [
        _light(TrafficLightStatusType.UNKNOWN, "a"),
        _light(TrafficLightStatusType.UNKNOWN, "b"),
    ]

    transformed = batch._apply_unknown_signal_unprotected_fallback(10, raw)
    assert {light.status for light in transformed} == {TrafficLightStatusType.GREEN}
    assert batch._unknown_signal_fallback_connector_ids == {"a", "b"}
    assert batch._unknown_signal_fallback_events == 1
    assert batch._unknown_signal_fallback_intersection_ticks == 1
    assert batch._unknown_signal_fallback_connector_ticks == 2

    # The adapter is queried twice at one tick; diagnostics must not double-count it.
    batch._apply_unknown_signal_unprotected_fallback(10, raw)
    assert batch._unknown_signal_fallback_events == 1
    assert batch._unknown_signal_fallback_intersection_ticks == 1


def test_mixed_known_and_unknown_signal_intersection_remains_controlled():
    batch = _unknown_signal_fallback_batch({"a": "junction", "b": "junction"})
    raw = [
        _light(TrafficLightStatusType.UNKNOWN, "a"),
        _light(TrafficLightStatusType.RED, "b"),
    ]

    transformed = batch._apply_unknown_signal_unprotected_fallback(10, raw)
    assert [light.status for light in transformed] == [
        TrafficLightStatusType.UNKNOWN,
        TrafficLightStatusType.RED,
    ]
    assert batch._unknown_signal_fallback_connector_ids == frozenset()
    assert batch._unknown_signal_fallback_events == 0


def test_unknown_intersection_falls_back_despite_known_signal_elsewhere():
    batch = _unknown_signal_fallback_batch({"a": "junction-a", "b": "junction-b"})
    raw = [
        _light(TrafficLightStatusType.UNKNOWN, "a"),
        _light(TrafficLightStatusType.RED, "b"),
    ]

    transformed = batch._apply_unknown_signal_unprotected_fallback(10, raw)

    assert [light.status for light in transformed] == [
        TrafficLightStatusType.GREEN,
        TrafficLightStatusType.RED,
    ]
    assert batch._unknown_signal_fallback_connector_ids == {"a"}
    assert batch._unknown_signal_fallback_events == 1


def test_absent_signal_frame_opens_active_signal_route_frontier_as_unprotected():
    batch = _unknown_signal_fallback_batch({"signal-edge": "junction"})
    batch._scenario.get_traffic_light_status_at_iteration = lambda _step: []
    signal_edge = SimpleNamespace(
        id="signal-edge", has_traffic_lights=lambda: True
    )
    ordinary_edge = SimpleNamespace(
        id="ordinary-edge", has_traffic_lights=lambda: False
    )
    agent = SimpleNamespace(
        is_active=lambda _step: True,
        has_valid_path=lambda: True,
        end_segment=SimpleNamespace(outgoing_edges=[signal_edge, ordinary_edge]),
    )

    status = batch._traffic_light_status(10, [agent])

    assert status[TrafficLightStatusType.GREEN] == ["signal-edge"]
    assert status[TrafficLightStatusType.RED] == []
    assert batch._unknown_signal_fallback_connector_ids == {"signal-edge"}
    assert batch._unknown_signal_fallback_events == 1
    assert batch._unknown_signal_fallback_intersection_ticks == 1
    assert batch._unknown_signal_fallback_connector_ticks == 1

    # Stock IDMAgents asks the adapter for the same empty frame once more.  It must not erase
    # the route-frontier scope or count the same physical intersection twice.
    assert batch._apply_unknown_signal_unprotected_fallback(10, []) == []
    assert batch._unknown_signal_fallback_connector_ids == {"signal-edge"}
    assert batch._unknown_signal_fallback_events == 1
    assert batch._unknown_signal_fallback_intersection_ticks == 1
    assert batch._unknown_signal_fallback_connector_ticks == 1


def _moving_idm_agent(x, y, heading, speed):
    geometry = translate(
        rotate(box(-2.0, -1.0, 2.0, 1.0), heading, use_radians=True),
        xoff=x,
        yoff=y,
    )
    return SimpleNamespace(
        velocity=speed,
        length=4.0,
        width=2.0,
        polygon=geometry,
        to_se2=lambda: SimpleNamespace(x=x, y=y, heading=heading),
    )


def _moving_idm_agent_with_path(x, y, heading, speed, path):
    agent = _moving_idm_agent(x, y, heading, speed)
    agent.get_path_to_go = lambda: [
        SimpleNamespace(x=px, y=py, heading=ph) for px, py, ph in path
    ]
    return agent


def test_late_spawn_crossing_guard_uses_simultaneous_current_velocity_slices_only():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_crossing_horizon_s = 1.0
    batch._spawn_crossing_sample_dt = 0.1
    candidate = _moving_idm_agent(-4.0, 0.0, 0.0, 4.0)
    crossing = _moving_idm_agent(0.0, -4.0, np.pi / 2.0, 4.0)

    unsafe = batch._unsafe_crossing_spawn_neighbors(
        candidate, SimpleNamespace(agents={"crossing": crossing})
    )

    assert unsafe == ["crossing"]


def test_late_spawn_crossing_guard_ignores_parallel_and_time_separated_motion():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_crossing_horizon_s = 1.0
    batch._spawn_crossing_sample_dt = 0.1
    candidate = _moving_idm_agent(-4.0, 0.0, 0.0, 4.0)
    parallel = _moving_idm_agent(-4.0, 5.0, 0.0, 4.0)
    # This actor reaches the spatial crossing point after the candidate has passed it.
    late_crossing = _moving_idm_agent(0.0, -8.0, np.pi / 2.0, 4.0)

    unsafe = batch._unsafe_crossing_spawn_neighbors(
        candidate,
        SimpleNamespace(agents={"parallel": parallel, "late": late_crossing}),
    )

    assert unsafe == []


def test_late_spawn_crossing_guard_reuses_live_prediction_cache():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_crossing_horizon_s = 1.0
    batch._spawn_crossing_sample_dt = 0.1
    crossing = _moving_idm_agent(0.0, -4.0, np.pi / 2.0, 4.0)
    manager = SimpleNamespace(agents={"crossing": crossing})
    cache = {}

    first = batch._unsafe_crossing_spawn_neighbors(
        _moving_idm_agent(-4.0, 0.0, 0.0, 4.0), manager, cache
    )
    cached_boxes = cache["crossing"][1]
    second = batch._unsafe_crossing_spawn_neighbors(
        _moving_idm_agent(-5.0, 0.0, 0.0, 5.0), manager, cache
    )

    assert first == ["crossing"]
    assert second == ["crossing"]
    assert cache["crossing"][1] is cached_boxes


def test_late_spawn_rail_guard_defers_adjacent_paths_that_merge_downstream():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_rail_conflict_lookahead_m = 100.0
    candidate = _moving_idm_agent_with_path(
        0.0, 0.0, 0.0, 8.0,
        [(0.0, 0.0, 0.0), (40.0, 0.0, 0.0), (80.0, 0.0, 0.0)],
    )
    merging = _moving_idm_agent_with_path(
        0.0, 4.0, 0.0, 5.0,
        [(0.0, 4.0, 0.0), (35.0, 4.0, 0.0), (50.0, 0.0, 0.0)],
    )

    unsafe = batch._unsafe_converging_rail_spawn_neighbors(
        candidate, SimpleNamespace(agents={"merging": merging})
    )

    assert unsafe == ["merging"]


def test_late_spawn_rail_guard_leaves_parallel_lane_to_normal_idm():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_rail_conflict_lookahead_m = 100.0
    candidate = _moving_idm_agent_with_path(
        0.0, 0.0, 0.0, 8.0,
        [(0.0, 0.0, 0.0), (80.0, 0.0, 0.0)],
    )
    parallel = _moving_idm_agent_with_path(
        0.0, 4.0, 0.0, 5.0,
        [(0.0, 4.0, 0.0), (80.0, 4.0, 0.0)],
    )

    unsafe = batch._unsafe_converging_rail_spawn_neighbors(
        candidate, SimpleNamespace(agents={"parallel": parallel})
    )

    assert unsafe == []


def test_late_spawn_rail_guard_reuses_live_buffer_cache():
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._spawn_rail_conflict_lookahead_m = 100.0
    merging = _moving_idm_agent_with_path(
        0.0, 4.0, 0.0, 5.0,
        [(0.0, 4.0, 0.0), (35.0, 4.0, 0.0), (50.0, 0.0, 0.0)],
    )
    manager = SimpleNamespace(agents={"merging": merging})
    cache = {}
    candidates = [
        _moving_idm_agent_with_path(
            0.0, offset, 0.0, 8.0,
            [(0.0, offset, 0.0), (40.0, offset, 0.0), (80.0, 0.0, 0.0)],
        )
        for offset in (0.0, 0.5)
    ]

    first = batch._unsafe_converging_rail_spawn_neighbors(candidates[0], manager, cache)
    cached_rail = cache["merging"][2]
    second = batch._unsafe_converging_rail_spawn_neighbors(candidates[1], manager, cache)

    assert first == ["merging"]
    assert second == ["merging"]
    assert cache["merging"][2] is cached_rail


def test_pdm_observation_covers_reference_eight_second_trajectory():
    observation = PDMObservation(
        trajectory_sampling=TrajectorySampling(num_poses=80, interval_length=0.1),
        proposal_sampling=TrajectorySampling(num_poses=40, interval_length=0.1),
        map_radius=50,
    )

    assert observation._observation_sample_res == 2
    assert observation._observation_samples == 80
    assert len(observation._global_to_local_idcs) == 82


def test_pdm_trajectory_keeps_native_ten_hz_spacing():
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.initial_ego_center = np.zeros(2)

    velocity = SimpleNamespace(x=3.0, y=4.0)
    states = [
        SimpleNamespace(waypoint=SimpleNamespace(
            x=float(i), y=0.0, velocity=velocity, heading=0.0))
        for i in range(3)
    ]
    trajectory = converter.convert_to_trajectory(
        SimpleNamespace(_trajectory=states), wp_dt=SAMPLE_INTERVAL)

    assert trajectory.wp_dt == 0.1
    assert np.allclose(trajectory.waypoints[:, 0], [0.0, 1.0, 2.0])
    assert np.allclose(trajectory.velocities, [[3.0, 4.0]] * 3)






def test_source_frame_conversion_treats_sample_rate_as_lidar_period_count():
    scene_2hz = {"sample_rate": 10}
    assert source_frame_at_elapsed_time(scene_2hz, 0.49) == 0
    assert source_frame_at_elapsed_time(scene_2hz, 0.5) == 1
    assert source_frame_at_elapsed_time(scene_2hz, 47.5) == 95

    scene_10hz = {"sample_rate": 2}
    assert source_frame_at_elapsed_time(scene_10hz, 0.5) == 5


class _FakeBaseline:
    def __init__(self, heading, lateral_distance=0.0):
        self._heading = heading
        self._lateral_distance = lateral_distance

    def get_nearest_pose_from_position(self, point):
        return SimpleNamespace(
            x=float(point.x), y=float(point.y) + self._lateral_distance,
            heading=self._heading,
        )


class _FakeLane:
    def __init__(self, lane_id, roadblock_id, heading):
        self.id = lane_id
        self._roadblock_id = roadblock_id
        self.baseline_path = _FakeBaseline(heading)
        self.outgoing_edges = []

    def get_roadblock_id(self):
        return self._roadblock_id


def test_route_map_matching_keeps_lane_and_follows_graph_successor():
    along = _FakeLane("along", "road-a", 0.0)
    crossing = _FakeLane("crossing", "wrong-crossing", np.pi / 2)
    overlapping = _FakeLane("overlapping", "wrong-overlap", 0.0)
    successor = _FakeLane("successor", "road-b", 0.0)
    decoy = _FakeLane("decoy", "wrong-decoy", 0.3)
    overlapping.outgoing_edges = [successor]

    candidates = {
        0: [crossing, along],       # heading selects along, not API list order
        1: [crossing, along],       # continuity keeps along in overlap
        2: [overlapping],           # disconnected overlap must not enter the route
        3: [overlapping],           # even when it persists for more than one sample
        4: [decoy, successor],      # overlap's successor restores the real route graph
    }
    roadblocks = {
        "road-a": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-b")]),
        "road-b": SimpleNamespace(outgoing_edges=[]),
        "wrong-overlap": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-b")]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            # Return candidates through one layer only, as a real map query would.
            return candidates[int(round(point.x))] if str(layer).endswith("LANE") else []

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        np.zeros(2),
        _FakeMap(),
    )

    assert ids == ["road-a", "road-b"]


def test_route_map_matching_does_not_seed_continuity_from_disconnected_decoy():
    along = _FakeLane("along", "road-a", 0.0)
    successor = _FakeLane("successor", "road-b", 0.0)
    wrong = _FakeLane("wrong", "wrong-a", 0.0)
    wrong_successor = _FakeLane("wrong-successor", "wrong-b", 0.0)
    along.outgoing_edges = [successor]
    wrong.outgoing_edges = [wrong_successor]

    candidates = {
        0: [along],
        # A map overlap temporarily exposes only an unrelated movement.
        1: [wrong],
        # If the decoy became the continuity seed, its successor would beat the real lane.
        2: [wrong_successor, successor],
    }
    roadblocks = {
        "road-a": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-b")]),
        "road-b": SimpleNamespace(outgoing_edges=[]),
        "wrong-a": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="wrong-b")]),
        "wrong-b": SimpleNamespace(outgoing_edges=[]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            return candidates[int(round(point.x))] if str(layer).endswith("LANE") else []

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.zeros(2),
        _FakeMap(),
    )

    assert ids == ["road-a", "road-b"]


def test_route_map_matching_backtracks_an_overlapping_wrong_sibling():
    entry = _FakeLane("entry", "road-entry", 0.0)
    wrong = _FakeLane("wrong", "road-wrong", 0.0)
    correct = _FakeLane("correct", "road-correct", 0.0)
    exit_lane = _FakeLane("exit", "road-exit", 0.0)
    entry.outgoing_edges = [wrong, correct]
    correct.outgoing_edges = [exit_lane]

    # At the first overlapping pose the wrong connector is marginally better aligned.  At the
    # next pose the logged heading exposes the correct sibling, which must replace it rather than
    # permanently truncating the route.
    wrong.baseline_path._heading = 0.0
    correct.baseline_path._heading = 0.95
    candidates = {
        0: [entry],
        1: [wrong, correct],
        2: [wrong, correct],
        3: [exit_lane],
    }
    roadblocks = {
        "road-entry": SimpleNamespace(
            outgoing_edges=[SimpleNamespace(id="road-wrong"), SimpleNamespace(id="road-correct")]
        ),
        "road-wrong": SimpleNamespace(outgoing_edges=[]),
        "road-correct": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-exit")]),
        "road-exit": SimpleNamespace(outgoing_edges=[]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            return candidates[int(round(point.x))] if str(layer).endswith("LANE") else []

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    # Centred headings at the overlapping samples are about 0.46 and 0.98 rad.  This favours
    # wrong first, then correct.
    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 3.0]]),
        np.zeros(2),
        _FakeMap(),
    )

    assert ids == ["road-entry", "road-correct", "road-exit"]


def test_route_map_matching_bridges_an_unobserved_short_connector():
    entry = _FakeLane("entry", "road-entry", 0.0)
    exit_lane = _FakeLane("exit", "road-exit", 0.0)
    roadblocks = {
        "road-entry": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-tiny")]),
        "road-tiny": SimpleNamespace(outgoing_edges=[SimpleNamespace(id="road-exit")]),
        "road-exit": SimpleNamespace(outgoing_edges=[]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            if not str(layer).endswith("LANE"):
                return []
            return [entry] if point.x < 1.0 else [exit_lane]

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [2.0, 0.0]]), np.zeros(2), _FakeMap()
    )

    assert ids == ["road-entry", "road-tiny", "road-exit"]


def test_route_map_matching_rejects_containing_reverse_lane_for_nearby_forward_rail():
    reverse = _FakeLane("reverse", "road-reverse", np.pi)
    forward = _FakeLane("forward", "road-forward", 0.0)
    forward.baseline_path._lateral_distance = 2.0
    roadblocks = {
        "road-reverse": SimpleNamespace(outgoing_edges=[]),
        "road-forward": SimpleNamespace(outgoing_edges=[]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            return [reverse] if str(layer).endswith("LANE") else []

        def get_proximal_map_objects(self, point, radius, layers):
            return {layers[0]: [reverse, forward], layers[1]: []}

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [1.0, 0.0]]), np.zeros(2), _FakeMap()
    )

    assert ids == ["road-forward"]


def test_route_map_matching_keeps_aligned_gt_segment_across_broken_map_graph_seam():
    before = _FakeLane("before", "road-before", 0.0)
    after = _FakeLane("after", "road-after", 0.0)
    roadblocks = {
        "road-before": SimpleNamespace(outgoing_edges=[]),
        "road-after": SimpleNamespace(outgoing_edges=[]),
    }

    class _FakeMap:
        def get_all_map_objects(self, point, layer):
            if not str(layer).endswith("LANE"):
                return []
            return [before] if point.x < 5.0 else [after]

        def get_map_object(self, object_id, layer):
            return roadblocks.get(str(object_id)) if str(layer).endswith("ROADBLOCK") else None

    ids = route_roadblock_ids(
        np.array([[0.0, 0.0], [10.0, 0.0]]), np.zeros(2), _FakeMap()
    )

    assert ids == ["road-before", "road-after"]


def test_pdm_preserves_gt_matched_route_endpoint_during_correction():
    planner = PDMClosedPlanner.__new__(PDMClosedPlanner)
    planner._map_api = object()
    planner._route_roadblock_dict = {"first": object(), "last": object()}
    planner._route_is_gt_matched = True
    planner._load_route_dicts = Mock()
    ego_state = object()

    target = (
        "odyssey.components.agents.policy.pdm_planner.abstract_pdm_planner."
        "route_roadblock_correction"
    )
    with patch(target, return_value=["first", "last"]) as correction:
        planner._route_roadblock_correction(ego_state)

    correction.assert_called_once_with(
        ego_state,
        planner._map_api,
        planner._route_roadblock_dict,
        cut_route_loops=False,
    )
    planner._load_route_dicts.assert_called_once_with(["first", "last"])


def test_pdm_builds_full_centerline_for_long_gt_matched_route():
    planner = PDMClosedPlanner.__new__(PDMClosedPlanner)
    blocks = [SimpleNamespace(id=str(index)) for index in range(41)]
    planner._route_roadblock_dict = {block.id: block for block in blocks}
    planner._route_lane_dict = {}
    planner._route_is_gt_matched = True
    current_lane = SimpleNamespace(id="lane-0", get_roadblock_id=lambda: "0")
    path_lane = SimpleNamespace(
        id="path-lane",
        baseline_path=SimpleNamespace(discrete_path=[StateSE2(0.0, 0.0, 0.0)]),
    )

    target = (
        "odyssey.components.agents.policy.pdm_planner.abstract_pdm_planner.Dijkstra"
    )
    with patch(target) as dijkstra:
        dijkstra.return_value.search.return_value = ([path_lane], True)
        path = planner._get_discrete_centerline(current_lane)

    dijkstra.return_value.search.assert_called_once_with(blocks[-1])
    assert path == [StateSE2(0.0, 0.0, 0.0)]
    assert planner._last_centerline_target_roadblock_id == "40"


def test_pdm_long_route_uses_endpoint_reachable_starting_lane_sibling():
    planner = PDMClosedPlanner.__new__(PDMClosedPlanner)
    dead_lane = SimpleNamespace(
        id="dead", get_roadblock_id=lambda: "start",
        baseline_path=SimpleNamespace(linestring=box(0, 0, 1, 1)),
    )
    through_lane = SimpleNamespace(
        id="through", get_roadblock_id=lambda: "start",
        baseline_path=SimpleNamespace(
            linestring=box(0, 1, 1, 2),
            discrete_path=[StateSE2(0.0, 1.0, 0.0)],
        ),
    )
    start_block = SimpleNamespace(id="start", interior_edges=[dead_lane, through_lane])
    target_block = SimpleNamespace(id="target", interior_edges=[])
    planner._route_roadblock_dict = {"start": start_block, "target": target_block}
    planner._route_lane_dict = {"dead": dead_lane, "through": through_lane}
    planner._route_is_gt_matched = True

    target = (
        "odyssey.components.agents.policy.pdm_planner.abstract_pdm_planner.Dijkstra"
    )
    with patch(target) as dijkstra:
        dijkstra.return_value.search.side_effect = [([dead_lane], False),
                                                    ([through_lane], True)]
        path = planner._get_discrete_centerline(dead_lane)

    assert path == [StateSE2(0.0, 1.0, 0.0)]
    assert planner._last_centerline_path_found is True
    assert planner._last_centerline_start_lane_id == "through"


class _Config(dict):
    def __getattr__(self, name):
        return self[name]


class _FakeAgent:
    def __init__(self, policy, position=(0.0, 0.0)):
        self.policy = policy
        self.navigation = None
        self.current_position = np.asarray(position, dtype=float)
        self.stepped = False
        self.destroyed = False

    def step(self):
        self.stepped = True

    def destroy(self):
        self.destroyed = True


def test_ego_progress_lifecycle_clock_applies_only_to_idm_vehicles(monkeypatch):
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._idm_lifecycle_clock = object()
    manager._idm_source_step = 17
    engine = SimpleNamespace(
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        current_scene={
            SD.OBJECT_TRACKS: {
                "vehicle": {"type": "VEHICLE"},
                "pedestrian": {"type": "PEDESTRIAN"},
            }
        },
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    replay_rows = {"vehicle": 91.0, "pedestrian": 23.0}
    assert manager._presence_step("vehicle", 80, replay_rows) == 17
    assert manager._presence_step("pedestrian", 80, replay_rows) == 23

    manager._idm_lifecycle_clock = None
    assert manager._presence_step("vehicle", 80, replay_rows) == 91


def test_intersection_sector_clock_opens_idm_gate_and_publishes_shared_rows(monkeypatch):
    from odyssey.scenario.hybrid_replay import SectorReplay

    route = np.stack([np.arange(101, dtype=float), np.zeros(101)], axis=1)
    actor_xy = np.stack([80.0 + np.arange(101) * 0.1, np.zeros(101)], axis=1)
    clock = SectorReplay(
        route,
        {"later": (np.arange(101), actor_xy)},
        sector_len_s=4.0,
        lead_m=5.0,
        src_dt=0.1,
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._dynamic_agents = {"ego": ego}
    manager._idm_spawn_clock = clock
    manager._idm_spawn_eligible = set()
    engine = SimpleNamespace(episode_step=0)
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager._update_idm_spawn_eligibility(0)
    assert manager.is_idm_spawn_eligible("later") is False
    assert manager._reactive_sector_rows["later"] == -1

    ego.current_position = np.asarray([75.0, 0.0])
    manager._update_idm_spawn_eligibility(1)
    assert manager.is_idm_spawn_eligible("later") is True
    assert manager._reactive_sector_rows["later"] >= 0

    # Eligibility is a one-way spawn door. Rewinding the observed ego projection must not
    # evict an already admitted IDM actor or close its retry path.
    ego.current_position = np.asarray([10.0, 0.0])
    manager._update_idm_spawn_eligibility(2)
    assert manager.is_idm_spawn_eligible("later") is True


def test_reactive_sector_row_controls_vulnerable_and_idm_fallback_presence(monkeypatch):
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._idm_spawn_clock = object()
    manager._reactive_sector_rows = {"ped": 31.0, "bike": 32.0, "rigid": 33.0}
    manager._idm_lifecycle_clock = object()
    manager._idm_source_step = 99
    engine = SimpleNamespace(
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        current_scene={SD.OBJECT_TRACKS: {
            "ped": {"type": "PEDESTRIAN"},
            "bike": {"type": "BICYCLE"},
            "rigid": {"type": "VEHICLE"},
        }},
        _nuplan_idm_batch=SimpleNamespace(
            is_gt_replay_vehicle=lambda token: token == "rigid"),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    assert manager._presence_step("ped", 80, {}) == 31
    assert manager._presence_step("bike", 80, {}) == 32
    assert manager._presence_step("rigid", 80, {}) == 33
    assert manager.object_source_row("rigid", 80) == 33


def test_ineligible_initial_idm_actor_is_removed_without_being_retired():
    class _Occupancy:
        def __init__(self, tokens):
            self.tokens = set(tokens)

        def contains(self, token):
            return token in self.tokens

        def remove(self, tokens):
            self.tokens.difference_update(tokens)

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch.engine = SimpleNamespace(
        agent_manager=SimpleNamespace(
            is_idm_spawn_eligible=lambda token: token == "eligible"
        )
    )
    batch._spawn_sector_waiting_tokens = set()
    occupancy = _Occupancy({"eligible", "later"})
    idm_manager = SimpleNamespace(
        agents={"eligible": object(), "later": object()},
        agent_occupancy=occupancy,
    )

    batch._remove_ineligible_initial_agents(idm_manager, 0)

    assert set(idm_manager.agents) == {"eligible"}
    assert occupancy.tokens == {"eligible"}
    assert batch._spawn_sector_waiting_tokens == {"later"}


def test_nuplan_idm_proxy_persists_inside_source_end_safety_radius(monkeypatch):
    removed = []
    ended_policy = SimpleNamespace(remove_from_batch=lambda: removed.append("traffic"))
    ended = _FakeAgent(ended_policy, position=(50.0, 0.0))
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))

    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "traffic": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "traffic": [(0, 1)]}
    scene = {"log_length": 10}
    fake_batch = SimpleNamespace(
        scene=scene,
        prepare_step=lambda step: None,
        can_materialize=lambda token: True,
    )
    engine = SimpleNamespace(
        episode_step=2,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=fake_batch,
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert manager._dynamic_agents["traffic"] is ended
    assert not ended.destroyed
    assert ended.stepped
    assert removed == []
    assert ego.stepped


def test_progress_clock_does_not_start_50m_retention_before_progress_source_end(
    monkeypatch,
):
    removed = []
    traffic = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("traffic")),
        position=(100.0, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._idm_lifecycle_clock = SimpleNamespace(source_row=lambda _xy: 1.0)
    manager._idm_source_step = 0
    manager._dynamic_agents = {"ego": ego, "traffic": traffic}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "traffic": [(0, 1)]}
    scene = {"log_length": 200}
    engine = SimpleNamespace(
        episode_step=100,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None,
            can_materialize=lambda token: True,
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert manager.idm_source_step == 1
    assert manager._dynamic_agents["traffic"] is traffic
    assert traffic.stepped
    assert removed == []


def test_nuplan_idm_source_end_liveness_retires_stalled_proxy_inside_radius(monkeypatch):
    removed = []
    ended = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("traffic")),
        position=(10.0, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "traffic": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 20)], "traffic": [(0, 1)]}
    manager._source_end_progress = {}
    scene = {"log_length": 20}
    engine = SimpleNamespace(
        episode_step=2,
        sim_dt=0.1,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
            nuplan_idm_source_end_liveness_enabled=True,
            nuplan_idm_source_end_stall_timeout_s=1.0,
            nuplan_idm_source_end_progress_epsilon_m=0.5,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None, can_materialize=lambda token: True
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()
    assert manager._dynamic_agents["traffic"] is ended
    assert manager.source_end_liveness_stats == {
        "stall_retirements": 0,
        "retained_agent_ticks": 1,
    }

    engine.episode_step = 12
    manager.step()

    assert "traffic" not in manager._dynamic_agents
    assert ended.destroyed
    assert removed == ["traffic"]
    assert manager.source_end_liveness_stats == {
        "stall_retirements": 1,
        "retained_agent_ticks": 1,
    }


def test_nuplan_idm_source_end_liveness_resets_after_progress(monkeypatch):
    removed = []
    ended = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("traffic")),
        position=(10.0, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "traffic": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 20)], "traffic": [(0, 1)]}
    manager._source_end_progress = {}
    scene = {"log_length": 20}
    engine = SimpleNamespace(
        episode_step=2,
        sim_dt=0.1,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
            nuplan_idm_source_end_liveness_enabled=True,
            nuplan_idm_source_end_stall_timeout_s=1.0,
            nuplan_idm_source_end_progress_epsilon_m=0.5,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None, can_materialize=lambda token: True
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()
    ended.current_position = np.array([10.6, 0.0])
    engine.episode_step = 11
    manager.step()
    engine.episode_step = 12
    manager.step()

    assert manager._dynamic_agents["traffic"] is ended
    assert not ended.destroyed
    assert removed == []


def test_nuplan_idm_proxy_is_retired_outside_source_end_safety_radius(monkeypatch):
    removed = []
    ended = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("traffic")),
        position=(50.01, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "traffic": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "traffic": [(0, 1)]}
    scene = {"log_length": 10}
    engine = SimpleNamespace(
        episode_step=2,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None, can_materialize=lambda token: True
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "traffic" not in manager._dynamic_agents
    assert ended.destroyed
    assert not ended.stepped
    assert removed == ["traffic"]
    assert ego.stepped


def test_nuplan_non_vehicle_fallback_despawns_at_source_end_inside_radius(monkeypatch):
    removed = []
    ended = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("pedestrian")),
        position=(1.0, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "pedestrian": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "pedestrian": [(0, 1)]}
    scene = {"log_length": 10}
    engine = SimpleNamespace(
        episode_step=2,
        current_scene={SD.OBJECT_TRACKS: {"pedestrian": {"type": "PEDESTRIAN"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None, can_materialize=lambda token: True
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "pedestrian" not in manager._dynamic_agents
    assert ended.destroyed
    assert not ended.stepped
    assert removed == ["pedestrian"]
    assert ego.stepped


def test_nuplan_gt_replay_vehicle_does_not_use_idm_distance_retention(monkeypatch):
    removed = []
    ended = _FakeAgent(
        SimpleNamespace(remove_from_batch=lambda: removed.append("parked")),
        position=(1.0, 0.0),
    )
    ego = _FakeAgent(SimpleNamespace(), position=(0.0, 0.0))
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "parked": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "parked": [(0, 1)]}
    scene = {"log_length": 10}
    engine = SimpleNamespace(
        episode_step=2,
        current_scene={SD.OBJECT_TRACKS: {"parked": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_source_end_keep_radius=50.0,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene,
            prepare_step=lambda step: None,
            can_materialize=lambda token: True,
            is_gt_replay_vehicle=lambda token: token == "parked",
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "parked" not in manager._dynamic_agents
    assert ended.destroyed
    assert not ended.stepped
    assert removed == ["parked"]
    assert ego.stepped


def test_nuplan_idm_source_end_despawn_can_be_reenabled(monkeypatch):
    removed = []
    ended = _FakeAgent(SimpleNamespace(remove_from_batch=lambda: removed.append("traffic")))
    ego = _FakeAgent(SimpleNamespace())
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "traffic": ended}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ego": [(0, 10)], "traffic": [(0, 1)]}
    scene = {"log_length": 10}
    engine = SimpleNamespace(
        episode_step=2,
        current_scene={SD.OBJECT_TRACKS: {"traffic": {"type": "VEHICLE"}}},
        global_config=_Config(
            agent_policy="nuplan_idm_policy",
            nuplan_idm_despawn_on_source_end=True,
        ),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=SimpleNamespace(
            scene=scene, prepare_step=lambda step: None, can_materialize=lambda token: True
        ),
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "traffic" not in manager._dynamic_agents
    assert ended.destroyed
    assert not ended.stepped
    assert removed == ["traffic"]


def test_nuplan_idm_removed_vehicle_is_retired_but_deferred_candidate_is_not():
    class _Occupancy:
        def __init__(self, tokens):
            self.tokens = set(tokens)

        def contains(self, token):
            return token in self.tokens

        def remove(self, tokens):
            self.tokens.difference_update(tokens)

    manager = SimpleNamespace(
        agents={"appeared": object()},
        agent_occupancy=_Occupancy({"appeared"}),
    )
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._obs = SimpleNamespace(
        _idm_agent_manager=manager,
        _get_idm_agent_manager=lambda: manager,
    )
    batch._idm_start_step = 15
    batch._ever_admitted = {"appeared"}
    batch._retired = set()
    batch._poses = {"appeared": object()}
    batch._source_modes = {"appeared": "idm"}

    batch.remove_agent("appeared")
    batch.remove_agent("never-admitted")

    assert batch._retired == {"appeared"}
    assert "appeared" not in manager.agents
    assert "appeared" not in manager.agent_occupancy.tokens


def test_nuplan_idm_proxy_waits_until_batch_admission(monkeypatch):
    order = []
    scene = {"log_length": 10}
    batch = SimpleNamespace(
        scene=scene,
        prepare_step=lambda step: order.append(("prepare", step)),
        can_materialize=lambda token: token != "pending",
    )
    ego = _FakeAgent(SimpleNamespace())
    admitted = _FakeAgent(SimpleNamespace())
    ego.step = lambda: order.append(("step", "ego"))
    admitted.step = lambda: order.append(("step", "admitted"))

    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego, "admitted": admitted}
    manager._static_agents = {}
    manager._agent_valid_periods = {
        "ego": [(0, 10)], "admitted": [(0, 10)], "pending": [(0, 10)]
    }
    engine = SimpleNamespace(
        episode_step=2,
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=batch,
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "pending" not in manager._dynamic_agents
    assert order == [("step", "ego"), ("step", "admitted"), ("prepare", 2)]


def test_nuplan_idm_removes_unmaterializable_static_proxy(monkeypatch):
    scene = {"log_length": 10}
    batch = SimpleNamespace(
        scene=scene, prepare_step=lambda step: None,
        can_materialize=lambda token: token != "refused",
    )
    ego = _FakeAgent(SimpleNamespace())
    refused = _FakeAgent(SimpleNamespace())
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {"ego": ego}
    manager._static_agents = {"refused": refused}
    manager._agent_valid_periods = {"ego": [(0, 10)], "refused": [(0, 10)]}
    engine = SimpleNamespace(
        episode_step=2,
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=batch,
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)

    manager.step()

    assert "refused" not in manager._static_agents
    assert refused.destroyed


def test_nuplan_idm_admission_occupancy_matches_reference_open_loop_types():
    class _Occupancy:
        def __init__(self):
            self.tokens = set()

        def contains(self, token):
            return token in self.tokens

        def insert(self, token, geometry):
            self.tokens.add(token)

    def obj(token, kind=TrackedObjectType.VEHICLE):
        return SimpleNamespace(
            track_token=token, tracked_object_type=kind,
            box=SimpleNamespace(geometry=box(0, 0, 1, 1)),
        )

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._obs = SimpleNamespace(_open_loop_detections_types=[
        TrackedObjectType[name] for name in OPEN_LOOP_DETECTION_TYPES
    ])
    occupancy = _Occupancy()
    manager = SimpleNamespace(agent_occupancy=occupancy)
    tracks = SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=[
        obj("vehicle"),
        obj("pedestrian", TrackedObjectType.PEDESTRIAN),
        obj("barrier", TrackedObjectType.BARRIER),
    ]))

    inserted = batch._insert_open_loop_for_admission(tracks, manager)

    assert inserted == ["pedestrian", "barrier"]
    assert occupancy.tokens == {"pedestrian", "barrier"}
    assert "vehicle" not in occupancy.tokens


def test_nuplan_idm_controls_even_a_source_static_vehicle():
    assert _nuplan_idm_controls_vehicle(
        _Config(agent_policy="nuplan_idm_policy"), {"type": "VEHICLE"})
    assert not _nuplan_idm_controls_vehicle(
        _Config(agent_policy="trajectory_policy"), {"type": "VEHICLE"})
    assert not _nuplan_idm_controls_vehicle(
        _Config(agent_policy="nuplan_idm_policy"), {"type": "TRAFFIC_CONE"})


def test_nuplan_idm_multirate_interpolation_wraps_heading_and_defers_spawn():
    start = {
        "existing": (0.0, 0.0, np.deg2rad(179.0), 2.0),
        "leaving": (3.0, 4.0, 0.0, 0.0),
    }
    target = {
        "existing": (10.0, 4.0, np.deg2rad(-179.0), 4.0),
        "spawning": (20.0, 0.0, 0.0, 1.0),
    }

    halfway = NuPlanIDMBatch._interpolate_idm_poses(start, target, 0.5)
    assert set(halfway) == {"existing", "leaving"}
    assert np.allclose(halfway["existing"][:2], [5.0, 2.0])
    assert np.isclose(abs(halfway["existing"][2]), np.pi)
    assert np.isclose(halfway["existing"][3], 3.0)

    boundary = NuPlanIDMBatch._interpolate_idm_poses(start, target, 1.0)
    assert set(boundary) == {"existing", "spawning"}


def test_nuplan_scenario_adapter_caches_repeated_source_frame_conversion():
    calls = []
    converter = SimpleNamespace(
        convert_to_detections_tracks_from_scene=lambda iteration: calls.append(iteration) or object()
    )
    from odyssey.components.agents.policy.nuplan_idm_policy import _ScenarioAdapter
    scenario = _ScenarioAdapter(converter, map_api=object())

    first = scenario.get_tracked_objects_at_iteration(7)
    second = scenario.get_tracked_objects_at_iteration(7)

    assert first is second
    assert calls == [7]


def test_scene_observation_active_index_preserves_gaps_order_and_clamping():
    def track(positions):
        return {"state": {"position": np.asarray(positions, dtype=float)}}

    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.scene = {
        "sdc_id": "ego",
        "object_track": {
            "first": track([[1, 0, 0], [0, 0, 0], [2, 0, 0]]),
            "ego": track([[1, 1, 0], [1, 1, 0], [1, 1, 0]]),
            "short_live": track([[3, 0, 0], [4, 0, 0]]),
            "short_dead": track([[5, 0, 0], [0, 0, 0]]),
            "late": track([[0, 0, 0], [6, 0, 0], [7, 0, 0]]),
        },
    }
    converter._scene_track_ids = None
    converter._scene_track_valid = None

    ids_at = lambda step: [token for token, _ in converter._scene_track_items_at(step)]

    assert ids_at(0) == ["first", "short_live", "short_dead"]
    assert ids_at(1) == ["short_live", "late"]
    assert ids_at(2) == ["first", "short_live", "late"]
    # Existing conversion clamps each short track independently to its last source row.
    assert ids_at(99) == ["first", "short_live", "late"]
    assert "ego" not in ids_at(0)


def test_scene_observation_uses_per_object_sector_source_row():
    def track(positions):
        return {"state": {"position": np.asarray(positions, dtype=float)}}

    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.scene = {
        "sdc_id": "ego",
        "object_track": {
            "ped": track([[0, 0, 0], [4, 0, 0], [0, 0, 0]]),
            "ego": track([[1, 1, 0]] * 3),
        },
    }
    converter.engine = SimpleNamespace(managers={
        "agent_manager": SimpleNamespace(
            object_source_row=lambda token, step: 1 if token == "ped" else step)
    })
    converter._scene_track_ids = None
    converter._scene_track_valid = None

    assert [token for token, _ in converter._scene_track_items_at(2)] == ["ped"]


def test_late_nuplan_proxy_starts_at_batch_pose_not_raw_source_pose():
    class _Proxy:
        def __init__(self):
            self.position = np.array([100.0, 100.0])
            self.heading = 1.0
            self.velocity = np.array([0.0, 0.0])
            self.angular_velocity = 2.0
            self.reset_count = 0

        def set_position(self, value):
            self.position = np.asarray(value)

        def set_heading_theta(self, value):
            self.heading = value

        def set_velocity(self, value):
            self.velocity = np.asarray(value)

        def set_angular_velocity(self, value):
            self.angular_velocity = value

        def reset(self):
            self.reset_count += 1

    idm_proxy = _Proxy()
    open_loop_proxy = _Proxy()
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._dynamic_agents = {"idm": idm_proxy}
    manager._static_agents = {"open_loop": open_loop_proxy}
    batch = SimpleNamespace(
        source_mode_for=lambda token: "idm" if token == "idm" else "open_loop",
        pose_for=lambda step, token: (1.0, 2.0, np.pi / 2, 3.0),
    )

    manager._sync_proxy_to_nuplan_pose(batch, "idm", 42)
    manager._sync_proxy_to_nuplan_pose(batch, "open_loop", 42)

    assert np.allclose(idm_proxy.position, [1.0, 2.0])
    assert np.isclose(idm_proxy.heading, np.pi / 2)
    assert np.allclose(idm_proxy.velocity, [0.0, 3.0])
    assert idm_proxy.angular_velocity == 0.0
    assert idm_proxy.reset_count == 1
    assert np.allclose(open_loop_proxy.position, [100.0, 100.0])
    assert open_loop_proxy.reset_count == 0


def test_post_propagation_guard_rejects_only_overlapping_new_agent():
    class _Occupancy:
        def __init__(self):
            self.tokens = {"existing", "overlap", "safe", "ego"}

        def contains(self, token):
            return token in self.tokens

        def remove(self, tokens):
            self.tokens.difference_update(tokens)

    def tracked(token, geometry):
        return SimpleNamespace(track_token=token, box=SimpleNamespace(geometry=geometry))

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._deferred_tokens = set()
    batch._idm_initial_speed_records = {}
    manager = SimpleNamespace(
        agents={"existing": object(), "overlap": object(), "safe": object()},
        agent_occupancy=_Occupancy(),
    )
    tracks = SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=[
        tracked("existing", box(0, 0, 2, 2)),
        tracked("overlap", box(1, 0, 3, 2)),
        tracked("safe", box(10, 0, 12, 2)),
    ]))
    ego_state = SimpleNamespace(car_footprint=SimpleNamespace(geometry=box(-10, 0, -8, 2)))

    rejected = batch._reject_overlapping_new_agents(
        {"overlap", "safe"}, tracks, ego_state, manager, step=42)

    assert rejected == {"overlap"}
    assert set(manager.agents) == {"existing", "safe"}
    assert manager.agent_occupancy.tokens == {"existing", "safe", "ego"}
    assert batch._deferred_tokens == {"overlap"}


def test_ordered_projection_preserves_p_turn_without_jumping_branches():
    # The route intentionally returns close to its first straight after a large loop.  The
    # later branch is spatially closer to this query, so raw LineString.project jumps forward.
    # Ordered projection must stay on the current branch; once progress reaches the return leg,
    # the exact same geometry must remain available rather than being cut as a "loop".
    states = [
        StateSE2(0.0, 0.0, 0.0),
        StateSE2(10.0, 0.0, 0.0),
        StateSE2(20.0, 0.0, 0.0),
        StateSE2(20.0, 10.0, np.pi / 2),
        StateSE2(0.0, 10.0, np.pi),
        StateSE2(0.0, 1.0, -np.pi / 2),
        StateSE2(10.0, 1.0, 0.0),
    ]
    path = PDMPath(states)
    query = Point(5.0, 0.6)

    raw_progress = float(path.project(query))
    outbound_progress = path.project_near(
        query, reference_progress=5.0, heading=0.0
    )
    return_progress = path.project_near(
        query, reference_progress=64.0, heading=0.0
    )

    assert raw_progress > 50.0
    assert np.isclose(outbound_progress, 5.0)
    assert return_progress > 50.0
    assert path.length > return_progress  # the intentional P-turn is still present
