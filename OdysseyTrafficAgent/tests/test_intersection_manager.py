from dataclasses import fields
from collections import defaultdict
from math import pi
from types import SimpleNamespace

import pytest
from shapely.geometry import LineString, box
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from odyssey.components.agents.policy.intersection_manager import (
    EgoSafetyState,
    IntersectionManager,
    IntersectionManagerConfig,
    IntersectionWorldState,
    TargetedLeadIDMAgentManager,
)


class _Occupancy:
    def __init__(self, geometries=None):
        self.geometries = dict(geometries or {})

    def get_all_ids(self):
        return list(self.geometries)

    def get(self, token):
        return self.geometries[token]

    def set(self, token, geometry):
        self.geometries[token] = geometry

    def contains(self, token):
        return token in self.geometries

    def remove(self, tokens):
        for token in tokens:
            del self.geometries[token]

    def insert(self, token, geometry):
        self.set(token, geometry)

    def intersects(self, geometry):
        return _Occupancy({
            token: candidate for token, candidate in self.geometries.items()
            if candidate.intersects(geometry)
        })

    @property
    def size(self):
        return len(self.geometries)

    def get_nearest_entry_to(self, token):
        own = self.geometries[token]
        candidates = {
            other: geometry for other, geometry in self.geometries.items() if other != token
        }
        nearest = min(candidates, key=lambda other: own.distance(candidates[other]))
        return nearest, candidates[nearest], own.distance(candidates[nearest])


class _Agent:
    def __init__(self, token, position, path_end, intersection, velocity=2.0, heading=0.0):
        self.token = token
        self.position = tuple(position)
        self.path_end = tuple(path_end)
        self.velocity = velocity
        self.heading = heading
        self.width = 2.0
        self.length = 4.0
        parent = SimpleNamespace(intersection=intersection)
        self._route = [SimpleNamespace(id=f"{token}-segment", parent=parent)]
        self.propagated_leads = []

    def has_valid_path(self):
        return True

    def get_route(self):
        return self._route

    def get_path_to_go(self):
        return [
            SimpleNamespace(x=self.position[0], y=self.position[1]),
            SimpleNamespace(x=self.path_end[0], y=self.path_end[1]),
        ]

    def to_se2(self):
        return SimpleNamespace(
            x=self.position[0], y=self.position[1], heading=self.heading
        )

    def is_active(self, _iteration):
        return True

    def plan_route(self, _traffic_light_status):
        pass

    def get_progress_to_go(self):
        return self.path_end[0] - self.position[0]

    def propagate(self, lead, _tspan):
        self.propagated_leads.append(lead)

    @property
    def projected_footprint(self):
        return self.polygon

    @property
    def polygon(self):
        x, y = self.position
        return box(x - self.length / 2, y - self.width / 2,
                   x + self.length / 2, y + self.width / 2)


def _intersection():
    return SimpleNamespace(id="junction-1", polygon=box(-2.0, -2.0, 2.0, 2.0))


def _ego(geometry=None, velocity=(0.0, 0.0)):
    return EgoSafetyState(
        footprint=geometry or box(100.0, 100.0, 102.0, 102.0),
        velocity_x=velocity[0],
        velocity_y=velocity[1],
    )


def _manager(**overrides):
    values = dict(sim_dt=0.1, decel_max=2.0, log_interval_ticks=0)
    values.update(overrides)
    return IntersectionManager(IntersectionManagerConfig(**values))


def _crossing_agents(intersection=None):
    intersection = intersection or _intersection()
    return {
        "a": _Agent("a", (-6.0, 0.0), (12.0, 0.0), intersection),
        "b": _Agent("b", (0.0, -6.0), (0.0, 12.0), intersection),
    }


def _world(
    tick,
    agents,
    extra=None,
    controlled_lane_connector_ids=frozenset(),
    green_lane_connector_ids=frozenset(),
):
    occupancy = {token: agent.polygon for token, agent in agents.items()}
    occupancy.update(extra or {})
    return IntersectionWorldState(
        tick=tick,
        occupancy=_Occupancy(occupancy),
        controlled_lane_connector_ids=frozenset(controlled_lane_connector_ids),
        green_lane_connector_ids=frozenset(green_lane_connector_ids),
    )


def test_v1_grants_exactly_one_candidate_and_injects_only_the_held_gate():
    manager = _manager()
    agents = _crossing_agents()
    world = _world(10, agents)

    decisions = manager.update(world, agents, _ego())

    assert decisions["a"].enter
    assert decisions["a"].reason is None
    assert not decisions["b"].enter
    assert decisions["b"].reason == "fifo_wait"

    leads = manager.targeted_virtual_leads(decisions)
    assert list(leads) == ["b"]
    assert world.occupancy.geometries == {token: agent.polygon for token, agent in agents.items()}


def test_overlapping_sibling_intersection_ids_share_physical_movement_holders():
    manager = _manager()
    first_parent = SimpleNamespace(id="junction-a", polygon=box(-2.0, -2.0, 2.0, 2.0))
    second_parent = SimpleNamespace(id="junction-b", polygon=box(-2.0, -2.0, 2.0, 2.0))
    agents = {
        "a": _Agent("a", (-6.0, 0.0), (12.0, 0.0), first_parent),
        "b": _Agent("b", (0.0, -6.0), (0.0, 12.0), second_parent),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert sum(decision.enter for decision in decisions.values()) == 1
    assert {decision.reason for decision in decisions.values()} == {None, "active_npc"}


def test_v2_grants_disjoint_buffered_movements_in_parallel():
    manager = _manager()
    intersection = SimpleNamespace(
        id="junction-wide", polygon=box(-5.0, -5.0, 5.0, 5.0)
    )
    agents = {
        "a": _Agent("a", (-8.0, -2.0), (20.0, -2.0), intersection),
        "b": _Agent("b", (-8.0, 2.0), (20.0, 2.0), intersection),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert decisions["b"].enter
    assert manager.states["junction-wide"].granted_npc_ids == {"a", "b"}
    assert manager.targeted_virtual_leads(decisions) == {}


def test_v1_compatibility_switch_serializes_even_disjoint_movements():
    manager = _manager(allow_non_conflicting_movements=False)
    intersection = SimpleNamespace(
        id="junction-wide", polygon=box(-5.0, -5.0, 5.0, 5.0)
    )
    agents = {
        "a": _Agent("a", (-8.0, -2.0), (20.0, -2.0), intersection),
        "b": _Agent("b", (-8.0, 2.0), (20.0, 2.0), intersection),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert not decisions["b"].enter
    assert decisions["b"].reason == "fifo_wait"


def test_active_npc_holds_others_until_its_rear_has_left_the_junction():
    manager = _manager()
    agents = _crossing_agents()
    manager.update(_world(10, agents), agents, _ego())

    agents["a"].position = (0.0, 0.0)
    inside = manager.update(_world(11, agents), agents, _ego())
    assert inside["b"].reason == "active_npc"
    assert manager.states["junction-1"].active_npc_id == "a"

    # At x=5 the four-metre vehicle's rear is x=3, completely beyond the x=2 exit.
    agents["a"].position = (5.0, 0.0)
    released = manager.update(_world(12, agents), agents, _ego())
    assert released["b"].enter
    assert manager.states["junction-1"].active_npc_id is None
    assert manager.states["junction-1"].granted_npc_id == "b"


def test_dont_block_the_box_holds_when_downstream_vehicle_occupies_storage():
    manager = _manager()
    agents = {"a": _crossing_agents()["a"]}
    downstream_vehicle = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(10, agents, {"downstream": downstream_vehicle}), agents, _ego()
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"


def test_adaptive_shield_enforces_dont_block_box_for_lone_entry():
    manager = _manager(contention_only=True)
    agents = {"a": _crossing_agents()["a"]}
    downstream_vehicle = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(10, agents, {"downstream": downstream_vehicle}), agents, _ego()
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"
    assert list(manager.targeted_virtual_leads(decisions)) == ["a"]
    assert manager.states["junction-1"].granted_npc_ids == set()


def test_adaptive_shield_leaves_a_lone_empty_entry_on_stock_idm():
    manager = _manager(contention_only=True)
    agents = {"a": _crossing_agents()["a"]}

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.states["junction-1"].granted_npc_ids == set()
    assert manager.get_stats() == {
        "updates": 1,
        "hold_reason_ticks": {
            "active_npc": 0,
            "downstream_blocked": 0,
            "ego_safety_envelope": 0,
            "fifo_wait": 0,
        },
        "stall_events": 0,
        "preexisting_active_stall_events": 0,
        "active_stall_retirements": 0,
        "downstream_reservation_conflicts": 0,
        "downstream_reservation_ttl_revocations": 0,
        "peak_downstream_reservations": 0,
        "intersection_stopped_seconds": 0.0,
        "candidate_ticks": 1,
        "contention_candidate_ticks": 0,
        "safety_candidate_ticks": 0,
        "unmanaged_candidate_ticks": 1,
        "downstream_moving_same_flow_ignored_ticks": 0,
        "downstream_transient_stop_ignored_ticks": 0,
        "downstream_persistent_blocker_ticks": 0,
        "downstream_cross_direction_blocker_ticks": 0,
        "downstream_insufficient_path_ticks": 0,
        "downstream_unclassified_blocker_ticks": 0,
        "rear_ego_hold_suppressions": 0,
    }


def test_adaptive_shield_arms_for_lone_ego_conflict():
    manager = _manager(contention_only=True)
    agents = {"a": _crossing_agents()["a"]}

    decisions = manager.update(
        _world(10, agents), agents, _ego(box(-0.5, -0.5, 0.5, 0.5))
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "ego_safety_envelope"
    assert list(manager.targeted_virtual_leads(decisions)) == ["a"]
    assert manager.get_stats()["safety_candidate_ticks"] == 1


def test_adaptive_shield_enforces_downstream_storage_during_npc_contention():
    manager = _manager(contention_only=True)
    agents = _crossing_agents()
    downstream_vehicle = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(10, agents, {"downstream": downstream_vehicle}), agents, _ego()
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"
    assert decisions["b"].enter


def test_adaptive_shield_leaves_same_direction_following_to_stock_idm():
    manager = _manager(contention_only=True)
    intersection = SimpleNamespace(
        id="junction-wide", polygon=box(-5.0, -5.0, 5.0, 5.0)
    )
    agents = {
        "a": _Agent("a", (-7.1, 0.0), (20.0, 0.0), intersection, velocity=2.0),
        "b": _Agent("b", (-12.0, 0.0), (20.0, 0.0), intersection, velocity=4.0),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert decisions["b"].enter
    assert manager.states["junction-wide"].granted_npc_ids == set()
    assert manager.targeted_virtual_leads(decisions) == {}


def test_downstream_storage_counts_current_ego_footprint_as_occupied():
    manager = _manager()
    agents = {"a": _crossing_agents()["a"]}
    downstream_ego = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(10, agents, {"ego": downstream_ego}),
        agents,
        _ego(downstream_ego),
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"


def test_downstream_storage_uses_current_npc_body_not_idm_projected_footprint():
    manager = _manager()
    candidate = _crossing_agents()["a"]
    passage = manager._next_passage("a", candidate)
    blocker = _Agent("b", (20.0, 0.0), (30.0, 0.0), _intersection())
    # The IDM occupancy rail extends into the receiving slot, while the actual
    # vehicle body is already well beyond it.
    occupancy = _Occupancy({
        "a": candidate.polygon,
        "b": box(3.0, -1.0, 7.0, 1.0),
    })

    assert manager.has_downstream_storage(
        passage, occupancy, "a", {"a": candidate, "b": blocker}
    )


def test_moving_same_flow_downstream_npc_is_left_to_stock_idm():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=1.0
    )
    agents = {"a": candidate, "b": blocker}

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["downstream_moving_same_flow_ignored_ticks"] == 1


def test_same_flow_classification_uses_route_direction_after_a_turn():
    manager = _manager(contention_only=True)
    intersection = _intersection()
    candidate = _Agent("a", (-6.0, 0.0), (2.0, 20.0), intersection)
    candidate.get_path_to_go = lambda: [
        SimpleNamespace(x=-6.0, y=0.0),
        SimpleNamespace(x=0.0, y=0.0),
        SimpleNamespace(x=2.0, y=0.0),
        SimpleNamespace(x=2.0, y=20.0),
    ]
    blocker = _Agent(
        "b",
        (2.0, 5.0),
        (2.0, 20.0),
        intersection,
        velocity=1.0,
        heading=pi / 2,
    )
    agents = {"a": candidate, "b": blocker}

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert manager.get_stats()["downstream_moving_same_flow_ignored_ticks"] == 1
    assert manager.get_stats()["downstream_cross_direction_blocker_ticks"] == 0


def test_transient_same_flow_stop_does_not_arm_downstream_hold():
    manager = _manager(contention_only=True, downstream_stopped_persistence_s=2.0)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=0.0
    )
    agents = {"a": candidate, "b": blocker}

    for tick in range(10, 29):
        decisions = manager.update(_world(tick, agents), agents, _ego())
        assert decisions["a"].enter

    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["downstream_transient_stop_ignored_ticks"] == 19


def test_persistent_same_flow_stop_arms_downstream_hold_after_two_seconds():
    manager = _manager(contention_only=True, downstream_stopped_persistence_s=2.0)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=0.0
    )
    agents = {"a": candidate, "b": blocker}

    for tick in range(10, 30):
        decisions = manager.update(_world(tick, agents), agents, _ego())

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"
    assert list(manager.targeted_virtual_leads(decisions)) == ["a"]
    assert manager.get_stats()["downstream_persistent_blocker_ticks"] == 1


def test_moving_or_leaving_downstream_corridor_resets_stop_persistence():
    manager = _manager(contention_only=True, downstream_stopped_persistence_s=2.0)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=0.0
    )
    agents = {"a": candidate, "b": blocker}
    for tick in range(10, 29):
        manager.update(_world(tick, agents), agents, _ego())

    blocker.velocity = 1.0
    moving = manager.update(_world(29, agents), agents, _ego())
    blocker.velocity = 0.0
    restarted = manager.update(_world(30, agents), agents, _ego())

    assert moving["a"].enter
    assert restarted["a"].enter
    assert manager._downstream_stopped_first_tick[("junction-1", "a", "b")] == 30

    blocker.position = (20.0, 0.0)
    manager.update(_world(31, agents), agents, _ego())
    assert manager._downstream_stopped_first_tick == {}


def test_cross_direction_downstream_npc_is_an_immediate_physical_blocker():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b",
        (5.0, 0.0),
        (5.0, 20.0),
        _intersection(),
        velocity=2.0,
        heading=pi / 2,
    )
    agents = {"a": candidate, "b": blocker}

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"
    assert manager.get_stats()["downstream_cross_direction_blocker_ticks"] == 1


def test_rear_ego_envelope_suppresses_new_persistent_downstream_virtual_stop():
    manager = _manager(contention_only=True, downstream_stopped_persistence_s=2.0)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=0.0
    )
    agents = {"a": candidate, "b": blocker}
    rear_ego = _ego(box(-13.0, -1.0, -9.0, 1.0), velocity=(8.0, 0.0))

    for tick in range(10, 30):
        decisions = manager.update(_world(tick, agents), agents, rear_ego)

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["rear_ego_hold_suppressions"] == 1


def test_stopped_ego_directly_behind_suppresses_only_virtual_downstream_hold():
    manager = _manager(contention_only=True, downstream_stopped_persistence_s=0.1)
    candidate = _crossing_agents()["a"]
    blocker = _Agent(
        "b", (5.0, 0.0), (20.0, 0.0), _intersection(), velocity=0.0
    )
    agents = {"a": candidate, "b": blocker}
    stopped_rear_ego = _ego(box(-13.0, -1.0, -9.0, 1.0), velocity=(0.0, 0.0))

    decisions = manager.update(_world(10, agents), agents, stopped_rear_ego)

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["rear_ego_hold_suppressions"] == 1


def test_cross_direction_ego_envelope_is_not_suppressed_as_rear_following():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    agents = {"a": candidate}
    crossing_ego = _ego(box(-0.5, -4.0, 0.5, -2.0), velocity=(0.0, 4.0))

    decisions = manager.update(_world(10, agents), agents, crossing_ego)

    assert not decisions["a"].enter
    assert decisions["a"].reason == "ego_safety_envelope"


def test_f1281_like_rear_ego_and_short_downstream_rail_gets_no_virtual_stop():
    manager = _manager(contention_only=True)
    candidate = _Agent("a", (-6.0, 0.0), (5.0, 0.0), _intersection())
    agents = {"a": candidate}
    rear_ego = _ego(box(-13.0, -1.0, -9.0, 1.0), velocity=(8.0, 0.0))

    decisions = manager.update(_world(0, agents), agents, rear_ego)

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["downstream_insufficient_path_ticks"] == 1


def test_short_downstream_rail_is_diagnostic_only_without_rear_ego():
    manager = _manager(contention_only=True)
    candidate = _Agent("a", (-6.0, 0.0), (5.0, 0.0), _intersection())
    agents = {"a": candidate}

    decisions = manager.update(_world(0, agents), agents, _ego())

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["downstream_insufficient_path_ticks"] == 1


def test_rear_ego_suppresses_unclassified_downstream_physical_hold():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    agents = {"a": candidate}
    rear_ego = _ego(box(-13.0, -1.0, -9.0, 1.0), velocity=(8.0, 0.0))
    unclassified_blocker = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(0, agents, {"unknown": unclassified_blocker}), agents, rear_ego
    )

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}
    assert manager.get_stats()["downstream_unclassified_blocker_ticks"] == 1
    assert manager.get_stats()["rear_ego_hold_suppressions"] == 1


def test_cross_traffic_ego_hold_wins_over_unclassified_downstream_geometry():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    agents = {"a": candidate}
    crossing_ego = _ego(box(-0.5, -4.0, 0.5, -2.0), velocity=(0.0, 4.0))
    unclassified_blocker = box(3.0, -1.0, 7.0, 1.0)

    decisions = manager.update(
        _world(0, agents, {"unknown": unclassified_blocker}),
        agents,
        crossing_ego,
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "ego_safety_envelope"
    assert manager.get_stats()["rear_ego_hold_suppressions"] == 0


def test_same_direction_ego_ahead_remains_an_immediate_entry_blocker():
    manager = _manager(contention_only=True)
    candidate = _crossing_agents()["a"]
    agents = {"a": candidate}
    ego_ahead = _ego(box(-0.5, -0.5, 0.5, 0.5), velocity=(2.0, 0.0))

    decisions = manager.update(_world(10, agents), agents, ego_ahead)

    assert not decisions["a"].enter
    assert decisions["a"].reason == "ego_safety_envelope"


def test_pre_entry_grant_is_rechecked_when_downstream_becomes_blocked():
    manager = _manager()
    agents = _crossing_agents()
    manager.update(_world(10, agents), agents, _ego())

    downstream_vehicle = box(3.0, -1.0, 7.0, 1.0)
    decisions = manager.update(
        _world(11, agents, {"downstream": downstream_vehicle}), agents, _ego()
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "downstream_blocked"
    assert decisions["b"].enter


def test_same_flow_downstream_reservation_is_left_to_stock_idm():
    manager = _manager()
    first = _intersection()
    second = SimpleNamespace(id="junction-2", polygon=first.polygon)
    agents = {
        "a": _Agent("a", (-6.0, 0.0), (20.0, 0.0), first),
        "b": _Agent("b", (-6.0, 0.0), (20.0, 0.0), second),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert decisions["b"].enter
    assert manager.get_stats()["downstream_reservation_conflicts"] == 0
    assert manager.get_stats()["peak_downstream_reservations"] == 2


def test_provisional_reservation_waits_until_grant_is_within_commit_distance():
    manager = _manager(reservation_commit_distance_m=2.0)
    intersection = _intersection()
    agent = _Agent("a", (-10.0, 0.0), (20.0, 0.0), intersection, velocity=4.0)

    manager.update(_world(10, {"a": agent}), {"a": agent}, _ego())
    assert manager.states["junction-1"].granted_npc_ids == {"a"}
    assert manager._downstream_reservations == {}

    agent.position = (-4.0, 0.0)
    manager.update(_world(11, {"a": agent}), {"a": agent}, _ego())
    assert set(manager._downstream_reservations) == {("junction-1", "a")}


def test_reservation_ttl_revokes_only_claim_and_keeps_holder_permit():
    manager = _manager(reservation_ttl_s=0.2)
    agents = _crossing_agents()
    manager.update(_world(10, agents), agents, _ego())
    assert ("junction-1", "a") in manager._downstream_reservations

    manager.update(_world(12, agents), agents, _ego())

    assert manager.states["junction-1"].granted_npc_ids == {"a"}
    assert ("junction-1", "a") not in manager._downstream_reservations
    assert manager.get_stats()["downstream_reservation_ttl_revocations"] == 1

    # Expiring the advisory downstream claim must not weaken physical serialization.
    agents["a"].position = (0.0, 0.0)
    decisions = manager.update(_world(13, agents), agents, _ego())
    assert manager.states["junction-1"].active_npc_ids == {"a"}
    assert decisions["b"].reason == "active_npc"


def test_recovered_active_stall_is_classified_without_forcing_recovery():
    manager = _manager(stall_timeout_s=0.2)
    intersection = _intersection()
    agent = _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)

    manager.update(_world(0, {"a": agent}), {"a": agent}, _ego())
    manager.update(_world(2, {"a": agent}), {"a": agent}, _ego())

    assert manager.states["junction-1"].active_npc_ids == {"a"}
    assert manager.get_stats()["preexisting_active_stall_events"] == 1
    assert manager.consume_active_stall_retirements() == set()


def test_opt_in_active_stall_liveness_releases_holder_for_atomic_retirement():
    manager = _manager(stall_timeout_s=0.2, retire_active_stalls=True)
    intersection = _intersection()
    agent = _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)

    manager.update(_world(0, {"a": agent}), {"a": agent}, _ego())
    manager.update(_world(2, {"a": agent}), {"a": agent}, _ego())

    assert manager.consume_active_stall_retirements() == {"a"}
    assert manager.states["junction-1"].active_npc_ids == set()
    assert manager.states["junction-1"].arrival_tick == {}
    assert manager.consume_active_stall_retirements() == set()
    assert manager.get_stats()["active_stall_retirements"] == 1


def test_active_stall_retirement_waits_until_caller_marks_holder_eligible():
    manager = _manager(stall_timeout_s=0.2, retire_active_stalls=True)
    intersection = _intersection()
    agent = _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)

    manager.update(_world(0, {"a": agent}), {"a": agent}, _ego())
    manager.update(_world(2, {"a": agent}), {"a": agent}, _ego())

    # The source track is still alive, so the batch supplies an empty eligible set.
    assert manager.consume_active_stall_retirements(set()) == set()
    assert manager.states["junction-1"].active_npc_ids == {"a"}
    assert manager.get_stats()["active_stall_retirements"] == 0

    # Once the current source frame no longer contains the token, consume atomically releases it.
    assert manager.consume_active_stall_retirements({"a"}) == {"a"}
    assert manager.states["junction-1"].active_npc_ids == set()
    assert manager.get_stats()["active_stall_retirements"] == 1


def test_conflicting_downstream_slot_is_reserved_across_distinct_intersections():
    manager = _manager(downstream_lateral_margin_m=2.0)
    first = _intersection()
    second = SimpleNamespace(id="junction-2", polygon=first.polygon)
    agents = {
        "a": _Agent("a", (-6.0, 0.0), (20.0, 0.0), first),
        "b": _Agent(
            "b", (0.0, -6.0), (0.0, 20.0), second, heading=pi / 2
        ),
    }

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert decisions["b"].reason == "downstream_blocked"
    assert manager.get_stats()["downstream_reservation_conflicts"] == 1
    assert manager.get_stats()["peak_downstream_reservations"] == 1


def test_downstream_reservation_is_released_after_rear_clears_junction():
    manager = _manager(downstream_lateral_margin_m=2.0)
    first = _intersection()
    second = SimpleNamespace(id="junction-2", polygon=first.polygon)
    agents = {
        "a": _Agent("a", (-6.0, 0.0), (20.0, 0.0), first),
        "b": _Agent(
            "b", (0.0, -6.0), (0.0, 20.0), second, heading=pi / 2
        ),
    }
    manager.update(_world(10, agents), agents, _ego())
    agents["a"].position = (0.0, 0.0)
    manager.update(_world(11, agents), agents, _ego())
    agents["a"].position = (12.0, 0.0)

    decisions = manager.update(_world(12, agents), agents, _ego())

    assert decisions["b"].enter


def test_granted_npc_that_crosses_junction_between_ticks_releases_reservation():
    manager = _manager()
    agents = {"a": _crossing_agents()["a"]}
    manager.update(_world(10, agents), agents, _ego())

    # A coarse/fast step can move the complete footprint from before to beyond the
    # junction without producing an intermediate polygon intersection.
    agents["a"].position = (6.0, 0.0)
    manager.update(_world(11, agents), agents, _ego())

    state = manager.states["junction-1"]
    assert not state.active_npc_ids
    assert not state.granted_npc_ids
    assert manager._downstream_reservations == {}


def test_current_state_ego_envelope_holds_only_a_conflicting_entry():
    manager = _manager()
    agents = {"a": _crossing_agents()["a"]}

    decisions = manager.update(
        _world(10, agents), agents, _ego(box(-0.5, -0.5, 0.5, 0.5))
    )

    assert not decisions["a"].enter
    assert decisions["a"].reason == "ego_safety_envelope"


def test_empty_route_intersection_is_a_strict_noop():
    manager = _manager()
    no_intersection = SimpleNamespace(id="unused", polygon=box(-2, -2, 2, 2))
    agent = _Agent("a", (-6.0, 0.0), (12.0, 0.0), no_intersection)
    agent._route = [SimpleNamespace(parent=SimpleNamespace(intersection=None))]
    agents = {"a": agent}
    world = _world(10, agents)
    before = dict(world.occupancy.geometries)

    decisions = manager.update(world, agents, _ego())
    leads = manager.targeted_virtual_leads(decisions)

    assert decisions == {}
    assert leads == {}
    assert world.occupancy.geometries == before


def test_single_npc_at_empty_intersection_gets_no_virtual_lead():
    manager = _manager()
    agents = {"a": _crossing_agents()["a"]}

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert decisions["a"].enter
    assert manager.targeted_virtual_leads(decisions) == {}


def test_future_inactive_npc_is_not_an_admission_candidate():
    manager = _manager()
    agents = _crossing_agents()
    agents["a"].is_active = lambda _tick: False

    decisions = manager.update(_world(10, agents), agents, _ego())

    assert set(decisions) == {"b"}
    assert decisions["b"].enter


def test_signal_controlled_intersection_is_left_entirely_to_stock_idm():
    manager = _manager()
    agents = _crossing_agents()
    agents["a"]._route[0].id = "controlled-connector"
    before = {token: agent.polygon for token, agent in agents.items()}
    world = _world(10, agents, controlled_lane_connector_ids={"controlled-connector"})

    decisions = manager.update(world, agents, _ego())

    assert decisions == {}
    assert manager.targeted_virtual_leads(decisions) == {}
    assert world.occupancy.geometries == before

    # Signal control is tick-local. If the adapter removes an all-UNKNOWN intersection from
    # the controlled set on a later tick, its movements must enter ordinary FIFO arbitration.
    decisions = manager.update(_world(11, agents), agents, _ego())
    assert set(decisions) == {"a", "b"}
    assert sum(decision.enter for decision in decisions.values()) == 1
    assert {decision.reason for decision in decisions.values()} == {None, "fifo_wait"}


def test_signal_conflict_shield_serializes_only_simultaneous_green_movements():
    manager = _manager(protect_signal_conflicts=True)
    agents = _crossing_agents()
    agents["a"]._route[0].id = "green-a"
    agents["b"]._route[0].id = "green-b"
    world = _world(
        10,
        agents,
        controlled_lane_connector_ids={"green-a", "green-b"},
        green_lane_connector_ids={"green-a", "green-b"},
    )

    decisions = manager.update(world, agents, _ego())

    assert decisions["a"].enter
    assert decisions["b"].reason == "fifo_wait"


def test_signal_conflict_shield_leaves_red_movement_to_stock_stop_line():
    manager = _manager(protect_signal_conflicts=True)
    agents = _crossing_agents()
    agents["a"]._route[0].id = "red-a"
    agents["b"]._route[0].id = "green-b"
    world = _world(
        10,
        agents,
        controlled_lane_connector_ids={"red-a", "green-b"},
        green_lane_connector_ids={"green-b"},
    )

    decisions = manager.update(world, agents, _ego())

    assert "a" not in decisions
    assert decisions["b"].enter


def test_red_vehicle_overlapping_broad_junction_is_not_recovered_as_active():
    manager = _manager(protect_signal_conflicts=True)
    agents = _crossing_agents()
    agents["a"].position = (0.0, 0.0)
    agents["a"]._route[0].id = "red-a"
    agents["b"]._route[0].id = "green-b"
    world = _world(
        10,
        agents,
        controlled_lane_connector_ids={"red-a", "green-b"},
        green_lane_connector_ids={"green-b"},
    )

    decisions = manager.update(world, agents, _ego())

    assert "a" not in decisions
    assert decisions["b"].enter
    assert not manager.states["junction-1"].active_npc_ids


def test_green_vehicle_already_inside_junction_is_recovered_as_active():
    manager = _manager(protect_signal_conflicts=True)
    agents = _crossing_agents()
    agents["a"].position = (0.0, 0.0)
    agents["a"]._route[0].id = "green-a"
    agents["b"]._route[0].id = "green-b"
    world = _world(
        10,
        agents,
        controlled_lane_connector_ids={"green-a", "green-b"},
        green_lane_connector_ids={"green-a", "green-b"},
    )

    decisions = manager.update(world, agents, _ego())

    assert manager.states["junction-1"].active_npc_ids == {"a"}
    assert decisions["b"].reason == "active_npc"


def test_ego_api_cannot_carry_route_future_or_planner_output():
    assert {item.name for item in fields(EgoSafetyState)} == {
        "footprint", "velocity_x", "velocity_y"
    }
    manager = _manager()
    agents = _crossing_agents()

    with pytest.raises(TypeError, match="ego route, future trajectory"):
        manager.update(
            _world(10, agents),
            agents,
            SimpleNamespace(
                footprint=box(100, 100, 102, 102),
                velocity_x=0.0,
                velocity_y=0.0,
                route="forbidden",
                planner_output="forbidden",
            ),
        )


def test_granted_pre_entry_stall_is_revoked_and_fifo_is_recomputed():
    manager = _manager(stall_timeout_s=0.2, progress_epsilon_m=0.5)
    agents = _crossing_agents()
    manager.update(_world(10, agents), agents, _ego())

    decisions = manager.update(_world(12, agents), agents, _ego())

    assert manager.get_stats()["stall_events"] == 1
    assert decisions["b"].enter
    assert decisions["a"].reason == "fifo_wait"


def test_virtual_lead_is_visible_only_while_its_target_npc_is_propagated():
    intersection = _intersection()
    agents = {
        "a": _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection),
        "b": _Agent("b", (0.0, 10.0), (20.0, 10.0), intersection),
    }
    occupancy = _Occupancy({token: agent.polygon for token, agent in agents.items()})
    manager = TargetedLeadIDMAgentManager(agents, occupancy, map_api=None)
    manager._filter_agents_out_of_range = lambda *_args: None
    barrier = box(3.0, 9.0, 3.2, 11.0)
    manager.set_targeted_virtual_leads({"b": barrier})
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    manager.propagate_agents(
        ego_state, 0.1, 0, defaultdict(list), [], radius=100.0
    )

    assert agents["a"].propagated_leads[0].progress == 20.0
    assert agents["b"].propagated_leads[0].progress < 20.0
    assert not any(token.startswith("stop_line_intersection_manager_")
                   for token in occupancy.get_all_ids())


def test_traffic_cone_is_not_an_idm_lead_but_pedestrian_still_is():
    intersection = _intersection()
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    def propagated_progress(kind):
        agent = _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection)
        occupancy = _Occupancy({"a": agent.polygon})
        manager = TargetedLeadIDMAgentManager({"a": agent}, occupancy, map_api=None)
        manager._filter_agents_out_of_range = lambda *_args: None
        detection = SimpleNamespace(
            track_token="open-loop-object",
            tracked_object_type=kind,
            box=SimpleNamespace(geometry=box(4.0, -0.5, 5.0, 0.5)),
        )
        manager.propagate_agents(
            ego_state, 0.1, 0, defaultdict(list), [detection], radius=100.0
        )
        assert "open-loop-object" not in occupancy.get_all_ids()
        return agent.propagated_leads[0].progress

    assert propagated_progress(TrackedObjectType.TRAFFIC_CONE) == 20.0
    assert propagated_progress(TrackedObjectType.PEDESTRIAN) < 20.0


def test_scoped_lane_ignores_non_vehicle_leads_but_keeps_vehicle_leads():
    intersection = _intersection()
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    def propagated_progress(kind):
        agent = _Agent("a", (0.0, 0.0), (20.0, 0.0), intersection)
        agent._route[0].id = "68604"
        agent._route[0].polygon = box(-3.0, -2.0, 10.0, 2.0)
        occupancy = _Occupancy({"a": agent.polygon})
        manager = TargetedLeadIDMAgentManager({"a": agent}, occupancy, map_api=None)
        manager.configure_scoped_vehicle_only_leads(["68604"])
        manager._filter_agents_out_of_range = lambda *_args: None
        detection = SimpleNamespace(
            track_token="open-loop-object",
            tracked_object_type=kind,
            box=SimpleNamespace(geometry=box(4.0, -0.5, 5.0, 0.5)),
        )
        manager.propagate_agents(
            ego_state, 0.1, 0, defaultdict(list), [detection], radius=100.0
        )
        assert "open-loop-object" not in occupancy.get_all_ids()
        return agent.propagated_leads[0].progress, manager.get_wait_cycle_stats()

    pedestrian_progress, stats = propagated_progress(TrackedObjectType.PEDESTRIAN)
    vehicle_progress, _ = propagated_progress(TrackedObjectType.VEHICLE)

    assert pedestrian_progress == 20.0
    assert stats["scoped_non_vehicle_lead_suppressions"] == 1
    assert vehicle_progress < 20.0


def test_scoped_circular_lane_ignores_manager_virtual_lead_but_keeps_physical_vehicle():
    intersection = _intersection()
    follower = _Agent("follower", (0.0, 0.0), (20.0, 0.0), intersection)
    follower._route[0].id = "68604"
    follower._route[0].polygon = box(-3.0, -2.0, 20.0, 2.0)
    leader = _Agent("leader", (8.0, 0.0), (20.0, 0.0), intersection)
    occupancy = _Occupancy({"follower": follower.polygon, "leader": leader.polygon})
    manager = TargetedLeadIDMAgentManager(
        {"follower": follower, "leader": leader}, occupancy, map_api=None
    )
    manager.configure_scoped_vehicle_only_leads(["68604"])
    manager.set_targeted_virtual_leads({"follower": box(3.0, -1.0, 4.0, 1.0)})
    manager._filter_agents_out_of_range = lambda *_args: None
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    manager.propagate_agents(
        ego_state, 0.1, 0, defaultdict(list), [], radius=100.0
    )

    # The manager's artificial stop at x=3 is ignored, while the real car at x=8 remains the
    # longitudinal lead selected by progress on the follower's rail.
    assert 3.0 < follower.propagated_leads[0].progress < 20.0


def test_projected_footprint_of_vehicle_behind_is_not_selected_as_lead():
    intersection = _intersection()
    front = _Agent("front", (10.0, 0.0), (30.0, 0.0), intersection)
    behind = _Agent("behind", (5.0, 0.0), (30.0, 0.0), intersection)
    # Mimic nuPlan occupancy: the rear vehicle's projected footprint reaches ahead of
    # the front vehicle even though its current physical box remains behind.
    occupancy = _Occupancy({
        "front": front.polygon,
        "behind": box(3.0, -1.0, 16.0, 1.0),
    })
    manager = TargetedLeadIDMAgentManager(
        {"front": front, "behind": behind}, occupancy, map_api=None
    )
    manager._filter_agents_out_of_range = lambda *_args: None
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    manager.propagate_agents(
        ego_state, 0.1, 0, defaultdict(list), [], radius=100.0
    )

    assert front.propagated_leads[0].progress == 20.0


def test_physical_path_lead_margin_reduces_reported_bumper_gap():
    intersection = _intersection()
    follower = _Agent("follower", (0.0, 0.0), (20.0, 0.0), intersection)
    leader = _Agent("leader", (8.0, 0.0), (20.0, 0.0), intersection)
    occupancy = _Occupancy({
        "follower": follower.polygon,
        "leader": leader.polygon,
    })
    manager = TargetedLeadIDMAgentManager(
        {"follower": follower, "leader": leader}, occupancy, map_api=None
    )
    manager.configure_physical_path_leads(0.5)

    lead = manager._nearest_physical_lead_on_path(
        "follower",
        follower,
        follower.get_path_to_go_linestring()
        if hasattr(follower, "get_path_to_go_linestring")
        else LineString([(0.0, 0.0), (20.0, 0.0)]),
        ["follower", "leader"],
    )

    assert lead == ("leader", 3.5)


def test_same_flow_adjacent_lane_vehicle_is_not_a_curved_path_lead():
    intersection = _intersection()
    bus = _Agent("bus", (0.0, 0.0), (30.0, 0.0), intersection)
    bus.width = 3.0
    adjacent = _Agent("adjacent", (5.0, 3.4), (30.0, 3.4), intersection)
    manager = TargetedLeadIDMAgentManager(
        {"bus": bus, "adjacent": adjacent},
        _Occupancy({"bus": bus.polygon, "adjacent": adjacent.polygon}),
        map_api=None,
    )
    path = LineString([(0.0, 0.0), (30.0, 0.0)])

    assert manager._selected_physical_lead_rerank_reason(
        "bus", bus, path, "adjacent"
    ) == "adjacent"
    assert manager._nearest_physical_lead_on_path(
        "bus", bus, path, ["bus", "adjacent"]
    ) is None

    # A perpendicular vehicle crossing the same rail remains a real physical lead.
    crossing = _Agent(
        "crossing", (5.0, 0.0), (5.0, 20.0), intersection,
        heading=pi / 2.0,
    )
    manager.agents["crossing"] = crossing
    manager.agent_occupancy.set("crossing", crossing.polygon)
    assert manager._selected_physical_lead_rerank_reason(
        "bus", bus, path, "crossing"
    ) is None


def test_stopped_active_intersection_holder_eventually_gets_priority_over_stopped_ego():
    intersection = _intersection()
    agent = _Agent("holder", (0.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)
    manager = TargetedLeadIDMAgentManager(
        {"holder": agent}, _Occupancy({"holder": agent.polygon}), map_api=None
    )
    manager._filter_agents_out_of_range = lambda *_args: None
    manager.configure_wait_cycle_monitor(timeout_ticks=3, stopped_speed_mps=0.5)
    manager.configure_active_intersection_tokens(["holder"])
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(4.0, -1.0, 6.0, 1.0)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    for iteration in range(4):
        manager.propagate_agents(
            ego_state, 0.1, iteration, defaultdict(list), [], radius=100.0
        )

    assert agent.propagated_leads[0].progress < 20.0
    assert agent.propagated_leads[-1].progress == 20.0
    stats = manager.get_wait_cycle_stats()
    assert stats["ego_deadlock_breaker_events"] == 1
    assert stats["ego_deadlock_breaker_ticks"] == 1


def test_normal_forward_lead_keeps_stock_projected_occupancy_distance():
    intersection = _intersection()
    follower = _Agent("follower", (0.0, 0.0), (20.0, 0.0), intersection)
    leader = _Agent("leader", (8.0, 0.0), (20.0, 0.0), intersection)
    occupancy = _Occupancy({
        "follower": box(-2.0, -1.0, 12.0, 1.0),
        "leader": box(6.0, -1.0, 20.0, 1.0),
    })
    manager = TargetedLeadIDMAgentManager(
        {"follower": follower, "leader": leader}, occupancy, map_api=None
    )
    manager._filter_agents_out_of_range = lambda *_args: None
    ego_state = SimpleNamespace(
        car_footprint=SimpleNamespace(geometry=box(100, 100, 102, 102)),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(x=0.0, y=0.0)
        ),
        rear_axle=SimpleNamespace(heading=0.0),
    )

    manager.propagate_agents(
        ego_state, 0.1, 0, defaultdict(list), [], radius=100.0
    )

    # Physical bumper gap is 4 m, but stock occupancy distance is 0 because both
    # projected footprints extend to the same route endpoint.  The normal case must
    # retain stock behavior; only a selected rear vehicle activates the correction.
    assert follower.propagated_leads[0].progress == 0.0
    assert manager.get_wait_cycle_stats()["rear_lead_rejections"] == 0


def test_wait_for_cycle_is_reported_only_after_persistence_threshold():
    agents = {
        "a": SimpleNamespace(velocity=0.0),
        "b": SimpleNamespace(velocity=0.0),
    }
    manager = TargetedLeadIDMAgentManager(agents, _Occupancy(), map_api=None)
    manager.configure_wait_cycle_monitor(timeout_ticks=3, stopped_speed_mps=0.5)

    manager._update_wait_cycles(10, {"a": "b", "b": "a"})
    manager._update_wait_cycles(11, {"a": "b", "b": "a"})
    assert manager.get_wait_cycle_stats()["wait_cycle_events"] == 0

    manager._update_wait_cycles(12, {"a": "b", "b": "a"})
    assert manager.get_wait_cycle_stats() == {
        "wait_cycle_events": 1,
        "wait_cycle_agent_ticks": 2,
        "wait_cycle_breaker_ticks": 1,
        "rear_lead_rejections": 0,
        "adjacent_lead_rejections": 0,
        "ego_deadlock_breaker_events": 0,
        "ego_deadlock_breaker_ticks": 0,
        "scoped_non_vehicle_lead_suppressions": 0,
    }
    assert manager._wait_cycle_release_leads == {"a": "b"}

    manager._update_wait_cycles(13, {})
    assert manager._wait_cycle_release_leads == {"a": "b"}
    manager._update_wait_cycles(18, {})
    assert manager._wait_cycle_release_leads == {}


def test_wait_cycle_releases_vehicle_moving_away_not_vehicle_driving_through_lead():
    intersection = _intersection()
    toward = _Agent("toward", (0.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)
    away = _Agent("away", (5.0, 0.0), (20.0, 0.0), intersection, velocity=0.0)
    manager = TargetedLeadIDMAgentManager(
        {"toward": toward, "away": away},
        _Occupancy({"toward": toward.polygon, "away": away.polygon}),
        map_api=None,
    )
    manager.configure_wait_cycle_monitor(timeout_ticks=1, stopped_speed_mps=0.5)

    manager._update_wait_cycles(10, {"toward": "away", "away": "toward"})

    assert manager._wait_cycle_release_leads == {"away": "toward"}


def test_wait_for_chain_ending_at_virtual_stop_is_not_a_cycle():
    agents = {
        "a": SimpleNamespace(velocity=0.0),
        "b": SimpleNamespace(velocity=0.0),
    }
    manager = TargetedLeadIDMAgentManager(agents, _Occupancy(), map_api=None)
    manager.configure_wait_cycle_monitor(timeout_ticks=1)

    manager._update_wait_cycles(10, {"a": "b"})

    assert manager.get_wait_cycle_stats()["wait_cycle_events"] == 0
