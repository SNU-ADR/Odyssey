"""The parking-entry connector veto changes no other map or nuPlan routing preference."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusType
from nuplan.planning.simulation.observation.idm import idm_agent as stock_idm

from odyssey.components.agents.policy import nuplan_idm_policy as policy


class _Baseline:
    def __init__(self, heading=0.0, curvature=0.0):
        self.heading = heading
        self.curvature = curvature
        self.discrete_path = [object()]

    def get_nearest_pose_from_position(self, _position):
        return SimpleNamespace(heading=self.heading)

    def get_curvature_at_arc_length(self, _distance):
        return self.curvature


class _Edge:
    def __init__(self, edge_id, heading=0.0, curvature=0.0, signalized=False):
        self.id = edge_id
        self.baseline_path = _Baseline(heading, curvature)
        self.outgoing_edges = []
        self._signalized = signalized

    def has_traffic_lights(self):
        return self._signalized


class _Map:
    def __init__(self, edges):
        self.edges = edges
        self.marker = object()

    def get_all_map_objects(self, _position, _layer):
        return self.edges


class _RouteAgent:
    def __init__(self, start):
        self._route = [start]
        self._state = SimpleNamespace(progress=1)
        self._policy = SimpleNamespace(target_velocity=10.0, headway_time=1.5)
        self._minimum_path_length = 20.0
        self._path = None

    @property
    def end_segment(self):
        return self._route[-1]

    def get_progress_to_go(self):
        return 0.0 if len(self._route) == 1 else 100.0

    def get_path_to_go(self):
        return []

    def get_route(self):
        return list(self._route)


def test_no_benchmark_scene_excludes_idm_connectors():
    assert policy._IDM_EXCLUDED_CONNECTORS == {}


def test_starting_segment_wraps_pi_without_changing_other_map_methods():
    # -179.4 and +179.4 degrees are adjacent; raw subtraction says 358.8 degrees.
    straight = _Edge("52643", heading=np.pi - 0.01)
    wrong_turn = _Edge("52832", heading=-np.pi + 0.3)
    base = _Map([wrong_turn, straight])
    view = policy._IDMStartingSegmentMap(base)
    position = SimpleNamespace(heading=-np.pi + 0.01)
    assert view.get_all_map_objects(position, SemanticMapLayer.LANE_CONNECTOR) == [straight]
    assert view.marker is base.marker


def test_wrapped_heading_keeps_ordinary_candidate_order():
    heading = 0.5
    close = _Edge("close", heading=0.55)
    far = _Edge("far", heading=0.8)
    view = policy._IDMStartingSegmentMap(_Map([far, close]))
    position = SimpleNamespace(heading=heading)
    assert view.get_all_map_objects(position, SemanticMapLayer.LANE) == [close]


def test_starting_segment_excludes_only_named_connectors():
    entries = [_Edge("51959"), _Edge("52056"), _Edge("52643")]
    view = policy._IDMStartingSegmentMap(_Map(entries), {"51959", "52056"})
    position = SimpleNamespace(heading=0.0)
    assert view.get_all_map_objects(position, SemanticMapLayer.LANE_CONNECTOR) == [entries[2]]
    empty = policy._IDMStartingSegmentMap(_Map(entries[:2]), {"51959", "52056"})
    assert empty.get_all_map_objects(position, SemanticMapLayer.LANE_CONNECTOR) == []


def test_route_veto_keeps_stock_ranking_of_remaining_edges(monkeypatch):
    monkeypatch.setattr(policy, "create_path_from_se2", lambda path: path)
    monkeypatch.setattr(stock_idm, "create_path_from_se2", lambda path: path)
    entries = [
        _Edge("51959", curvature=0.0),
        _Edge("52056", curvature=0.0),
        _Edge("52643", curvature=0.0),
        _Edge("52832", curvature=0.2),
    ]
    start = _Edge("48625")
    start.outgoing_edges = entries
    filtered = _RouteAgent(start)
    policy._install_idm_connector_filter(filtered, {"51959", "52056"})
    first_method = filtered.plan_route
    policy._install_idm_connector_filter(filtered, {"51959", "52056"})
    assert filtered.plan_route is first_method
    filtered.plan_route(defaultdict(list))
    assert [edge.id for edge in filtered._route] == ["48625", "52643"]

    stock_start = _Edge("48625")
    stock_start.outgoing_edges = entries[2:]
    stock = _RouteAgent(stock_start)
    stock_idm.IDMAgent.plan_route(stock, defaultdict(list))
    assert stock._route[-1].id == filtered._route[-1].id


def test_veto_still_obeys_red_lights_and_stops_without_alternative(monkeypatch):
    monkeypatch.setattr(policy, "create_path_from_se2", lambda path: path)
    forbidden = _Edge("47421")
    allowed_signal = _Edge("52432", signalized=True)
    start = _Edge("51507")
    start.outgoing_edges = [forbidden, allowed_signal]
    agent = _RouteAgent(start)
    policy._install_idm_connector_filter(agent, {"47421"})
    status = defaultdict(list)
    agent.plan_route(status)
    assert len(agent._route) == 1
    status[TrafficLightStatusType.GREEN].append("52432")
    agent.plan_route(status)
    assert agent._route[-1].id == "52432"



def test_live_route_guard_rejects_only_excluded_connectors():
    start = _Edge("48625")
    allowed = _Edge("52643")
    forbidden = _Edge("51959")
    agent = _RouteAgent(start)
    agent._route.append(allowed)
    policy._assert_no_excluded_idm_routes({"car": agent}, {"51959", "52056"})
    agent._route.append(forbidden)
    with pytest.raises(RuntimeError, match="car.*51959"):
        policy._assert_no_excluded_idm_routes({"car": agent}, {"51959", "52056"})
