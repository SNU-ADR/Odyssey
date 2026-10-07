"""Default IDM branch intent: source route choice and signal stopping are separate."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.observation.idm.idm_agent_manager import IDMAgentManager

from odyssey.components.agents.policy import nuplan_idm_policy as policy
from odyssey.components.agents.policy.idm_route_intent import (
    build_source_route_intent, choose_source_branch,
)


class Edge:
    def __init__(self, edge_id, points, curvature=0.0, signalized=False):
        self.id = edge_id
        self.outgoing_edges = []
        self._signalized = signalized
        self.baseline_path = SimpleNamespace(
            discrete_path=[SimpleNamespace(x=x, y=y) for x, y in points],
            get_curvature_at_arc_length=lambda _distance: curvature,
        )

    def has_traffic_lights(self):
        return self._signalized


class Agent:
    def __init__(self, start, intent):
        self._route = [start]
        self._odyssey_route_intent = intent
        self._odyssey_route_token = "car"
        self._odyssey_route_source_row = 7
        self._odyssey_route_decisions = []
        self._state = SimpleNamespace(progress=1.0)
        self._policy = SimpleNamespace(target_velocity=10.0, headway_time=1.5)
        self._minimum_path_length = 20.0

    @property
    def end_segment(self):
        return self._route[-1]

    def get_progress_to_go(self):
        return 0.0 if len(self._route) == 1 else 100.0

    def get_path_to_go(self):
        return []

    def get_route(self):
        return list(self._route)


def _intent(points):
    return build_source_route_intent(
        {"position": points, "valid": np.ones(len(points), dtype=bool)},
        np.zeros(2), 0,
    )


def test_local_branch_intent_goes_straight_first_then_turns():
    points = [(float(x), 0.0) for x in range(-10, 21)]
    points += [(20.0, float(y)) for y in range(1, 21)]
    intent = _intent(points)
    incoming = Edge("before", [(-10, 0), (0, 0)])
    early_turn = Edge("early_turn", [(0, 0), (0, 25)], curvature=0.0)
    through = Edge("through", [(0, 0), (20, 0)], curvature=0.2)
    first = Agent(incoming, intent)
    assert choose_source_branch(first, [early_turn, through], early_turn) is through

    second_incoming = Edge("second_before", [(0, 0), (20, 0)])
    later_turn = Edge("later_turn", [(20, 0), (20, 25)], curvature=0.2)
    later_through = Edge("later_through", [(20, 0), (40, 0)], curvature=0.0)
    second = Agent(second_incoming, intent)
    assert choose_source_branch(second, [later_through, later_turn], later_through) is later_turn


def test_red_logged_branch_is_selected_but_stop_line_remains_active(monkeypatch):
    monkeypatch.setattr(policy, "create_path_from_se2", lambda path: path)
    incoming = Edge("before", [(-10, 0), (0, 0)])
    red_turn = Edge("red_turn", [(0, 0), (0, 25)], curvature=0.2, signalized=True)
    green_straight = Edge("green_straight", [(0, 0), (25, 0)], signalized=True)
    incoming.outgoing_edges = [green_straight, red_turn]
    intent = _intent([(0.0, float(y)) for y in range(-10, 26)])
    agent = Agent(incoming, intent)
    policy._install_idm_connector_filter(agent, ())
    status = defaultdict(list, {
        TrafficLightStatusType.RED: ["red_turn"],
        TrafficLightStatusType.GREEN: ["green_straight"],
    })
    agent.plan_route(status)
    assert agent.end_segment is red_turn
    assert agent._odyssey_route_decisions[0][3:5] == ("green_straight", "red_turn")

    stop_line = object()
    map_api = SimpleNamespace(get_map_object=lambda _id, _layer: SimpleNamespace(
        stop_lines=[stop_line]
    ))
    manager = SimpleNamespace(_map_api=map_api)
    assert IDMAgentManager._get_relevant_stop_lines(manager, agent, status) == [stop_line]


def test_only_red_logged_branch_is_reserved_until_green(monkeypatch):
    monkeypatch.setattr(policy, "create_path_from_se2", lambda path: path)
    incoming = Edge("before", [(-10, 0), (0, 0)])
    red = Edge("red", [(0, 0), (0, 25)], signalized=True)
    incoming.outgoing_edges = [red]
    agent = Agent(incoming, _intent([(0.0, float(y)) for y in range(-10, 26)]))
    policy._install_idm_connector_filter(agent, ())
    agent.plan_route(defaultdict(list, {TrafficLightStatusType.RED: ["red"]}))
    assert agent.end_segment is red
    assert agent._odyssey_route_decisions[0][3:5] == ("", "red")


def test_without_reliable_intent_stock_green_route_or_wait_is_preserved(monkeypatch):
    monkeypatch.setattr(policy, "create_path_from_se2", lambda path: path)
    incoming = Edge("before", [(-10, 0), (0, 0)])
    red = Edge("red", [(0, 0), (0, 25)], signalized=True)
    green = Edge("green", [(0, 0), (25, 0)], signalized=True)
    incoming.outgoing_edges = [red, green]
    agent = Agent(incoming, None)
    policy._install_idm_connector_filter(agent, ())
    status = defaultdict(list, {
        TrafficLightStatusType.RED: ["red"],
        TrafficLightStatusType.GREEN: ["green"],
    })
    agent.plan_route(status)
    assert agent.end_segment is green

    only_red = Edge("before2", [(-10, 0), (0, 0)])
    only_red.outgoing_edges = [red]
    waiting = Agent(only_red, None)
    policy._install_idm_connector_filter(waiting, ())
    waiting.plan_route(status)
    assert waiting.end_segment is only_red


def test_late_admission_intent_uses_mapped_source_row():
    batch = policy.NuPlanIDMBatch.__new__(policy.NuPlanIDMBatch)
    points = np.asarray([(float(x), 0.0) for x in range(30)])
    batch.scene = {"object_track": {"car": {"state": {
        "position": points, "valid": np.ones(30, dtype=bool),
    }}}}
    batch.converter = SimpleNamespace(object_source_row=lambda _token, _step: 5)
    batch._origin = np.asarray([100.0, 200.0])
    batch._idm_route_decisions = []
    agent = Agent(Edge("before", [(0, 0), (1, 0)]), None)
    del agent._odyssey_route_intent
    batch._attach_idm_route_intent("car", agent, 99)
    assert agent._odyssey_route_source_row == 5
    assert np.allclose(agent._odyssey_route_intent[0][0], [105.0, 200.0])
