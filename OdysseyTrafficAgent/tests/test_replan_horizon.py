"""replan_horizon_sim_steps: the K-1 coast steps reuse the last adopted plan, minus the rows already driven."""
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey.common.dataclasses import Trajectory
from odyssey.components.agents.client import planner_client as module
from odyssey.utils.cadence import PLAN_FILE_DT, plan_file_stride


def client(monkeypatch, k, sim_dt=0.1, rows=40):
    c = module.PlannerClient.__new__(module.PlannerClient)
    c.config = {"replan_horizon_sim_steps": k, "num_history": 16}
    c.agent = SimpleNamespace()
    monkeypatch.setattr(module.PlannerClient, "engine", property(lambda self: SimpleNamespace(sim_dt=sim_dt)))
    xy = np.stack([np.arange(rows, dtype=float), np.zeros(rows)], 1)
    c._cached_traj = Trajectory(waypoints=xy, velocities=xy.copy(), headings=np.zeros(rows),
                                angular_velocities=np.zeros(rows), wp_dt=PLAN_FILE_DT)
    return c


def test_a_coast_step_drops_the_rows_already_driven(monkeypatch):
    c = client(monkeypatch, k=5)
    t = c.get_trajectory(15 + 2)                         # second step after the planning step (15)
    drop = 2 * plan_file_stride(0.1)
    assert t.waypoints[0, 0] == drop and len(t.waypoints) == 40 - drop


def test_a_planning_step_asks_the_planner(monkeypatch):
    c = client(monkeypatch, k=5)
    with pytest.raises(Exception):                       # a planning step waits for the plan file instead of using the cache
        c.get_trajectory(15 + 5)


def test_k1_never_coasts(monkeypatch):
    c = client(monkeypatch, k=1)
    with pytest.raises(Exception):
        c.get_trajectory(15 + 2)
