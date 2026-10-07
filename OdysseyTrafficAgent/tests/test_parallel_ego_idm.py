import threading

from odyssey.components.agents.ego_agent import EgoAgent
from odyssey.manager.agent_manager import BaseAgentManager


class _Policy:
    is_current_step_valid = True

    def __init__(self, result="planned"):
        self.result = result
        self.calls = 0

    def act(self):
        self.calls += 1
        return self.result


class _Controller:
    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


def _bare_ego():
    ego = EgoAgent.__new__(EgoAgent)
    ego.policy = _Policy()
    ego.controller = _Controller()
    ego.navigation = object()
    ego.trajectory = "old"
    ego._traj_step = 7
    ego.accel_calls = 0
    ego.set_acceleration = lambda value: setattr(ego, "accel_calls", ego.accel_calls + 1)
    return ego


def test_ego_plan_does_not_commit_state_until_explicit_commit():
    ego = _bare_ego()

    result = ego.compute_step_plan()

    assert result == (True, "planned")
    assert ego.trajectory == "old"
    assert ego.controller.calls == 0
    assert ego._traj_step == 7

    ego.commit_step_plan(result)

    assert ego.trajectory == "planned"
    assert ego.controller.calls == 1
    assert ego.accel_calls == 1
    assert ego._traj_step == 8


def test_regular_ego_step_keeps_plan_then_commit_behavior():
    ego = _bare_ego()

    ego.step()

    assert ego.policy.calls == 1
    assert ego.trajectory == "planned"
    assert ego.controller.calls == 1
    assert ego._traj_step == 8


def test_parallel_ego_and_idm_read_same_precommit_snapshot():
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._ego_planner_executor = None
    planner_started = threading.Event()
    idm_finished = threading.Event()
    npc = type("NPC", (), {"state": "old"})()

    class Ego:
        state = "old"

        def compute_step_plan(self):
            planner_started.set()
            assert self.state == "old"
            assert npc.state == "old"
            # If manager ran planner then IDM serially, this wait would time out.
            assert idm_finished.wait(2.0)
            return True, "plan"

        def commit_step_plan(self, result):
            assert idm_finished.is_set()
            assert result == (True, "plan")
            self.state = "new"

    ego = Ego()

    class Batch:
        def prepare_step(self, step):
            assert step == 12
            assert planner_started.wait(2.0)
            assert ego.state == "old"
            assert npc.state == "old"
            idm_finished.set()

    try:
        manager._compute_ego_and_idm(ego, Batch(), 12)
    finally:
        manager._ego_planner_executor.shutdown(wait=True, cancel_futures=True)

    assert ego.state == "new"
    stats = manager.parallel_runtime_stats
    assert stats["steps"] == 1
    assert stats["ego_planner_s"] >= 0
    assert stats["idm_s"] >= 0
    assert stats["critical_path_s"] >= max(stats["ego_planner_s"], stats["idm_s"])
