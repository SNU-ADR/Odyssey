"""Exercise the real worker handler with a small recurrent native planner double."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from odyssey_runtime.planner import PlannerWorker, MemoryInputs, encode_jpeg
from odyssey_runtime.profile import ModelProfile
from tests.runtime.test_profile import spec


def test_all_ticks_episode_reset_route_exhaustion_and_sequence():
    from odyssey_bridge.route_sidecar import RouteExhausted

    class Agent:
        count = 100
        resets = 0

        def reset_cache(self):
            self.count = 0
            self.resets += 1

    agent = Agent()
    seen = []

    def infer(history):
        step = history[-1]["frame_idx"]
        seen.append(step)
        agent.count += 1
        if step == 2:
            raise RouteExhausted("finished")
        return np.full((8, 3), agent.count, dtype=np.float64)

    d = spec()
    d.update(stateful=True, reset_method="reset_cache")
    worker = PlannerWorker.__new__(PlannerWorker)
    worker.profile = ModelProfile(d)
    worker.planner = SimpleNamespace(agent=agent, infer=infer, PLAN_DT=0.5)
    worker.config = {}
    worker.inputs = MemoryInputs(worker.profile)
    worker.history = deque(maxlen=worker.profile.history_capacity)
    worker.episode, worker.last_step, worker.audit = None, -1, None
    jpeg = encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    worker.buffer = jpeg * len(worker.profile.cameras)

    def message(step, episode="first"):
        return dict(
            step=step,
            episode=episode,
            frame={"frame_idx": step},
            images=[(k, len(jpeg)) for k in worker.profile.cameras],
            payload_size=len(worker.buffer),
        )

    for step in range(3):
        reply = worker.handle(message(step))
        assert reply["step"] == step
        assert bool(reply.get("route_exhausted")) == (step == 2)
    assert seen == [0, 1, 2] and agent.count == 3 and agent.resets == 1
    with pytest.raises(ValueError, match="nonconsecutive"):
        worker.handle(message(4))
    with pytest.raises(ValueError, match="step zero"):
        worker.handle(message(1, "second"))
    reply = worker.handle(message(0, "second"))
    assert reply["episode"] == "second" and agent.count == 1 and agent.resets == 2
    assert len(worker.history) == 1
