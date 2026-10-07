"""Adapter template: subclass this only when the generic adapter cannot run your model.

A native navsim ``AbstractAgent`` (hydra yaml taking ``checkpoint_path``, ``initialize()``,
``get_sensor_config()``, ``get_feature_builders()``, ``forward(features[, targets])`` returning
``{"trajectory": (B, N, 3)}``) runs from the agent config alone. Point ``model.adapter`` at a
subclass like the one below when it does not, for example:

- the model emits poses every 0.1 s instead of 0.5 s (set ``PLAN_DT``);
- it needs inputs the feature builders do not produce (override ``_build_features``);
- it is called differently from ``agent.forward(features)`` (override ``_forward``);
- its output dict names the trajectory differently (override ``_extract_trajectory``).

Reference in your agent config, relative to the config file::

    model:
      adapter: my_adapter.py:MyPlanner
"""
from typing import Dict, List

import numpy as np

from odyssey_bridge.planners.base import NavsimPlanner


class MyPlanner(NavsimPlanner):
    #: name used in log lines
    TAG = "my_model"
    #: seconds between output poses; the runtime resamples to the simulator's 0.1 s grid.
    #: Must equal ``output.plan_dt`` in the agent config.
    # PLAN_DT = 0.1
    #: set to "features" or "targets" if THIS adapter feeds the SD route itself; leave unset
    #: to let the base class feed it where ``navigation.sd_route`` says.
    # SD_ROUTE = "targets"

    def build(self) -> None:
        super().build()
        # Guard the loaded config here when a wrong yaml would still load without an error,
        # e.g. `if not self.agent._config.my_flag: raise SystemExit(...)`.

    def _build_features(self, agent_input, scene_list: List[dict]) -> Dict:
        # Returns unbatched tensors; the base class adds the batch dim and moves them to the
        # device (also inside lists). ``scene_list[-1]`` is the newest raw frame.
        features = super()._build_features(agent_input, scene_list)
        return features

    def _forward(self, features: Dict) -> Dict:
        return super()._forward(features)

    def _extract_trajectory(self, out: Dict) -> np.ndarray:
        # (N, 3) ego-local x, y, heading at PLAN_DT spacing.
        return super()._extract_trajectory(out)
