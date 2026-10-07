"""ReCogDrive + SD-route conditioning, as published in OdysseyZoo.

Repo   OdysseyZoo/models/recogdrive  (vendored recogdrive + the sd-route / nav_prompt changes)
Cfg    recogdrive_sdroute_agent   use_sdroute=true: the route polyline becomes one token added to the
                                  ego-status conditioning; drop_command=true: the command one-hot is
                                  gone from the planner; nav_prompt=sdroute_text: the VLM prompt's
                                  navigation line is the same route written as text
Ckpt   ckpts/recogdrive_sdroute.ckpt      imitation learning, then GRPO
       ckpts/recogdrive_sdroute_il.ckpt   imitation learning only (same config)

Everything but the route is ReCogDrivePlanner: the same VLM directory, the one camera, the fp32 patch
on sm_75, the image-path AgentInput and the unknown-command clamp.

The route comes from the SAME baked sidecar the other SD-route models run on
(``route_sidecar.StaticRouteCenterline``): the (1,120,5) + mask pair the training cache carried,
120 points at 1 m. It reaches the agent in ``targets``, which is where ``ReCogDriveAgent.forward``
reads it in training as well. STRICT: a frame without a usable route stops the rollout, because the
command is gone from the planner and a zeroed route would give a run that looks healthy while
measuring nothing.

nav_prompt=sdroute_text writes the VLM's navigation line from that same route with
``recogdrive_features.sdroute_navigation`` -- the function the training cache was built with -- and
hands it over in ``features['nav_text']``. A route too short to describe falls back to the command,
exactly as it did in training.

build() checks the composed config's use_sdroute / drop_command / nav_prompt. It cannot tell the two
checkpoints apart (same parameter set): the agent config names the checkpoint.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .recogdrive import ReCogDrivePlanner


class _RouteCfg:
    """The two fields StaticRouteCenterline reads. 120 points over 120 m is what
    scripts/sdroute/make_sdroute_cache.py baked the training route with (sdroute_kv/route_target.py);
    the builder's own defaults happen to agree, but the length is stated here rather than inherited."""

    route_cl_num_points = 120
    route_cl_horizon = 120.0


class ReCogDriveSDRoutePlanner(ReCogDrivePlanner):
    """A subclass names its config and tag, and the values that prove the composed config is it."""
    SD_ROUTE = "targets"   # fed by this adapter; the profile must declare the same

    #: {setting: value} the composed agent must carry, on top of use_sdroute.
    ARM_VALUES: Dict[str, object] = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._route_builder = None
        self._newest_frame: Optional[dict] = None
        self._nav_text_logged = False

    # ------------------------------------------------------------------ build
    def _arm_value(self, name: str):
        """use_sdroute and nav_prompt are agent attributes; drop_command lives only on the planner
        config the agent built its action head from."""
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        return getattr(self.agent.action_head.config, name, None)

    def build(self) -> None:
        super().build()
        want = {"use_sdroute": True, **self.ARM_VALUES}
        for name, value in want.items():
            got = self._arm_value(name)
            if got != value:
                raise SystemExit(
                    f"[{self.TAG}] {name} is {got!r}, expected {value!r} -- the composed yaml "
                    f"({self.cfg_name}) is not the {self.TAG} arm. Use --cfg {self.DEFAULT_CFG}.")

        try:
            from ..route_sidecar import StaticRouteCenterline
        except ImportError:              # run as a plain script, not a package
            from route_sidecar import StaticRouteCenterline  # type: ignore[no-redef]
        self._route_builder = StaticRouteCenterline(
            _RouteCfg(), sidecar_dir=self.route_file, strict=True)
        print(f"[{self.TAG}] route centerline ENABLED (strict, source=sidecar, "
              f"{_RouteCfg.route_cl_num_points} points / {_RouteCfg.route_cl_horizon:.0f} m); "
              f"drop_command={self._arm_value('drop_command')} "
              f"nav_prompt={self.agent.nav_prompt!r}", flush=True)

    @property
    def uses_driving_command(self) -> bool:
        """Base asks for ``drop_driving_command`` on ``agent._config``, which ReCogDrive has neither
        of. Here the command reaches the model through the status one-hot unless drop_command is
        set, and through the VLM prompt when nav_prompt is 'command' -- either path means it is a
        real input, so the viewer should draw it."""
        return (not bool(self._arm_value("drop_command"))) or self.agent.nav_prompt == "command"

    # ------------------------------------------------------------------ infer
    def _build_features(self, agent_input, scene_list: List[dict]) -> Dict:
        # The route is built from the NEWEST raw frame (ego pose + scene token), which _forward has
        # no other way to see -- stash it here, where scene_list passes by.
        self._newest_frame = scene_list[-1]
        return super()._build_features(agent_input, scene_list)

    def _forward(self, features: Dict) -> Dict:
        rc = self._route_builder.build(self._newest_frame)
        if rc is None:                   # strict=True raises instead; a belt against strict=False
            raise RuntimeError(f"[{self.TAG}] no route centerline for this frame")
        route, mask = rc                 # (1,P,5) float32 / (1,P) bool, on CPU
        targets = {"route_centerline": route.to(self.device),
                   "route_centerline_mask": mask.to(self.device)}
        if self.agent.nav_prompt == "sdroute_text":
            from navsim.agents.recogdrive.recogdrive_features import sdroute_navigation

            features["nav_text"] = [sdroute_navigation(
                route[0], mask[0], features["high_command_one_hot"][0].detach().cpu())]
            if not self._nav_text_logged:
                print(f"[{self.TAG}] first navigation line: [{features['nav_text'][0].upper()}] "
                      f"(route_text.py; logged once)", flush=True)
                self._nav_text_logged = True
        return self.agent.forward(features, targets)


class ReCogDriveSDRouteNavTextPlanner(ReCogDriveSDRoutePlanner):
    """The prompt's navigation line is the route as text, e.g. [TURN RIGHT IN 50 M, THEN TURN LEFT]."""

    DEFAULT_CFG = "recogdrive_sdroute_agent"
    TAG = "recogdrive_sdroute"
    ARM_VALUES: Dict[str, object] = {"drop_command": True, "nav_prompt": "sdroute_text"}


class ReCogDriveSDRouteNavTextILPlanner(ReCogDriveSDRouteNavTextPlanner):
    """The same config with the imitation-only checkpoint (recogdrive_sdroute_il.ckpt, before GRPO);
    only the loaded weights differ, so it gets its own tag."""

    TAG = "recogdrive_sdroute_il"
