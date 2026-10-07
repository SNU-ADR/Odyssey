"""Build the (120,5) SD-route polyline target for a navsim Scene.

Thin wrapper over ``build_route_centerline_target`` in OdysseyZoo/sdroute/, the one builder every
model here trains and evaluates on, so all models see the same route: same PDM centerline, same
direction-aware HMM re-match onto the OSM graph, same 1 m resampling, same mask convention.

A missing builder raises instead of returning zeros: a silently zeroed route trains a model
that looks fine and has learned nothing from the route.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

#: Must match SDRouteSegmentEncoder's expectations (24 segments x 5 points).
ROUTE_NUM_POINTS: int = 120
ROUTE_HORIZON_M: float = 120.0

#: OdysseyZoo/sdroute, from models/<repo>/navsim/agents/sdroute/.
_SDROUTE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "sdroute",
))

#: Serialises the first load. lru_cache does not: concurrent first callers (the eval thread pool)
#: would all run it, and every one after the first would find the package in sys.modules while
#: it is still executing, without build_route_centerline_target yet.
_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _route_builder():
    """Load OdysseyZoo/sdroute by file path, under a private module name.

    Not `import sdroute`: this runs inside whichever repo's navsim is the host, with that repo's
    sys.path, and the builder must resolve to the same directory from all of them. Its own
    relative imports work because the package is registered before it executes; its PDM helpers
    (`navsim.planning.simulation.planner.pdm_planner`) come from the host's navsim.
    """
    init = os.path.join(_SDROUTE_DIR, "__init__.py")
    if not os.path.isfile(init):
        raise RuntimeError(f"[sdroute] SD-route builder not found at {init}.")
    name = "_odysseyzoo_sdroute"
    with _LOAD_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[_SDROUTE_DIR])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return sys.modules[name].build_route_centerline_target


def build_sdroute_target(scene, frame_idx: int, use_gt_walk: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scene -> (route_centerline (120,5) float32, route_centerline_mask (120,) bool).

    `use_gt_walk` walks the lane graph along the GT ego trajectory instead of trusting the
    dataset roadblock_ids + BFS correction -- the source every trained checkpoint here used.
    All failure paths inside the builder return zeros + an all-False mask, which the
    encoder and the cross-attention both handle explicitly.
    """
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

    build = _route_builder()
    frame = scene.frames[frame_idx]
    ego_pose = StateSE2(*frame.ego_status.ego_pose)
    ego_state = EgoState.build_from_rear_axle(
        ego_pose,
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(0),
        rear_axle_velocity_2d=StateVector2D(*frame.ego_status.ego_velocity),
        rear_axle_acceleration_2d=StateVector2D(*frame.ego_status.ego_acceleration),
    )
    # Global ego positions from this frame on. get_future_trajectory() returns ego-local
    # poses, which would not match the global lane polylines.
    gt_xy_global = (
        np.array([scene.frames[j].ego_status.ego_pose[:2] for j in range(frame_idx, len(scene.frames))])
        if use_gt_walk
        else None
    )
    return build(
        scene.map_api,
        frame.roadblock_ids,
        ego_state,
        ego_pose,
        ROUTE_NUM_POINTS,
        ROUTE_HORIZON_M,
        use_sd_route=True,
        map_location=scene.scene_metadata.map_name,
        use_gt_walk=use_gt_walk,
        gt_xy_global=gt_xy_global,
    )
