# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from abc import ABC

import torch

from odyssey.base_class.base_runnable import BaseRunnable


class RenderState(dict):
    CAMERAS = "cameras"
    LIDAR = "lidar"
    AGENT_STATE = "agent_state"   # {object_id: np.ndarray([x, y, heading])}
    # {object_id: log row the actor is currently replaying}. When the replay clock moves an actor
    # off the log clock (agent_replay=hybrid/sector), the row the simulator uses differs from the
    # sim step. Without this the renderer would draw a different pose than the scored one.
    # Empty means every actor follows the log clock.
    AGENT_SOURCE_ROW = "agent_source_row"
    # {traffic-light head id: observed source frame}.  This is absent unless a
    # rollout explicitly supplies tl_control_path, preserving natural replay.
    TL_SOURCE_FRAMES = "tl_source_frames"
    # {head: frame}: entries of TL_SOURCE_FRAMES whose label allows an unobserved frame
    # (tl_control allow_unobserved). The key is absent when there are none.
    TL_ALLOW_UNOBSERVED = "tl_allow_unobserved"
    TIMESTAMP = "timestamp"
    # Actor tokens held out of the world this step. Renderers draw actors from their own
    # checkpoint poses, so absence from AGENT_STATE does not stop them being drawn; this
    # says so explicitly, and keeps the drawn set equal to the simulated and scored one.
    SUPPRESSED_ACTORS = "suppressed_actors"
    # Actor tokens the simulator pins to their log pose (static agents, vehicles IDM returned to
    # their log pose). A vehicle that was parked in the log but is absent here is being driven by
    # the simulator.
    HELD_AT_LOG_POSE = "held_at_log_pose"
    # True on a step whose cameras are skipped (render_every). The renderer still reports the
    # ego pose -- the frame record is written every step and needs it -- but rasterises nothing
    # and runs no restorer, and returns an empty `cameras` mapping.
    SKIP_CAMERAS = "skip_cameras"


class BaseRenderer(BaseRunnable, ABC):

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.__from_scratch__()

    def __from_scratch__(self):
        pass

    def reset(self):
        self.__from_scratch__()

    @property
    def background_asset(self):
        return None

    def set_asset(self, asset):
        pass

    def render(self, render_state: RenderState):
        pass
