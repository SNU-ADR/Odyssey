# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Trajectory:
    waypoints: npt.NDArray[np.float32]
    velocities: npt.NDArray[np.float32] = None
    headings: npt.NDArray[np.float32] = None
    angular_velocities: npt.NDArray[np.float32] = None
    # Seconds between consecutive waypoints, declared by whoever built them.
    #
    # Consumers read this array two different ways -- the trackers by TIME, LogPlayController
    # by INDEX -- so they must agree on the spacing. The producer is the only party that knows
    # it, so it says so here instead of the consumer looking it up in a config key that nothing
    # keeps in sync. None means the producer did not declare it.
    wp_dt: float = None
