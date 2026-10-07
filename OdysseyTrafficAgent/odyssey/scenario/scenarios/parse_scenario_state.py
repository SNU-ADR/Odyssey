# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
Helper functions for parsing scenario information.
"""

import math
import numpy as np
import copy

from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def parse_object_track(object_dict, time_idx, sim_dt):
    # sim_dt is required, not defaulted. 0.5 is the frame spacing only for scenes whose
    # sample_rate is 10, and this function derives
    # angular_velocity for every track that lacks one, so a wrong dt doubles or halves
    # the yaw rate of every agent in the scene. The one caller passes it explicitly.
    """
    Parse object state of a list of timestamps.
    """

    state = object_dict[SD.STATE]

    assert isinstance(time_idx, np.ndarray)

    epi_length = len(state[SD.POSITION])
    time_idx = np.maximum(np.minimum(time_idx, epi_length - 1), 0)

    ret = {k: v[time_idx] for k, v in state.items()}

    if 'angular_velocity' not in ret:  # agents without angular velocit information.
        angular_velocity = compute_angular_velocity(
            initial_heading=state[SD.HEADING][time_idx],
            final_heading=state[SD.HEADING][np.minimum(time_idx + 1, epi_length - 1)],
            dt=sim_dt,
        )
        ret["angular_velocity"] = angular_velocity

    valid_indicator = (ret[SD.VALID] == 1)
    ret['angular_velocity'][~valid_indicator] = 0

    return ret


def parse_full_trajectory(state):
    """
    Get valid trajectory from the object_dict.
    """
    positions = state[SD.POSITION]
    valid_indicator = state[SD.VALID] == 1
    trajectory = copy.deepcopy(positions[valid_indicator, :2])
    return trajectory, valid_indicator
