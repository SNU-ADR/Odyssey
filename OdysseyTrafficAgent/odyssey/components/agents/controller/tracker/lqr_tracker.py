# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
Implementation of <LQR> tracking module,
    Used for translating the trajectory signal into steering and braking control.
"""


import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from odyssey.components.agents.controller.tracker.abstract_tracker import AbstractTracker
from odyssey.components.agents.controller.tracker import tracker_utils

from odyssey.utils import math_utils
from odyssey.utils.cadence import CONTROL_DT, HEADING_CHORD_M
from odyssey.components.agents.policy.pdm_planner.reference_config import (
    LQR_Q_LATERAL,
    LQR_Q_LONGITUDINAL,
    LQR_R_LATERAL,
    LQR_R_LONGITUDINAL,
)


class LateralStateIndex(IntEnum):
    """
    Index for solving lateral state transformation equation.
    """

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


logger = logging.getLogger(__name__)

class LQRTracker(AbstractTracker):
    #: one-shot, so a producer without headings says so once, not every step
    _warned_no_headings = False

    """
    Forked from: https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py

    Implements an LQR tracker for a kinematic bicycle model.

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(self, agent,):
        """
        Constructor for LQR controller
        agent.config includes:
            - q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
            - r_longitudinal: The weights for the R matrix for the longitudinal subystem.
            - q_lateral: The weights for the Q matrix for the lateral subystem.
            - r_lateral: The weights for the R matrix for the lateral subystem.
            - tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
            - stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
            - stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        """
        super(LQRTracker, self).__init__(agent=agent)

        self.config = self.agent.config

        # hyperparameters for LQR controller.
        q_longitudinal = np.array(self.config['q_longitudinal'])
        r_longitudinal = np.array(self.config['r_longitudinal'])
        q_lateral = np.array(self.config['q_lateral'])
        r_lateral = np.array(self.config['r_lateral'])
        if (self.config.get('ego_policy') == 'pdm_policy'
                and self.config.get('pdm_reference_lqr', True)):
            # PDM is a reference baseline in this repository. Its trajectory should be realized
            # by nuPlan's published LQR tuning, not by the high-lateral-gain tuning used for
            # unrelated learned-policy experiments. The switch is PDM-only so those experiments
            # keep their established defaults, and may be disabled for an explicit ablation.
            q_longitudinal = np.array(LQR_Q_LONGITUDINAL)
            r_longitudinal = np.array(LQR_R_LONGITUDINAL)
            q_lateral = np.array(LQR_Q_LATERAL)
            r_lateral = np.array(LQR_R_LATERAL)
            logger.info(
                "[PDM] using nuPlan reference LQR tuning q_long=%s r_long=%s "
                "q_lat=%s r_lat=%s",
                list(LQR_Q_LONGITUDINAL), list(LQR_R_LONGITUDINAL),
                list(LQR_Q_LATERAL), list(LQR_R_LATERAL))

        # Longitudinal LQR Parameters
        assert isinstance(q_longitudinal, np.ndarray) and len(q_longitudinal) == 1, \
            "q_longitudinal should be 1 float ndarray (velocity)."
        assert isinstance(r_longitudinal, np.ndarray) and len(r_longitudinal) == 1, \
            "r_longitudinal should be 1 float ndarray (acceleration)."
        self._q_longitudinal = np.diag(q_longitudinal)
        self._r_longitudinal = np.diag(r_longitudinal)

        # Lateral LQR Parameters
        assert isinstance(q_lateral, np.ndarray) and len(q_lateral) == 3, \
            "q_lateral should be 3 element-float ndarray (lateral_error, heading_error, steering_angle)."
        assert isinstance(r_lateral, np.ndarray) and len(r_lateral) == 1, \
            "r_lateral should be 1 element-float ndarray (steering_rate)."
        self._q_lateral = np.diag(q_lateral)
        self._r_lateral = np.diag(r_lateral)

        # assert all cost element in Q and R are positive definite.
        for attr in ["_q_lateral", "_q_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) >= 0.0), f"self.{attr} must be positive semi-definite."

        for attr in ["_r_lateral", "_r_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) > 0.0), f"self.{attr} must be positive definite."

        # Simulator related parameters for LQR
        # Note we want a horizon > 1 so that steering rate actually can impact lateral/heading error in discrete time.
        # The controller period is a constant, not a knob: making it configurable would give the
        # same value two spellings (CONTROL_DT and a config key) that could disagree.
        #
        # NB the PDM planner has its own `discretization_time`. That is a DIFFERENT clock --
        # its proposal simulator's, set from its own constructor default (batch_lqr.py:78),
        # never from this config. It is untouched.
        tracking_horizon = self.config['tracking_horizon']
        assert (
            tracking_horizon > 1
        ), "We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration."
        self._control_dt = CONTROL_DT
        self._tracking_horizon = tracking_horizon  # look ahead horizon for tracking action.
        self._wheel_base = self.agent.vehicle.wheel_base

        # Velocity/Curvature Estimation Parameters
        jerk_penalty = float(self.config['jerk_penalty']) # acceleration penalty.
        curvature_rate_penalty = float(self.config['curvature_rate_penalty'])  # steering rate penalty.
        assert jerk_penalty > 0.0, "The jerk penalty must be positive."
        assert curvature_rate_penalty > 0.0, "The curvature rate penalty must be positive."
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        stopping_proportional_gain = self.config['stopping_proportional_gain']
        stopping_velocity = self.config['stopping_velocity']  # stopping threshold.
        assert stopping_proportional_gain > 0, "stopping_proportional_gain has to be greater than 0."
        assert stopping_velocity > 0, "stopping_velocity has to be greater than 0."
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def _prepare_reference(self):
        """Build this step's reference and project the current tracking error onto it.

        Factored out of track_trajectory so MPCTracker can reuse it VERBATIM. The two trackers
        must differ only in the solve; if they also built their references differently, an A/B
        between them would not be attributable. Behaviour is unchanged for the LQR.

        :return: (initial_velocity, initial_lateral_state_vector), or None when the reference is
            un-buildable -- the caller then holds its last command (see the sanitisation note).
        """
        trajectory = self.agent.trajectory

        # transform waypoints to rear-axle coordinates.
        centered_waypoints = trajectory.waypoints
        centered_headings = trajectory.headings
        rear_waypoints = self.agent.rear_vehicle.get_rear_trajectory(centered_waypoints, centered_headings)
        self.waypoints = rear_waypoints

        if len(self.waypoints) == 0:
            raise ValueError("The waypoints should not be empty.")

        # Odyssey closed-loop robustness: at the tail of a long block rollout the planner can emit a
        # degenerate reference trajectory that leaves nothing to interpolate. Hold the last
        # valid command in that case so the rollout degrades gracefully instead of aborting the
        # ENTIRE scenario and discarding every step already simulated.
        #
        # Only non-finite poses are removed. Dropping near-duplicate waypoints would renumber
        # the time axis and pull the rest of the trajectory earlier; a fully degenerate
        # reference is handled by the except below, which holds the last command.
        #
        # The producer declares the spacing it built; the tracker does not look it up and does
        # not assume. Producers emit different spacings on purpose (the planner its wire grid, a
        # log replay the scene's), so the value is carried, not fixed.
        _wp_dt = getattr(trajectory, 'wp_dt', None)
        if not _wp_dt:
            raise ValueError(
                f"{type(trajectory).__name__} from {type(self.agent).__name__} did not declare "
                f"wp_dt. The tracker maps waypoint index to time and cannot guess the spacing; "
                f"guessing it wrong is what made the ego crawl. Set wp_dt where the trajectory "
                f"is built.")
        _wp_dt = float(_wp_dt)
        try:
            _wp = np.asarray(self.waypoints, dtype=float)
            _finite = np.isfinite(_wp).all(axis=1)
            if _finite.sum() < 2:
                raise ValueError("degenerate reference (< 2 finite waypoints)")
            self.waypoints = _wp[_finite]
            self._waypoint_times = np.flatnonzero(_finite).astype(float) * _wp_dt
            # The producer's own heading, as upstream does it. nuplan's tracker reads the
            # reference pose straight off the trajectory -- tracker_utils.py:372-374,
            # `[*state.rear_axle]` is (x, y, heading) -- and feeds that heading into the
            # velocity fit (`heading_profile=poses[:-1, 2]`) and the curvature profile. It
            # never reconstructs it from geometry: a heading read off >= 1 m chords cuts the
            # corners of a piecewise-linear plan and lags the true direction.
            _h = np.asarray(centered_headings, dtype=float) if centered_headings is not None else None
            if _h is not None and len(_h) == len(_wp):
                self._waypoint_headings = np.unwrap(_h[_finite])
            else:
                # The producer under-specified its plan. Recover a heading from the geometry
                # so the rollout continues, and say so once -- the recovered value is about a
                # degree off in the median and four at p90 against the planners that do emit
                # one, and that error lands straight in the lateral LQR.
                if not LQRTracker._warned_no_headings:
                    LQRTracker._warned_no_headings = True
                    logger.warning(
                        "%s supplied %s headings for %d waypoints; recovering them from the "
                        "waypoint geometry (%.1f m chord). A trajectory carries the orientation "
                        "its producer planned -- set Trajectory.headings there instead.",
                        type(trajectory).__name__,
                        "no" if _h is None else len(_h), len(_wp), HEADING_CHORD_M)
                self._waypoint_headings = tracker_utils.headings_from_waypoints(
                    self.waypoints)
        except (AssertionError, ValueError, IndexError):
            return None
        self._start_time = 0
        self._end_time = float(self._waypoint_times[-1])

        return self._compute_initial_velocity_and_lateral_state()

    def track_trajectory(self):
        """Inherited, see superclass."""
        prepared = self._prepare_reference()
        if prepared is None:
            return getattr(self, "_last_command", (0.0, 0.0))
        initial_velocity, initial_lateral_state_vector = prepared

        # Compute the velocity and curvature profile,
        #  This is optimized by minimal mean-square optimization.
        reference_velocity, reference_curvature = self._compute_reference_velocity_and_curvature_profile()

        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity

        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            # Do acceleration and steering command optimization.
            # This is achieved by LQR.
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = tracker_utils._generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=np.ones(self._tracking_horizon) * accel_cmd,
                discretization_time=self._control_dt,
            )[: self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(
                initial_lateral_state_vector,
                velocity_profile,
                reference_curvature,
            )
            
        # Return acceleration and steering command
        #  at the rear-axle.
        self._last_command = (accel_cmd, steering_rate_cmd)
        return accel_cmd, steering_rate_cmd

    def _compute_initial_velocity_and_lateral_state(self):
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        """
        initial_trajectory_waypoints = self.waypoints
        initial_trajectory_heading = float(self._waypoint_headings[0])

        # Determine initial error state.
        # position error.
        xy_error = self.agent.rear_vehicle.current_position - initial_trajectory_waypoints[0]

        # heading error.
        lateral_error = math_utils.rotate_points(
            np.array(xy_error).reshape(1, 2), 0, -initial_trajectory_heading)[0, 1]
        heading_error = math_utils.angle_diff(
            self.agent.rear_vehicle.current_heading, initial_trajectory_heading, 2 * np.pi)

        # Return initial velocity and lateral state vector.
        cur_velocity = self.agent.rear_vehicle.current_velocity
        cur_speed = math_utils.norm(cur_velocity[0], cur_velocity[1])

        initial_lateral_state_vector = np.array(
            [
                lateral_error,
                heading_error,
                self.agent.current_tire_steering,
            ],
        )

        return cur_speed, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_profile(self):
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._control_dt.
        :param current_iteration: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """
        times_s, pos_s, heading_s = tracker_utils.get_interpolated_reference_trajectory_poses(
            planning_waypoints=self.waypoints,
            waypoint_headings=self._waypoint_headings,
            waypoint_times=self._waypoint_times,
            discretization_time=self._control_dt,
            start_time=self._start_time,
            end_time=self._end_time)

        (
            velocity_profile,  # optimized velocity at each timestamp.
            acceleration_profile,  # optimized acceleration at each timestamp.
            curvature_profile,  # optimized curvature at each timestamp.
            curvature_rate_profile,  # optimized curvature_rate at each timestamp.
        ) = tracker_utils.get_velocity_curvature_profiles_with_derivatives_from_poses(
            discretization_time=self._control_dt,
            pos_s=pos_s,
            heading_s=heading_s,
            jerk_penalty=self._jerk_penalty,
            curvature_rate_penalty=self._curvature_rate_penalty,
        )

        reference_time = self._start_time + self._tracking_horizon * self._control_dt
        reference_velocity = np.interp(reference_time, times_s[:-1], velocity_profile)

        profile_times = [
            self._start_time + x * self._control_dt for x in range(self._tracking_horizon)
        ]
        reference_curvature_profile = np.interp(profile_times, times_s[:-1], curvature_profile)

        return float(reference_velocity), reference_curvature_profile

    def _stopping_controller(self, initial_velocity: float, reference_velocity: float):
        """
        Apply proportional controller when at near-stop conditions.

        Args:
            initial_velocity: [m/s] The current velocity of ego.
            reference_velocity: [m/s] The reference velocity to track.

        Return:
            Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocity - reference_velocity)
        return accel, 0.0

    def _longitudinal_lqr_controller(self, initial_velocity: float, reference_velocity: float):
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.

        Args:
            initial_velocity: [m/s] The current velocity of ego.
            reference_velocity: [m/s] The reference_velocity to track at a lookahead time.

        Return:
            Acceleration [m/s^2] command based on LQR.
        """
        # We assume that we hold the acceleration constant for the entire tracking horizon.
        # Given this, we can show the following where N = self._tracking_horizon and dt = self._control_dt:
        # velocity_N = velocity_0 + (N * dt) * acceleration (transformation function in LQR)
        # Thus: A = 1
        # B = N * dt
        # g = 0
        A = np.array([1.0], dtype=np.float32)
        B = np.array([self._tracking_horizon * self._control_dt], dtype=np.float32)

        accel_cmd = self._solve_one_step_lqr(
            initial_state=np.array([initial_velocity], dtype=np.float32),
            reference_state=np.array([reference_velocity], dtype=np.float32),
            Q=self._q_longitudinal,
            R=self._r_longitudinal,
            A=A,
            B=B,
            g=np.zeros(1, dtype=np.float32),
            angle_diff_indices=[],
        )

        return float(accel_cmd)

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: np.ndarray,
        velocity_profile: np.ndarray,
        curvature_profile: np.ndarray) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.

        Args:
            initial_lateral_state_vector: The current lateral state of ego.
            velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
            curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..

        Return:
            Steering rate [rad/s] command based on LQR.
        """
        assert len(velocity_profile) == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert len(curvature_profile) == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N = A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)
        I = np.eye(n_lateral_states, dtype=np.float32)

        A = I
        B = np.zeros((n_lateral_states, 1), dtype=np.float32)
        g = np.zeros(n_lateral_states, dtype=np.float32)

        # Convenience aliases for brevity.
        idx_lateral_error = LateralStateIndex.LATERAL_ERROR
        idx_heading_error = LateralStateIndex.HEADING_ERROR
        idx_steering_angle = LateralStateIndex.STEERING_ANGLE

        input_matrix = np.zeros((n_lateral_states, 1), np.float32)
        input_matrix[idx_steering_angle] = self._control_dt

        for index_step, (velocity, curvature) in enumerate(zip(velocity_profile, curvature_profile)):
            state_matrix_at_step = np.eye(n_lateral_states, dtype=np.float32)
            state_matrix_at_step[idx_lateral_error, idx_heading_error] = velocity * self._control_dt
            state_matrix_at_step[idx_heading_error, idx_steering_angle] = (
                velocity * self._control_dt / self._wheel_base
            )

            affine_term = np.zeros(n_lateral_states, dtype=np.float32)
            affine_term[idx_heading_error] = -velocity * curvature * self._control_dt

            A = state_matrix_at_step @ A
            B = state_matrix_at_step @ B + input_matrix
            g = state_matrix_at_step @ g + affine_term

        steering_rate_cmd = self._solve_one_step_lqr(
            initial_state=initial_lateral_state_vector,
            reference_state=np.zeros(n_lateral_states, dtype=np.float64),
            Q=self._q_lateral,
            R=self._r_lateral,
            A=A,
            B=B,
            g=g,
            angle_diff_indices=[idx_heading_error, idx_steering_angle],
        )

        return float(steering_rate_cmd)

    @staticmethod
    def _solve_one_step_lqr(
        initial_state: np.ndarray,
        reference_state: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        g: np.ndarray,
        angle_diff_indices: List[int] = [],):
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param Q: The state tracking 2-norm cost matrix.
        :param R: The input 2-norm cost matrix.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :param angle_diff_indices: The set of state indices for which we need to apply angle differences, if defined.
        :return: LQR optimal input for the 1-step problem.
        """
        state_error_zero_input = A @ initial_state + g - reference_state

        for angle_diff_index in angle_diff_indices:
            state_error_zero_input[angle_diff_index] = math_utils.angle_diff(
                state_error_zero_input[angle_diff_index], 0.0, 2 * np.pi
            )

        lqr_input = -np.linalg.inv(B.T @ Q @ B + R) @ B.T @ Q @ state_error_zero_input
        return lqr_input
