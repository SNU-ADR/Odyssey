# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import numpy as np

from odyssey.components.agents.controller.motion_model.abstract_motion_model import AbstractMotionModel
from odyssey.components.agents.controller import controller_utils

from odyssey.utils import math_utils


class KinematicBicycleModel(AbstractMotionModel):
    """
    A class describing the kinematic motion model where the rear axle is the point of reference.
    """

    def __init__(self, agent,):
        super(KinematicBicycleModel, self).__init__(agent)

        self.config = self.agent.config

        # Seconds this model integrates per call. Initialised to the outer step, because the
        # ego must advance exactly one outer step's worth of time per outer step -- that was a
        # separate `frame_rate` config key, and leaving it at 0.5 under a 0.1 s rollout moved
        # the ego five times too far while every individual file still looked correct.
        #
        # Still an attribute rather than a property: the sub-stepping controllers scale it to
        # base/N, integrate N times and restore it, which is how the plant runs at CONTROL_DT
        # under a coarser outer step.
        self._step_dt = float(self.agent.engine.sim_dt)

        # low pass filter time constant for acceleration in s
        self._accel_time_constant = self.config.get('accel_time_constant')

        # low pass filter time constant for steering angle in s
        self._steering_angle_time_constant = self.config.get('steering_angle_time_constant')

        # FORK: bypass switches for the two first-order actuator lags.
        #
        # Both lags are the discrete form of  dx/dt = (cmd - x)/tau, i.e.
        #     x_new = x + dt/(dt + tau) * (cmd - x)
        # so tau -> 0 makes the gain dt/(dt+0) = 1 and x_new = cmd exactly. That identity is why
        # these are implemented as a flag that forces the gain to 1 rather than as "set tau very
        # small": a tiny-but-nonzero tau still leaves a dt-dependent residue, and the whole point
        # of the switch is to remove the plant's dynamics from the comparison entirely.
        #
        # WHY: under direct control the model's (accel, kappa) IS the command, so a rollout mixes
        # two effects -- what the head predicted, and what the plant's lag did to it. Disabling the
        # lag makes the ego track the commanded control instantly, which isolates the head. It is a
        # DIAGNOSTIC setting, not a more-realistic one: real actuators do lag, and navsim's
        # reference plant lags too, so scores from a no-lag run are not comparable to normal runs.
        self._no_accel_lag = bool(self.config.get('no_accel_lag', False))
        self._no_steering_lag = bool(self.config.get('no_steering_lag', False))

    def propagate_state(self, accel_cmd, steering_rate_cmd):
        """Inherited, see super class."""
        cur_accel_vector = self.agent.rear_vehicle.current_acceleration
        cur_steering_angle = self.agent.current_tire_steering

        # SIGNED longitudinal acceleration, recovered by projecting the (global-frame) acceleration
        # onto the heading. `rear_vehicle.current_acceleration` rotates back to global on the way out
        # (vehicle_utils.py), so neither the raw x component nor its magnitude is the longitudinal
        # value: raw x varies with heading and flips sign past +-pi/2, and a magnitude has no sign at
        # all. The lag below converges to the commanded value only when `a` is signed; fed a
        # magnitude, it would realise only a fraction of any commanded deceleration.
        #
        # This matches navsim's BatchKinematicBicycleModel, the reference plant the controller gains
        # were tuned against, which uses the signed body-frame ACCELERATION_X directly.
        _h = self.agent.rear_vehicle.current_heading
        cur_accel_value = (cur_accel_vector[0] * np.cos(_h)
                           + cur_accel_vector[1] * np.sin(_h))

        # no_accel_lag: gain 1, i.e. the commanded acceleration is realised this step (see __init__).
        if self._no_accel_lag:
            updated_accel_value = accel_cmd
        else:
            updated_accel_value = (
                self._step_dt / (self._step_dt + self._accel_time_constant) *
                (accel_cmd - cur_accel_value) + cur_accel_value
            )

        # The LQR returns a steering RATE [rad/s]; it has to be integrated into an angle before
        # the first-order steering lag is applied. nuplan-devkit's KinematicBicycleModel does
        #     ideal_steering_angle = dt_control * tire_steering_rate + steering_angle
        #     updated = dt/(dt+tau) * (ideal_steering_angle - steering_angle) + steering_angle
        # This fork dropped the integration and fed steering_rate_cmd straight in where the
        # ideal ANGLE belongs, i.e. it compared a rad/s value against a rad value -- a
        # dimensional error that drives the steering toward the wrong target every step.
        ideal_steering_angle = self._step_dt * steering_rate_cmd + cur_steering_angle
        # no_steering_lag: gain 1, so the angle reaches its ideal target this step. Only the LAG is
        # removed -- the rate is still integrated into an angle first, because that integration is
        # the unit conversion (rad/s -> rad), not part of the filter.
        if self._no_steering_lag:
            updated_steering_angle = ideal_steering_angle
        else:
            updated_steering_angle = (
                self._step_dt / (self._step_dt + self._steering_angle_time_constant) *
                (ideal_steering_angle - cur_steering_angle) + cur_steering_angle
            )
        updated_steering_rate = (updated_steering_angle - cur_steering_angle) / self._step_dt

        # Update the state.
        x_dot = self.agent.rear_vehicle.current_velocity[0]
        y_dot = self.agent.rear_vehicle.current_velocity[1]
        longitudinal_speed = math_utils.norm(x_dot, y_dot)
        yaw_dot = longitudinal_speed * np.tan(self.agent.current_tire_steering) / self.agent.vehicle.wheel_base

        ########## Then update the pos / heading / vel / acc / angular_vel / angular_acc #######
        new_rear_pos = controller_utils.forward_integrate(
            np.array(self.agent.rear_vehicle.current_position),
            np.array([x_dot, y_dot]),
            self._step_dt)

        new_rear_heading = math_utils.principal_value(controller_utils.forward_integrate(
            self.agent.rear_vehicle.current_heading,
            yaw_dot, self._step_dt))

        # No speed in the Lateral dimension in the Kinematic Bicycle model.
        new_rear_longitudinal_speed = controller_utils.forward_integrate(
            longitudinal_speed, updated_accel_value, self._step_dt)
        new_rear_velocity = [new_rear_longitudinal_speed, 0]
        new_rear_acceleration = [updated_accel_value, 0]

        new_tire_steering = np.clip(
            controller_utils.forward_integrate(
                cur_steering_angle, updated_steering_rate, self._step_dt
            ),
            -self.agent.MAX_STEERING,
            self.agent.MAX_STEERING
        )

        new_angular_velocity = (
            new_rear_longitudinal_speed * np.tan(new_tire_steering) / self.agent.vehicle.wheel_base
        )
        new_angular_acc = (
            (new_angular_velocity - self.agent.rear_vehicle.current_angular_velocity) / self._step_dt
        )

        self.agent.rear_vehicle.update_center_agent(
            new_rear_pos=new_rear_pos,
            new_rear_heading=new_rear_heading,
            new_rear_velocity=new_rear_velocity,
            new_rear_acceleration=new_rear_acceleration,
            new_rear_angular_velocity=new_angular_velocity,
            new_rear_angular_acc=new_angular_acc,
            new_tire_steering=new_tire_steering,
            new_action=[steering_rate_cmd, accel_cmd],
        )
