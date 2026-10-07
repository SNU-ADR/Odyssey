# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.controller.abstract_controller import AbstractController
from odyssey.components.agents.controller.motion_model.build_motion_model import build_motion_model
from odyssey.utils.cadence import resolve_plant_substeps
from odyssey.components.agents.controller.tracker.build_tracker import build_tracker
from odyssey.engine.engine_utils import get_engine


class TwoStageController(AbstractController):
    """
    Implements a two stage tracking controller. The two stages comprises of:
        1. an AbstractTracker - This is to simulate a low level controller layer that is present in real AVs.
        2. an AbstractMotionModel - Describes how the AV evolves according to a physical model.
    """

    def __init__(self, agent):
        """
        Constructor for TwoStageController
        :param scenario: Scenario
        :param tracker: The tracker used to compute control actions
        :param motion_model: The motion model to propagate the control actions
        """
        super(TwoStageController, self).__init__(agent)

        self._tracker = build_tracker(self.agent.config)(self.agent)
        self._motion_model = build_motion_model(self.agent.config)(self.agent)

        # LQR control-period fix (two_stage_substeps). The LQR gains are tuned for a 0.1s loop, but the
        # default single frame_rate (0.5s) Euler step both integrates AND refreshes the command 5x too
        # coarsely. N>1 sub-steps the tracker+bicycle at frame_rate/N, N times per outer step, so the
        # 0.1s-tuned gains run at the cadence they were designed for. N=1 => original behaviour exactly.
        # See default_runner.yaml and pure_pursuit's pp_substeps (same idea, transparent controller).
        self._substeps = resolve_plant_substeps(
            self.agent.config.get('two_stage_substeps'),
            self._motion_model._step_dt, 'two_stage_controller')

    def step(self):
        """Inherited, see superclass."""
        if self.agent.trajectory is None:
            # No plan this step: hold the last low-level action, single native-rate integration.
            accel_cmd, steering_rate_cmd = self.agent.lower_action
            self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)
            return

        # Consume one planner waypoint for this outer step -- ONCE, not per sub-step (the plan is
        # fixed for the whole 0.5s window; only the ego pose advances between sub-steps).
        for attr in ['waypoints', 'velocities', 'headings', 'angular_velocities']:
            if hasattr(self.agent.trajectory, attr):
                value = getattr(self.agent.trajectory, attr)
                if isinstance(value, list) and len(value) > 1:
                    setattr(self.agent.trajectory, attr, value[1:])

        if self._substeps == 1:
            # Original path, untouched: one LQR solve + one frame_rate (0.5s) Euler step.
            accel_cmd, steering_rate_cmd = self._tracker.track_trajectory()
            self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)
            return

        # Sub-stepped path: re-solve the LQR against the fixed plan from the advancing ego and
        # integrate the bicycle at frame_rate/N, N times -- the 0.1s cadence the gains assume. The
        # motion model reads its dt from _step_dt each call, so we scale it for the loop and
        # restore it afterwards (finally) so nothing leaks if the tracker raises mid-loop.
        mm = self._motion_model
        base_step_dt = mm._step_dt
        try:
            mm._step_dt = base_step_dt / self._substeps
            for _ in range(self._substeps):
                accel_cmd, steering_rate_cmd = self._tracker.track_trajectory()
                mm.propagate_state(accel_cmd, steering_rate_cmd)
        finally:
            mm._step_dt = base_step_dt
