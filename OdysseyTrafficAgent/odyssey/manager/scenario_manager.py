# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import copy
import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2

from odyssey.manager.base_manager import BaseManager

from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.utils.cadence import CONTROL_DT, resolve_cadence
from odyssey.manager import signal_patch, tlc_timetable_set

import logging
logger = logging.getLogger(__name__)


def _angle_lerp(a, b, t):
    """Shortest-arc interpolation between headings a->b at fraction t (elementwise)."""
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + t * d


def _resample_scene(scene, S):
    """Upsample every time-indexed array in `scene` by an integer factor S (insert S-1 frames
    between each original pair, endpoints preserved) so the simulator can step at native_dt / S.

    Used for the fine-grained (e.g. 0.1 s) closed loop: GT agents/ego then move every sub-step
    instead of teleporting once per native step. Preserves the ScenarioDescription invariant that
    every per-frame state array is zeroed wherever valid == 0, and holds piecewise-constant signals
    (traffic-light state) with a nearest-floor sample. `sample_rate` is divided by S so that
    sim_dt (= sample_rate * 0.05) reflects the new sub-step. S <= 1 is a no-op.
    """
    if S <= 1:
        return scene

    def _src_index(n):
        new_n = (n - 1) * S + 1
        idx = np.arange(new_n) / float(S)          # fractional position in the original timeline
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        return new_n, i0, i1, idx - i0             # (new_len, lo, hi, frac)

    # --- dynamic agents + ego: interpolate pose/velocity, AND the validity, zero invalid frames ---
    for obj in scene[SD.OBJECT_TRACKS].values():
        st = obj[SD.STATE]
        valid = np.asarray(st[SD.VALID]).astype(float)
        n = len(valid)
        if n < 2:
            continue
        _, i0, i1, frac = _src_index(n)
        # a sub-frame is valid only if BOTH bracketing originals are valid; exact originals (frac==0)
        # keep their own validity so the endpoints are untouched.
        new_valid = np.where(frac == 0.0, valid[i0], valid[i0] * valid[i1])
        mask = new_valid > 0
        new_st = {}
        for key, arr in st.items():
            a = np.asarray(arr)
            if key == SD.VALID:
                new_st[key] = new_valid.astype(a.dtype)
                continue
            if key == SD.HEADING:
                out = _angle_lerp(a[i0], a[i1], frac)
            else:                                   # position/velocity/length/width/height: linear
                f = frac[:, None] if a.ndim > 1 else frac
                out = (1.0 - f) * a[i0] + f * a[i1]
            out[~mask] = 0.0                         # keep "invalid frames are zeroed" invariant
            new_st[key] = out
        obj[SD.STATE] = new_st

    # --- traffic lights: piecewise-constant, hold each state across its S sub-frames ---
    for tl in scene.get(SD.DYNAMIC_MAP_STATES, {}).values():
        tstate = tl.get(SD.STATE)
        inner = tstate.item() if isinstance(tstate, np.ndarray) and tstate.ndim == 0 else tstate
        if isinstance(inner, dict) and SD.TRAFFIC_LIGHT_STATES in inner:
            tls = np.asarray(inner[SD.TRAFFIC_LIGHT_STATES])
            n = len(tls)
            if n >= 2:
                new_n = (n - 1) * S + 1
                inner[SD.TRAFFIC_LIGHT_STATES] = tls[np.minimum(np.arange(new_n) // S, n - 1)]

    scene['sample_rate'] = scene['sample_rate'] / float(S)
    # log_length gates frame emission (DataManager: episode_step < log_length). The rollout now runs
    # S x more steps, so scale it too; the openscene frame TEMPLATES are left at the original count
    # and DataManager maps each sub-step to its containing original frame via episode_step // S.
    if scene.get(SD.LENGTH):
        scene[SD.LENGTH] = int(scene[SD.LENGTH]) * S
    return scene


class ScenarioManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100
    PRIORITY = -10

    def __init__(self, info_dict=None):
        super(ScenarioManager, self).__init__()
        scenes = info_dict
        num_scenes = len(scenes)
        logger.info(f"Loading {num_scenes} scenarios")

        # Fine-grained closed loop: when the outer step is finer than the scenario's own frame
        # spacing, upsample every GT track so agents/ego move each sub-step. Done once here,
        # before SD construction/centralize/summaries, so all downstream managers see the dense
        # track and sim_dt reflects the resolved step.
        #
        # resolve_cadence is the only place the step is derived, and it refuses rather than
        # rounds: an outer step that does not divide the scene's frames, or the controller
        # period, or into the scoring grid, is a configuration that cannot be run correctly.
        # It also checks the scene's declared sample_rate against its own timestamps -- that
        # one field sets every duration in the run.
        self.scenes = dict()
        self.scenes_ids = dict()
        for idx, (scene_id, scene) in enumerate(scenes.items()):
            cadence = resolve_cadence(self.global_config, scene, scene_id)
            # Store this scene's cadence on the scene. _resample_scene divides sample_rate by S,
            # so calling resolve_cadence again later always yields upsample_n == 1 -- this value
            # can only be saved now. DataManager reads it when building the openscene template
            # index; re-deriving it there as log_length // number of templates would equal S only
            # if log_length == number of templates, a condition nothing checks.
            scene["cadence"] = cadence
            if cadence.upsample_n > 1:
                _resample_scene(scene, cadence.upsample_n)
                if idx == 0:
                    logger.info(
                        "[CADENCE] outer step %.4fs: upsampled GT tracks x%d "
                        "(scene frames %.4fs) | plant %d sub-steps of %.4fs | "
                        "plan stride %d | scoring every %d steps",
                        cadence.sim_dt, cadence.upsample_n, cadence.scene_dt,
                        cadence.plant_substeps, CONTROL_DT, cadence.step_stride,
                        cadence.score_stride_steps)
            # Opt-in (tl_signal_patch_path, or the tlc_timetable_set preset; both default null): camera-verified
            # signal timetable written into this scene's traffic_light_state arrays -- the one table every signal
            # consumer reads (planner, IDM, rewards, metric, TLC snapshot, tl_control). No-op when both are unset.
            tlc_timetable_set.log_scene(self.global_config, scene_id)
            signal_patch.apply_configured(self.global_config, scene_id, scene, cadence.upsample_n)
            scene["route_var_name"] = None
            scene["trigger_points"] = []
            scene["scenarios"] = [
                {
                "name": "Freeride",
                "trigger_points": [StateSE2(1, 1, 0)]
                },
            ]

            converted_scene = SD.centralize_to_ego_car_initial_position(SD(scene))
            converted_scene = SD.update_summaries(converted_scene)
            self.scenes[scene_id] = converted_scene
            self.scenes_ids[idx] = scene_id

        self.num_scenarios = num_scenes
        # scenarios for this worker.
        self.available_scenario_indices = [
            i for i in range(self.num_scenarios)
        ]

        # stat
        self.coverage = [0 for _ in range(self.num_scenarios)]

    # some properties of the class.
    @property
    def current_scene_summary(self):
        return self.current_scene[SD.SUMMARY.SUMMARY]

    @property
    def current_scene_length(self):
        return self.current_scene[SD.LENGTH]

    @property
    def current_scene_index(self):
        idx = self.engine.global_random_seed
        assert idx in self.available_scenario_indices, \
            "scenario index exceeds range, scenario index: {}".format(idx)
        return idx

    @property
    def current_scene_id(self):
        index = self.current_scene_index
        return self.scenes_ids[index]

    @property
    def current_scene(self):
        index = self.current_scene_index
        return self.get_scene(index)

    def _get_scene(self, i):
        assert i in self.available_scenario_indices, \
            "scenario index exceeds range, scenario index: {}".format(i)
        scenario_id = self.scenes_ids[i]
        ret = self.scenes[scenario_id]
        assert isinstance(ret, SD)
        return ret

    def get_scene(self, i, should_copy=False):
        ret = self._get_scene(i)
        self.coverage[i] = 1
        if should_copy:
            return copy.deepcopy(ret)
        return ret

    @property
    def data_coverage(self):
        return sum(self.coverage) / len(self.coverage)

    @property
    def all_scenes_completed(self):
        return all(self.coverage)

    def before_step(self):
        pass

    def step(self):
        step_info = {}
        return step_info

    def clear_stored_scenarios(self):
        self.scenes = {}
        self.scenes_ids = {}

    def filter_short_scenarios(self, min_length: int):
        """Filter out scenarios whose log_length is shorter than min_length."""
        original_count = len(self.available_scenario_indices)
        remaining_indices = []
        for idx in self.available_scenario_indices:
            scene_id = self.scenes_ids[idx]
            scene = self.scenes[scene_id]
            log_length = scene.get(SD.LENGTH, float('inf'))
            if log_length < min_length:
                logger.warning(
                    f"Skipping scenario {scene_id}: log_length ({log_length}) < required obs_len ({min_length})"
                )
                self.coverage[idx] = 1
            else:
                remaining_indices.append(idx)

        skipped_count = original_count - len(remaining_indices)
        if skipped_count > 0:
            logger.warning(f"Filtered out {skipped_count} short scenarios, {len(remaining_indices)} remaining.")
            self.available_scenario_indices = remaining_indices
            if len(remaining_indices) > 0:
                self.engine.seed(remaining_indices[0])

    def reset(self):
        reset_info = {}
        return reset_info

    def after_reset(self):
        reset_info = {}
        return reset_info

    def get_metadata(self):
        state = super().get_metadata()
        raw_scene_data = self.current_scene
        state["raw_scene_data"] = raw_scene_data
        return state

    def destroy(self):
        """
        Clear memory
        """
        super().destroy()
        self.clear_stored_scenarios()

    def next_scene(self):
        """
        Switch to the next available scene
        """
        current_idx = self.current_scene_index

        current_position = self.available_scenario_indices.index(current_idx)

        next_position = (current_position + 1) % len(self.available_scenario_indices)

        self.engine.seed(self.available_scenario_indices[next_position])
