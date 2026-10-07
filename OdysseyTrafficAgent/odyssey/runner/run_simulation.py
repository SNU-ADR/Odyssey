# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import logging
import os
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig

from odyssey.runner.builders.env_builder import build_envs
from odyssey.runner.executor import run_runners
from odyssey.utils.cadence import (
    CONTROL_DT, PLANNER_POSE_DT, SUBSTEP_KEY_BY_CONTROLLER, resolve_plant_substeps, scene_dt)

logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = '../configs'
CONFIG_NAME = 'default_runner'


def _log_cadence_check(cfg: DictConfig, scenes_dict: dict) -> None:
    """Print every stage's tick period, and shout if the plant is not at CONTROL_DT.

    Two things have to hold:

        (1) the ego advances one outer step's worth of time per outer step
        (2) the plant integrates at the controller period
                outer step / substeps == CONTROL_DT

    (1) is structural: the motion model takes its dt from the engine's outer step, so
    there is no second key to leave stale. This function checks (2).

    Log-only on purpose; resolve_cadence is where this becomes an error.
    """
    ego_controller = str(cfg.get('ego_controller', 'two_stage_controller') or 'two_stage_controller')
    substep_key = SUBSTEP_KEY_BY_CONTROLLER.get(ego_controller)

    # The outer tick is rollout_dt when set. When it is not, the world advances one scene
    # frame per step, so the scene's own spacing is the tick; a fixed 0.5 would be wrong for
    # every scene whose sample_rate is not 10.
    rollout_dt = cfg.get('rollout_dt', None)
    if rollout_dt:
        outer_tick, tick_src = float(rollout_dt), 'rollout_dt'
    else:
        scene_dts = sorted({round(scene_dt(sc), 6) for sc in scenes_dict.values()
                            if 'sample_rate' in sc})
        if len(scene_dts) == 1:
            outer_tick, tick_src = scene_dts[0], 'scene sample_rate'
        elif scene_dts:
            outer_tick, tick_src = scene_dts[0], f'scene sample_rate (MIXED: {scene_dts})'
        else:
            outer_tick, tick_src = PLANNER_POSE_DT, 'fallback -- no sample_rate in any scene'

    # The plant advances exactly one outer step per outer step -- the motion model takes its
    # dt from the engine, so the two cannot disagree any more. What can still be wrong is the
    # sub-step count, which decides whether the plant integrates at the controller period.
    # Resolved exactly as the controller resolves it: an unset key DERIVES outer_tick /
    # CONTROL_DT. Reading it as 1 here reported a 0.5 s run as "1 sub-step of 0.5s" and raised a
    # false CADENCE MISMATCH while the controller was in fact integrating 5 x 0.1 s.
    try:
        substeps = (resolve_plant_substeps(cfg.get(substep_key), outer_tick, ego_controller)
                    if substep_key else 1)
    except ValueError:
        substeps = 1                     # not a whole number of control periods -> flagged below
    substep_dt = outer_tick / substeps
    logger.info(
        "[CADENCE] outer tick=%.4fs (%s) -> renderer / scenario replay / planner query | "
        "plant(%s) %d sub-steps of %.4fs | control_dt=%.4fs",
        outer_tick, tick_src, ego_controller, substeps, substep_dt, CONTROL_DT,
    )

    problems = []
    if abs(substep_dt - CONTROL_DT) > 1e-9:
        why = (f"the plant integrates at {substep_dt:.4f}s but the controller period is "
               f"{CONTROL_DT:.4f}s.")
        if substep_key:
            fix = (f"pass {substep_key}={outer_tick / CONTROL_DT:g} "
                   f"(= frame_rate/{CONTROL_DT:g})")
        else:
            fix = f"{ego_controller} has no sub-step key -- it cannot honour control_dt"
        problems.append(f"{why}\n!!!     FIX: {fix}")

    if problems:
        banner = "!" * 100
        body = "".join(f"!!! {p}\n" for p in problems)
        msg = (f"\n{banner}\n!!! CADENCE MISMATCH !!!\n{body}{banner}\n")
        logger.error(msg)
        print(msg, flush=True)


def _log_velocity_sanity_check(data_file_path: str, scenes_dict: dict) -> None:
    """Flag any object_track whose 'velocity' is entirely zero across its valid frames.

    Background: OmniRe-onboarded scenes (merge_chunks.py/patch_openscene_infos.py)
    inherited an all-zero 'velocity' array from DriveStudio's raw per-instance annotations
    (which only ever store `obj_to_world` pose, never velocity) -- base_agent.py seeds every
    agent's (ego included) initial `_cur_velocity` directly from this field, so a degenerate
    array there silently starts the ego's KBM integration from a false "at rest" state even
    when the scenario's real (nuPlan-sourced) speed is high. The real vis_tokens pipeline
    populates this correctly via extract_traffic() pulling from the nuPlan DB -- so a scene
    failing this check is a DATA problem with that specific scenario file, not a code path
    difference. Report the scenario's own path so the offending pkl is unambiguous.
    """
    for scene_id, scene in scenes_dict.items():
        object_track = scene.get('object_track') or {}
        sdc_id = scene.get('sdc_id')
        bad_tracks = []
        for track_id, track in object_track.items():
            state = track.get('state') or {}
            vel = state.get('velocity')
            valid = state.get('valid')
            if vel is None or valid is None:
                continue
            valid = np.asarray(valid).astype(bool).reshape(-1)
            vel = np.asarray(vel)
            if valid.sum() == 0:
                continue
            vel_valid = vel[valid]
            if np.allclose(vel_valid, 0):
                bad_tracks.append(track_id)
        if not bad_tracks:
            logger.info("[VELOCITY CHECK] scene=%s (%s): velocity field OK for all %d tracked object(s)",
                        scene_id, data_file_path, len(object_track))
            continue

        sdc_bad = sdc_id in bad_tracks
        other_bad = len(bad_tracks) - (1 if sdc_bad else 0)
        # A handful of genuinely-parked/static background objects reading exactly zero is normal
        # real data, not a bug -- only the SDC is guaranteed to be non-stationary-by-construction
        # (base_agent.py seeds ITS KBM integration from this field) and worth an alarm. Other
        # tracks get a quiet count instead of the banner, so a normal scene with parked cars
        # doesn't cry wolf on every single run.
        if other_bad:
            logger.info("[VELOCITY CHECK] scene=%s (%s): %d/%d non-SDC track(s) read exactly-zero "
                        "velocity (could be genuinely parked/static -- not alarmed on its own)",
                        scene_id, data_file_path, other_bad, len(object_track))
        if not sdc_bad:
            continue

        banner = "!" * 100
        msg = (
            f"\n{banner}\n"
            f"!!! VELOCITY FIELD ALL-ZERO ON THE SDC/EGO !!!  scene={scene_id}\n"
            f"!!! path: {data_file_path}\n"
            "!!! base_agent.py seeds the ego's initial _cur_velocity from this field -- the "
            "closed-loop rollout will start the ego's KBM integration from a FALSE at-rest "
            "state regardless of the scenario's true initial speed !!!\n"
            "!!! FIX: re-derive velocity for this scenario (e.g. pull real values from the "
            "nuPlan DB via nuplan_lidar_pc_tokens + extract_traffic()) "
            "before trusting this run's results !!!\n"
            f"{banner}\n"
        )
        logger.error(msg)
        print(msg, flush=True)


def _apply_restorer(cfg: DictConfig) -> None:
    """`restorer=fixer_h1b16|fixer_pretrained` picks the restorer applied to the cameras.

    The renderer reads the choice from ODYSSEY_RESTORER (restorer_client.install_from_env), so
    it is exported HERE, before any renderer exists. Each preset starts its Fixer worker.
    """
    from odyssey_renderer.omnire.restorer_client import RESTORER_CHOICES
    # `fixer_h1b16`, matching default_runner.yaml's own default (${oc.env:ODYSSEY_RESTORER,fixer_h1b16}).
    # This branch only fires when the key is explicitly null.
    choice = str(cfg.get('restorer') or 'fixer_h1b16').strip()
    if choice not in RESTORER_CHOICES and not choice.startswith('http'):
        raise ValueError('restorer must be one of %s (or an http url), got %r'
                         % ('|'.join(RESTORER_CHOICES), choice))
    os.environ['ODYSSEY_RESTORER'] = choice
    logger.info('restorer=%s', choice)
    # save_pre_restore: same route as the choice itself, for the same reason (set before the
    # renderer exists).
    save_pre = str(cfg.get('save_pre_restore', False)).strip().lower() in ('1', 'true')
    os.environ['ODYSSEY_SAVE_PRE_RESTORE'] = '1' if save_pre else '0'

    # render_cams: same route again. _select_cams() reads ODYSSEY_RENDER_CAMS, and a launcher that
    # exported it keeps winning -- the config default IS that env var, so overriding the key
    # on the command line is the only way this differs from today.
    # Both spellings: "cam_f0,cam_l0,cam_r0" (an exported ODYSSEY_RENDER_CAMS, which is a string) and
    # [cam_f0,cam_l0,cam_r0] (a command-line override -- hydra reads a bare comma list as a SWEEP
    # and refuses to run, so an override has to bracket it).
    _cams = cfg.get('render_cams', '')
    cams = (','.join(str(c) for c in _cams) if isinstance(_cams, (list, tuple, ListConfig))
            else str(_cams or '')).strip()
    if cams:
        os.environ['ODYSSEY_RENDER_CAMS'] = cams
        logger.info('render_cams: rendering only %s', cams)
    if save_pre:
        logger.info('save_pre_restore: unrestored cameras -> sensor_blobs_pre_restore/')


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info('Odyssey is running...')

    _apply_restorer(cfg)

    # Construct simulations/environments
    scenes_dict = pickle.load(open(cfg.data_file_path, 'rb'))
    # After the load: the outer tick falls back to the scene's own frame spacing when
    # rollout_dt is unset, so this check needs the scenes.
    _log_cadence_check(cfg, scenes_dict)
    _log_velocity_sanity_check(cfg.data_file_path, scenes_dict)
    envs = build_envs(cfg=cfg, scene_dict=scenes_dict)

    logger.info('Running simulation...')
    run_runners(envs=envs, cfg=cfg)
    logger.info('Finished running simulation!')

if __name__ == "__main__":
    main()
