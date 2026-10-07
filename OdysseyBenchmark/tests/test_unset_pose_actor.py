"""Actors that arrive at the origin without a pose are excluded from scoring.

An actor whose source log has ended can carry local coordinates (0,0) for one frame just before
it disappears. denormalize_from_ego_center maps that to initial_ego_center, where the rollout
started; early on the ego is still there, so **an actor that is not on screen is recorded inside
the ego** and the scorer counts a contact the ego could not have avoided.

Pinned here:
  1. An actor carrying the local origin is not added to DetectionsTracks.
  2. Actors off the origin are kept, including parked vehicles with zero speed (they outlive their source).
  3. The number dropped is recorded in visibility_stats (reported in one line at the end of the run).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("nuplan")
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (  # noqa: E402
    OdysseyToNuPlanConverter)

EGO_CENTER = np.array([664481.358, 3997877.112])


def _agent(obj_id, xy, velocity=(0.0, 0.0)):
    return SimpleNamespace(id=obj_id, current_position=np.asarray(xy, dtype=float),
                           current_heading=0.25,
                           current_velocity=np.asarray(velocity, dtype=float),
                           _length=0.8, _width=0.8, _height=1.7)


def _converter(agents):
    conv = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    conv.scene = {
        "token": "tok", "sdc_id": "ego",
        # Set once by ScenarioManager. Without it _sim_dt fails instead of guessing.
        "cadence": SimpleNamespace(sim_dt=0.1),
        "object_track": {a.id: {"type": "PEDESTRIAN"} for a in agents},
    }
    conv.base_timestamp = 0.0
    conv.initial_ego_center = EGO_CENTER
    conv.engine = SimpleNamespace(
        agent_manager=SimpleNamespace(all_agents={a.id: a for a in agents}),
        managers={})
    return conv


def _tokens(tracks):
    return {o.metadata.track_token for o in tracks.tracked_objects.tracked_objects}


def test_origin_pose_actor_is_dropped():
    # ghost: local origin, which maps back to exactly where the ego started.
    agents = [_agent("ghost", [0.0, 0.0]), _agent("real", [12.0, -3.0])]
    conv = _converter(agents)
    tracks = conv.convert_to_detections_tracks_from_agent_input(2)
    assert _tokens(tracks) == {"real"}
    assert conv.visibility_stats["skipped_unset_pose"] == 1


def test_stationary_actor_off_the_origin_is_kept():
    """Zero speed alone is no reason to drop: a parked vehicle must stay scored after its source ends."""
    agents = [_agent("parked", [4.0, 0.0])]
    conv = _converter(agents)
    assert _tokens(conv.convert_to_detections_tracks_from_agent_input(2)) == {"parked"}
    assert conv.visibility_stats["skipped_unset_pose"] == 0


def test_actor_moving_through_the_origin_is_kept():
    """Dropping on the origin alone would remove actors passing the start point; speed must also be zero."""
    agents = [_agent("crossing", [0.0, 0.0], velocity=(1.4, 0.0))]
    conv = _converter(agents)
    assert _tokens(conv.convert_to_detections_tracks_from_agent_input(2)) == {"crossing"}
    assert conv.visibility_stats["skipped_unset_pose"] == 0


def test_dropped_actor_is_not_placed_on_the_ego():
    """Pins the behaviour without the drop: an origin actor left in place lands on the ego."""
    from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (
        denormalize_from_ego_center)
    placed = denormalize_from_ego_center(np.zeros(2), EGO_CENTER)
    assert np.allclose(placed[:2], EGO_CENTER), placed


# --- Fixing already recorded runs by rescoring ---------------------------------------------

def _row(token, kind, x, y, vx=0.0, vy=0.0):
    return [token, kind, x, y, 0.0, 0.8, 0.8, 1.7, vx, vy]


def test_rescore_drops_the_teleported_last_frame():
    """Recorded runs do not keep the origin, so match the signature: last row + teleport + zero speed."""
    from odyssey_benchmark.driving_metrics import drop_unset_pose_frames
    # A pedestrian walks normally, then jumps to the start point on its last frame only.
    frames = [
        [_row("ped", "PEDESTRIAN", 40.0, 0.0, 0.0, 1.5), _row("car", "VEHICLE", 5.0, 0.0, 3.0, 0.0)],
        [_row("ped", "PEDESTRIAN", 40.0, 0.15, 0.0, 1.5), _row("car", "VEHICLE", 5.3, 0.0, 3.0, 0.0)],
        [_row("ped", "PEDESTRIAN", 0.0, 0.0), _row("car", "VEHICLE", 5.6, 0.0, 3.0, 0.0)],
    ]
    out, dropped = drop_unset_pose_frames(frames)
    assert dropped == 1
    assert [len(f) for f in out] == [2, 2, 1]
    assert [r[0] for r in out[2]] == ["car"]


def test_rescore_keeps_a_parked_actor_and_a_normal_disappearance():
    """A parked vehicle has zero speed but does not jump. An actor that simply disappears has a normal last row."""
    from odyssey_benchmark.driving_metrics import drop_unset_pose_frames
    frames = [
        [_row("parked", "VEHICLE", 12.0, 3.0), _row("ped", "PEDESTRIAN", 9.0, 0.0, 0.0, 1.0)],
        [_row("parked", "VEHICLE", 12.0, 3.0), _row("ped", "PEDESTRIAN", 9.0, 0.1, 0.0, 1.0)],
        [_row("parked", "VEHICLE", 12.0, 3.0)],
    ]
    out, dropped = drop_unset_pose_frames(frames)
    assert dropped == 0
    assert out is frames                      # returned unchanged when nothing is dropped


def test_rescore_needs_all_three_conditions():
    """Nothing is dropped unless all three conditions hold; the check must never err towards deleting real actors."""
    from odyssey_benchmark.driving_metrics import drop_unset_pose_frames
    # Teleported but still has speed (not the bug signature).
    moving = [[_row("t", "PEDESTRIAN", 40.0, 0.0, 0.0, 1.5)],
              [_row("t", "PEDESTRIAN", 0.0, 0.0, 0.1, 0.0)]]
    assert drop_unset_pose_frames(moving)[1] == 0
    # Zero speed but the jump is small (an actor that just stopped).
    small = [[_row("t", "PEDESTRIAN", 1.0, 0.0)], [_row("t", "PEDESTRIAN", 1.2, 0.0)]]
    assert drop_unset_pose_frames(small)[1] == 0
    # An actor with a single frame has no previous row, so it is not judged.
    single = [[_row("t", "PEDESTRIAN", 0.0, 0.0)]]
    assert drop_unset_pose_frames(single)[1] == 0
