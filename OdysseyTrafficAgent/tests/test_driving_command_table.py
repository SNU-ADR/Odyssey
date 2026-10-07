"""log_progress: the log's label for the point of the drive the ego has reached, verbatim."""
import numpy as np
import pytest

from odyssey.manager import data_manager as dm
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.utils.driving_command_table import (
    LOOKAHEAD_M, SOURCE_LOG, SOURCE_TRACK, SOURCE_TRACK_HELD,
    DrivingCommandTable, build_driving_command_table)

L, S, R, U = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1])


def _info(x, y, command):
    return {"ego2global_translation": [x, y, 0.0], "ego2global_rotation": [1, 0, 0, 0],
            "driving_command": np.array(command, dtype=np.int64)}


def test_locate_is_by_position_not_by_clock():
    # the log drives 0..99 m at 1 m / frame; the ego has only reached x = 12 when the clock says 60
    table = build_driving_command_table(
        [_info(x, 0, R if 40 <= x < 60 else S) for x in range(100)])
    k = table.locate([12.0, 3.0], anchor_s=10.0, time_index=60)      # 3 m to the side: irrelevant
    assert k == 12 and table.onehot[k].tolist() == S
    assert table.onehot[table.locate([45.0, -2.5], 44.0, 0)].tolist() == R


def test_locate_does_not_jump_to_the_other_pass_of_a_loop():
    # out along y = 0, back along y = 4: the return pass runs 4 m from the outbound one
    out = [_info(x, 0, S) for x in range(0, 60)]
    back = [_info(x, 4, L) for x in range(59, -1, -1)]
    table = build_driving_command_table(out + back)
    ego = [20.0, 2.6]                          # nearer the RETURN pass (1.4 m) than the outbound (2.6 m)
    assert table.onehot[table.locate(ego, anchor_s=19.0, time_index=20)].tolist() == S
    assert table.onehot[table.locate(ego, anchor_s=float(table.cum[99]), time_index=99)].tolist() == L


def test_an_ego_that_drifts_off_the_drive_keeps_the_label_it_reached():
    """Lateral drift must not move the answer -- that is the whole point of indexing by progress.

    Replayed over the 62 rollouts on disk (84,502 steps), the live ego sits a median
    2.3 m from the logged frame it snaps to, 12.4 m at p95 and 39.0 m at worst -- so the window has
    to hold the label steady across a lane-width and then some. It does: within the search window
    the nearest logged point stays the one the ego reached, whatever the lateral offset.

    Beyond the window it does NOT hold -- at ~100 m to the side, nearest-in-window drifts to the
    window edge and can hand back an earlier frame's label. That is pinned here as the known bound
    rather than fixed: no measured step comes within 60 m of it (p100 = 39 m, >= 100 m on 0.000% of
    steps), and clamping instead would hide a scene whose ego has genuinely left the map.
    """
    table = build_driving_command_table(
        [_info(x, 0, R if x >= 30 else S) for x in range(60)])
    for lateral in (0.5, 2.0, 5.0, 10.0, 30.0):          # every drift the rollouts actually show
        k = table.locate([31.0, lateral], anchor_s=31.0, time_index=2)
        assert table.onehot[k].tolist() == R, (lateral, table.onehot[k].tolist())


def test_standstill_takes_the_frame_nearest_the_clock():
    infos = [_info(0, 0, S)] * 5 + [_info(0, 0, L)] * 5 + [_info(x, 0, L) for x in range(1, 10)]
    table = build_driving_command_table(infos)
    assert table.locate([0.0, 0.0], 0.0, time_index=2) == 2
    assert table.locate([0.0, 0.0], 0.0, time_index=8) == 8


def _turning_track(sign, straight_m=40, radius=20.0, spacing=1.0):
    """Infos along a track that runs straight then turns (sign +1 left, -1 right). All `unknown`.

    Long enough that the straight part alone exceeds the lookahead, so the early frames must read
    `straight` and the ones whose lookahead lands in the curve must read the turn.
    """
    pts = [(x, 0.0) for x in np.arange(0.0, straight_m + spacing, spacing)]
    for arc in np.arange(spacing, radius * np.pi / 2, spacing):
        a = arc / radius
        pts.append((straight_m + radius * np.sin(a), sign * radius * (1 - np.cos(a))))
    return [_info(x, y, U) for x, y in pts]


def test_unknown_is_filled_from_the_gt_track():
    """A turn ahead on the logged track becomes left/right; a straight one stays straight."""
    for sign, turn in ((1, L), (-1, R)):
        table = build_driving_command_table(_turning_track(sign))
        assert table.onehot.tolist()[0] == U           # the log's own labels are untouched
        # s0 = 0: the whole 20 m lookahead is on the straight run
        assert table.filled[0].tolist() == S
        assert table.source[0] == SOURCE_TRACK
        # s0 = 30 m: the +20 m point is 10 m into the curve, past the 2 m threshold
        assert table.filled[30].tolist() == turn
        assert table.source[30] == SOURCE_TRACK


def test_fill_holds_the_last_full_lookahead_instead_of_shortening_it():
    """Under LOOKAHEAD_M of track left, the command is held, not recomputed on a stub.

    A shortened lookahead shrinks the lateral offset against a fixed threshold, so a turn reads
    `straight` -- measured: agreement falls to 54% in the last 2-5 m. The held value is the one
    derived at the last point that still had the full lookahead.
    """
    table = build_driving_command_table(_turning_track(1))
    total = float(table.cum[-1])
    taper = [i for i in range(len(table.cum)) if total - table.cum[i] < LOOKAHEAD_M]
    assert taper, "fixture must have a taper zone"
    last_full = min(taper) - 1
    assert all(table.source[i] == SOURCE_TRACK_HELD for i in taper)
    assert all(table.filled[i].tolist() == table.filled[last_full].tolist() for i in taper)
    assert table.filled[taper[0]].tolist() == L        # the turn survives instead of flattening


def test_logged_labels_are_never_overwritten_by_the_track():
    """Only `unknown` is filled. A log that says straight through a curve keeps saying straight."""
    infos = _turning_track(1)
    for i in (0, 30):
        infos[i]["driving_command"] = np.array(S, dtype=np.int64)
    table = build_driving_command_table(infos)
    for i in (0, 30):
        assert table.filled[i].tolist() == S
        assert table.source[i] == SOURCE_LOG


def test_a_track_shorter_than_the_lookahead_leaves_unknown_standing():
    """No point on the track has a full lookahead, so there is nothing honest to fill with.

    Inventing `straight` here would be the silent substitution the module refuses; `unknown`
    standing is visible, and the consumer still clamps it (planners/recogdrive.py).
    """
    infos = [_info(0, 0, S), _info(1, 0, U), _info(2, 0, U), _info(3, 0, R)]
    table = build_driving_command_table(infos)
    assert table.onehot.tolist() == [S, U, U, R]
    assert table.filled.tolist() == [S, U, U, R]
    assert table.source == [SOURCE_LOG] * 4

    allunk = build_driving_command_table([_info(0, 0, U), _info(1, 0, U)])
    assert allunk.filled.tolist() == [U, U]


def test_manager_follows_progress_writes_source_and_restarts_with_the_episode(monkeypatch):
    infos = {f"t{x}": _info(x, 0, R if x >= 30 else S) for x in range(60)}

    class Engine:
        current_scene = {SD.ID: "scene-000", "metadata": {"openscene_data_infos_dict": infos}}
        global_config = {}

    monkeypatch.setattr(dm.DataManager, "engine", property(lambda self: Engine))
    manager = object.__new__(dm.DataManager)
    manager._command_table_cache, manager._command_progress_s = {}, None

    def step(x, clock):
        ego2global = np.eye(4)
        ego2global[:2, 3] = [x, 1.5]
        frame = {"ego2global": ego2global, "driving_command": np.array(S, dtype=np.int16)}
        manager._progress_driving_command(frame, ego_vehicle=None, base_idx=clock)
        return frame

    slow = [step(x, clock) for clock, x in enumerate(np.linspace(0, 10, 40))]   # 10 m in 40 ticks
    assert all(f["driving_command"].tolist() == S for f in slow)               # clock 30+ says R
    assert slow[-1]["driving_command_source"] == SOURCE_LOG
    assert slow[-1]["driving_command"].dtype == np.int16
    assert step(31.0, 41)["driving_command"].tolist() == R

    assert isinstance(manager._command_table_cache["scene-000"], DrivingCommandTable)
    manager._command_progress_s = None                     # what DataManager.reset() does
    assert step(0.0, 0)["driving_command"].tolist() == S


def test_frame_route_is_the_scorers_route_and_shares_one_cache(monkeypatch):
    calls = []

    def scorer_route(positions, center, map_api, logger=None):
        calls.append((np.asarray(positions).tolist(), np.asarray(center).tolist(), map_api))
        return [49560, "50104", "48708"]

    class Engine:
        current_scene = {SD.ID: "scene-000", "map": "us-ma-boston",
                         "metadata": {"old_origin_in_current_coordinate": [-331478.6, -4691010.8]}}
        global_config = {}

    class Ego:
        object_track = {SD.POSITION: np.array([[0.9, -0.4], [68.5, -151.3]])}

    monkeypatch.setattr(dm.DataManager, "engine", property(lambda self: Engine))
    monkeypatch.setattr(dm.DataManager, "_scene_map_api", lambda self, scene: "API")
    monkeypatch.setattr(dm, "route_roadblock_ids", scorer_route)
    manager = object.__new__(dm.DataManager)
    manager._route_roadblock_ids_cache = {}

    frames = [{}, {}]
    for frame in frames:
        manager._attach_route_info(frame, Ego)
    assert all(f["route_roadblock_ids"] == ["49560", "50104", "48708"] for f in frames)
    assert frames[0]["map_name"] == "us-ma-boston"
    # resolved once per scene, with the inputs MetricManager gives the same function:
    # the logged track, and local -> global as minus old_origin_in_current_coordinate
    assert calls == [([[0.9, -0.4], [68.5, -151.3]], [331478.6, 4691010.8], "API")]
    # one route per scene, resolved once and shared by every frame
    assert manager._route_roadblock_ids(Engine.current_scene, Ego) is manager._route_roadblock_ids_cache["scene-000"]
    assert len(calls) == 1


def test_a_scene_whose_route_cannot_be_resolved_leaves_the_frame_without_one(monkeypatch):
    class Engine:
        current_scene = {SD.ID: "no-map", "metadata": {}}       # no old_origin -> KeyError inside
        global_config = {}

    monkeypatch.setattr(dm.DataManager, "engine", property(lambda self: Engine))
    manager = object.__new__(dm.DataManager)
    manager._route_roadblock_ids_cache = {}
    frame = {}
    manager._attach_route_info(frame, ego_vehicle=None)
    assert "route_roadblock_ids" not in frame and "map_name" not in frame
