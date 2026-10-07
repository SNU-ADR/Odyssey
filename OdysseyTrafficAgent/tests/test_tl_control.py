"""TrafficLightController rules, one per test, using fake control data and timetables."""
import pytest

from odyssey_renderer.omnire.tl_control import (
    TL_CONTROL_SCENE_EXCLUSIONS,
    TrafficLightController,
    excluded_scene_reason,
    normalize,
)

R, G, U = "TRAFFIC_LIGHT_RED", "TRAFFIC_LIGHT_GREEN", "TRAFFIC_LIGHT_UNKNOWN"


def control(**over):
    c = {
        "schema": "tl_control/1", "num_frames": 100,
        "heads": {
            "A": {"node": True, "representatives": {"red": 10, "green": 60}, "unusable": {}, "hold_frame": 90},
            "B": {"node": True, "representatives": {"red": 12}, "unusable": {}, "hold_frame": 80},
            "K": {"node": True, "representatives": {"red": 5, "green": 55}, "unusable": {}, "hold_frame": 70},
            "N": {"node": False, "representatives": {}, "unusable": {}, "hold_frame": None},
        },
        "connectors": {"c1": ["A", "B"], "c2": ["K"], "c3": ["A"]},
        "keep_natural": ["K"],
    }
    c.update(over)
    return c


def table(**over):
    t = {"c1": [R] * 50 + [G] * 50, "c2": [R] * 50 + [G] * 50, "c3": [R] * 50 + [G] * 50}
    t.update(over)
    return t


def test_rep_follows_timetable_state():
    ctl = TrafficLightController(control(), table())
    assert ctl.resolve(20).mapping["A"] == 10
    assert ctl.resolve(70).mapping["A"] == 60


def test_missing_rep_is_reported_not_silent():
    r = TrafficLightController(control(), table()).resolve(70)
    assert "B" not in r.mapping and r.report["B"] == "cannot:green"


def test_keep_natural_untouched_on_identity_clock():
    r = TrafficLightController(control(), table()).resolve(20)
    assert "K" not in r.mapping and r.report["K"] == "keep:red"


def test_keep_natural_uses_rep_when_clock_shifts():
    """When sector replay shifts the rows, natural playback no longer matches the timetable."""
    ctl = TrafficLightController(control(), table(), source_row=lambda c, s: s + 40)
    r = ctl.resolve(20)                       # row 60 -> green
    assert r.mapping["K"] == 55 and r.report["K"] == "rep:green"


def test_conflicting_connectors_leave_head_natural():
    r = TrafficLightController(control(), table(c3=[G] * 100)).resolve(20)
    assert "A" not in r.mapping and r.report["A"].startswith("conflict")


def test_route_filter_resolves_conflict():
    r = TrafficLightController(control(), table(c3=[G] * 100)).resolve(20, route_connectors=["c1"])
    assert r.mapping["A"] == 10


def test_unknown_and_negative_row_are_natural():
    ctl = TrafficLightController(control(), table(c1=[U] * 100), source_row=lambda c, s: -1 if c == "c3" else s)
    r = ctl.resolve(20)
    assert "A" not in r.mapping and "A" not in r.report


def test_outside_window_every_node_head_has_a_frame():
    r = TrafficLightController(control(), table()).resolve(150)   # timetable also ended -> all unknown
    assert set(r.mapping) == {"A", "B", "K"}
    assert r.mapping == {"A": 90, "B": 80, "K": 70}
    assert all(v == "hold" for v in r.report.values())


def test_unusable_color_is_not_used():
    c = control()
    c["heads"]["A"]["unusable"] = {"green": True}
    r = TrafficLightController(c, table()).resolve(70, route_connectors=["c1"])
    assert "A" not in r.mapping and r.report["A"] == "cannot:green"


def test_nodeless_light_never_mapped():
    c = control(connectors={"c1": ["N"]})
    assert TrafficLightController(c, table()).resolve(20).mapping == {}


def test_accepts_map_json_style_states():
    assert normalize("red") == "red" and normalize(None) is None and normalize(U) is None


def test_rejects_non_control_input():
    with pytest.raises(ValueError):
        TrafficLightController({"heads": {}}, {})


def test_audited_scene_exclusions_cover_full_and_partial_failures():
    assert set(TL_CONTROL_SCENE_EXCLUSIONS) == {
        "odyssey_scene011", "odyssey_scene021", "odyssey_scene024", "odyssey_scene056", "odyssey_scene065", "odyssey_scene081",
    }
    assert excluded_scene_reason("odyssey_scene024")
    assert excluded_scene_reason("odyssey_scene081")
    assert excluded_scene_reason("odyssey_scene032") is None


def driven_lane_control():
    return {
        "schema": "tl_control/1", "num_frames": 100,
        "heads": {
            "A": {"node": True, "representatives": {"red": 10, "green": 60},
                  "unusable": {}, "hold_frame": 90},
        },
        "connectors": {
            "right_turn": ["A"],
            "right_straight": ["A"],
            "left_straight": ["A"],
        },
        "keep_natural": [],
    }


def driven_lane_scenario():
    line = lambda points, entries=(): {
        "polyline": points, "entry_lanes": list(entries),
    }
    return {
        "dynamic_map_states": {
            "rt": {"type": "TRAFFIC_LIGHT", "traffic_light_lane": "right_turn",
                   "state": {"traffic_light_state": [G] * 100}},
            "rs": {"type": "TRAFFIC_LIGHT", "traffic_light_lane": "right_straight",
                   "state": {"traffic_light_state": [R] * 100}},
            "ls": {"type": "TRAFFIC_LIGHT", "traffic_light_lane": "left_straight",
                   "state": {"traffic_light_state": [R] * 100}},
        },
        "map_features": {
            "right_lane": line([(0, 0), (5, 0), (10, 0)]),
            "left_lane": line([(0, 3), (5, 3), (10, 3)]),
            "right_turn": line([(10, 0), (11, 0), (11, -5)], ["right_lane"]),
            "right_straight": line([(10, 0), (15, 0), (20, 0)], ["right_lane"]),
            "left_straight": line([(10, 3), (15, 3), (20, 3)], ["left_lane"]),
        },
    }


def test_driven_lane_replaces_planned_connector_sharing_the_same_head():
    ctl = TrafficLightController.from_scenario(driven_lane_control(), driven_lane_scenario())
    route = ["right_lane", "left_lane", "right_turn"]
    scope, match = ctl.connector_scope(route, ego_xy=(5, 3), ego_heading=0.0)
    assert scope == ("left_straight",)
    assert match["entry_lane"] == "left_lane"
    assert match["replaced_connectors"] == ["right_turn"]
    assert match["signal_source_overrides"] == {"right_turn": "left_straight"}
    assert ctl.resolve(20, route_connectors=scope).mapping["A"] == 10


def test_driven_lane_preserves_planned_maneuver_when_ego_is_on_planned_lane():
    ctl = TrafficLightController.from_scenario(driven_lane_control(), driven_lane_scenario())
    route = ["right_lane", "left_lane", "right_turn"]
    scope, match = ctl.connector_scope(route, ego_xy=(5, 0), ego_heading=0.0)
    assert scope == ("right_turn",)
    assert match["entry_lane"] == "right_lane"
    assert ctl.resolve(20, route_connectors=scope).mapping["A"] == 60


def test_driven_lane_hysteresis_avoids_midline_flapping():
    ctl = TrafficLightController.from_scenario(driven_lane_control(), driven_lane_scenario())
    route = ["right_lane", "left_lane", "right_turn"]
    scope, match = ctl.connector_scope(
        route, ego_xy=(5, 1.6), ego_heading=0.0, previous_entry_lane="right_lane")
    assert scope == ("right_turn",)
    assert match["entry_lane"] == "right_lane"
    assert match["reason"] == "hysteresis"


def test_driven_lane_choice_is_held_while_ego_crosses_the_intersection():
    ctl = TrafficLightController.from_scenario(driven_lane_control(), driven_lane_scenario())
    route = ["right_lane", "left_lane", "right_turn"]
    scope, match = ctl.connector_scope(
        route, ego_xy=(20, -10), ego_heading=-1.57, previous_entry_lane="left_lane")
    assert scope == ("left_straight",)
    assert match["entry_lane"] == "left_lane"
    assert match["reason"] == "intersection_hold"


# ---- pin_uncontrolled (opt-in) --------------------------------------------------------------
import hashlib  # noqa: E402
import json  # noqa: E402


def _fingerprint(make):
    out = []
    for scope in (None, [], ["c1"], ["c2"], ["c1", "c3"]):
        ctl = make()
        for step in range(-3, 105):
            r = ctl.resolve(step, scope)
            out.append([step, scope, sorted(r.mapping.items()), sorted(r.report.items()), r.in_window])
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


# Fingerprints measured before pin_uncontrolled was added: the default (off) path must match exactly
BEFORE_IDENTITY = "5068f41f4fea1ed0a96be4f9f00eaa4df1a067280aaf624e631b0421c66a9b3e"
BEFORE_SHIFTED = "1f22cde23c1ab92df21cf048a19667ed80ae1dfd7161fe98f0824e39187af2b2"


def test_default_off_is_identical_to_before():
    assert _fingerprint(lambda: TrafficLightController(control(), table())) == BEFORE_IDENTITY
    assert _fingerprint(lambda: TrafficLightController(control(), table(), pin_uncontrolled=False)) == BEFORE_IDENTITY
    assert _fingerprint(lambda: TrafficLightController(control(), table(), source_row=lambda c, s: s - 3)) == BEFORE_SHIFTED
    assert TrafficLightController(control(), table()).pin_uncontrolled is False


def test_pin_same_frame_when_red_equals_green():
    c = control(heads=dict(control()["heads"], A={"node": True, "representatives": {"red": 33, "green": 33},
                                                    "unusable": {}, "hold_frame": 90}))
    ctl = TrafficLightController(c, table(), pin_uncontrolled=True)
    r = ctl.resolve(20, [])                        # out of scope (scope is empty)
    assert r.mapping["A"] == 33 and r.report["A"] == "pin:same"


def test_pin_db_colour_outside_scope():
    ctl = TrafficLightController(control(), table(), pin_uncontrolled=True)
    r = ctl.resolve(20, ["c2"])                    # A belongs to c1/c3 -- out of scope
    assert r.mapping["A"] == 10 and r.report["A"] == "pin:db:red"
    assert r.report["K"] == "keep:red" and "K" not in r.mapping          # keep_natural is untouched
    r = ctl.resolve(70, [])
    assert r.mapping["A"] == 60 and r.report["A"] == "pin:db:green"


def test_pin_hold_last_pinned_frame_when_db_unknown():
    ctl = TrafficLightController(control(), table(c1=[R] * 50 + [U] * 50, c3=[R] * 50 + [U] * 50),
                                 pin_uncontrolled=True)
    assert ctl.resolve(20, ["c1"]).report["A"] == "rep:red"             # a controlled head is unchanged
    r = ctl.resolve(60, [])                        # DB colour unknown -> last drawn frame
    assert r.mapping["A"] == 10 and r.report["A"] == "pin:hold"


def test_pin_lookahead_before_first_pin():
    ctl = TrafficLightController(control(), table(c1=[U] * 30 + [G] * 70, c3=[U] * 30 + [G] * 70),
                                 pin_uncontrolled=True)
    r = ctl.resolve(5, [])                         # nothing drawn yet; the first upcoming colour is green
    assert r.mapping["A"] == 60 and r.report["A"] == "pin:lookahead"
    r = ctl.resolve(6, [])
    assert r.mapping["A"] == 60 and r.report["A"] == "pin:hold"


def test_pin_any_rep_green_first_when_no_future_colour():
    ctl = TrafficLightController(control(), table(c1=[U] * 100, c3=[U] * 100), pin_uncontrolled=True)
    r = ctl.resolve(5, [])
    assert r.mapping["A"] == 60 and r.report["A"] == "pin:lookahead:any"
    ctl = TrafficLightController(control(), table(c1=[U] * 100), pin_uncontrolled=True)   # B: red representative only
    assert ctl.resolve(5, []).mapping["B"] == 12


def test_pin_leaves_skipped_and_repless_heads_alone():
    c = control(heads=dict(control()["heads"], E={"node": True, "representatives": {}, "unusable": {},
                                                    "hold_frame": 77}),
                connectors={"c1": ["A", "B", "E"], "c2": ["K"], "c3": ["A"]})
    ctl = TrafficLightController(c, table(), pin_uncontrolled=True)
    assert ctl.pin_heads == ["A", "B"]             # K: keep_natural, N: no node, E: no representative
    r = ctl.resolve(20, [])
    assert "E" not in r.mapping and "K" not in r.mapping and "N" not in r.mapping
    r = ctl.resolve(150, [])                        # outside the window: pinned heads use a representative, the rest hold_frame
    assert r.mapping["A"] in (10, 60) and r.report["A"].startswith("pin:")
    assert r.mapping["E"] == 77 and r.report["E"] == "hold" and r.mapping["K"] == 70


def test_pin_conflicting_db_colours_hold_instead_of_guessing():
    c = control(connectors={"c1": ["A"], "c2": ["K"], "c3": ["A"]})
    ctl = TrafficLightController(c, table(c3=[G] * 100), pin_uncontrolled=True)
    ctl.resolve(80, ["c1"])                         # both green -> rep:green 60
    r = ctl.resolve(20, [])                         # c1 red, c3 green -> no single colour -> hold
    assert r.mapping["A"] == 60 and r.report["A"] == "pin:hold"


def test_default_off_does_not_parse_reps_at_construction():
    # With the default (off), a non-integer representative on an unused head still builds and runs as before
    c = control(heads=dict(control()["heads"], B={"node": True, "representatives": {"red": "bad"},
                                                    "unusable": {}, "hold_frame": 80}))
    ctl = TrafficLightController(c, table())
    assert ctl.pin_heads == [] and ctl.head_connectors == {}
    assert ctl.resolve(20, ["c2"]).report == {"K": "keep:red"}
    with pytest.raises(ValueError):
        TrafficLightController(c, table(), pin_uncontrolled=True)     # when on, construction fails loudly


def test_lookahead_starts_at_the_row_where_the_sector_opens():
    # The sector opens at step 10 on row 40. Rows 0-39 are green, 40+ red -> use the opening row colour (red)
    series = [G] * 40 + [R] * 60
    src = lambda c, s: -1 if s < 10 else s - 10 + 40            # noqa: E731
    ctl = TrafficLightController(control(), table(c1=series, c3=series), source_row=src, pin_uncontrolled=True)
    r = ctl.resolve(0, [])
    assert r.mapping["A"] == 10 and r.report["A"] == "pin:lookahead"
    # A sector that never opens is skipped -> green representative first
    ctl = TrafficLightController(control(), table(c1=series, c3=series), source_row=lambda c, s: -1,
                                 pin_uncontrolled=True)
    assert ctl.resolve(0, []).report["A"] == "pin:lookahead:any" and ctl.resolve(1, []).mapping["A"] == 60
