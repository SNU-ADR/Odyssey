"""allow_unobserved: scene-limited exception (odyssey_scene060, odyssey_scene075).

A label head may carry allow_unobserved {colour: frame}; that representative need not be an observed frame.
tl_control accepts it only for the allowed scenes, reports rep_unobserved:<colour>, and hands the renderer the
one allowed (head, frame). Without the key everything is byte-identical (fingerprints of test_tl_control).
"""
import json

import pytest
import torch.nn as nn

from test_tl_control import BEFORE_IDENTITY, BEFORE_SHIFTED, _fingerprint, control, table
from test_tl_control_allow_excluded import _manager
from odyssey_renderer.base_renderer import RenderState
from odyssey_renderer.omnire.tl_control import ALLOW_UNOBSERVED_SCENES, TrafficLightController
from odyssey_renderer.omnire.traffic_light import OmniReTrafficLightSubModel
from odyssey_renderer.omnire.engine import OmniReRenderEngine

R, G = "TRAFFIC_LIGHT_RED", "TRAFFIC_LIGHT_GREEN"


def allowed_control(scene="odyssey_scene060", frame=7):
    c = control(scene_id=scene)
    c["heads"]["B"] = {"node": True, "representatives": {"red": frame, "green": 62}, "unusable": {},
                       "hold_frame": 80, "allow_unobserved": {"red": frame}}
    return c


def test_allowed_scenes_are_exactly_c085_and_c101():
    assert ALLOW_UNOBSERVED_SCENES == frozenset({"odyssey_scene060", "odyssey_scene075"})


def test_default_without_the_key_is_identical_to_before():
    assert _fingerprint(lambda: TrafficLightController(control(), table())) == BEFORE_IDENTITY
    assert _fingerprint(lambda: TrafficLightController(control(), table(), source_row=lambda c, s: s - 3)) == BEFORE_SHIFTED
    assert TrafficLightController(control(), table()).allow_unobserved == {}
    assert TrafficLightController(control(), table()).resolve(20).unobserved_ok == {}


@pytest.mark.parametrize("scene", ["odyssey_scene060", "odyssey_scene075", "odyssey_scene060"])
def test_allowed_scene_maps_the_frame_and_reports_rep_unobserved(scene):
    ctl = TrafficLightController(allowed_control(scene), table())
    r = ctl.resolve(20, ["c1"])
    assert r.mapping["B"] == 7 and r.report["B"] == "rep_unobserved:red" and r.unobserved_ok == {"B": 7}
    assert r.report["A"] == "rep:red" and "A" not in r.unobserved_ok        # other heads unchanged
    g = ctl.resolve(70, ["c1"])
    assert g.mapping["B"] == 62 and g.report["B"] == "rep:green" and g.unobserved_ok == {}
    assert r.counts()["rep_unobserved"] == 1


@pytest.mark.parametrize("scene", ["odyssey_scene971", "odyssey_scene032", "", "odyssey_scene008"])
def test_allowance_in_any_other_scene_fails(scene):
    with pytest.raises(ValueError, match="allow_unobserved is only allowed for scenes"):
        TrafficLightController(allowed_control(scene), table())


def test_allowance_must_equal_the_representative():
    c = allowed_control()
    c["heads"]["B"]["allow_unobserved"] = {"red": 8}
    with pytest.raises(ValueError, match="must equal the red representative 7"):
        TrafficLightController(c, table())


def test_render_manager_passes_the_allowance_only_when_used(tmp_path):
    path = tmp_path / "odyssey_scene060.json"
    path.write_text(json.dumps(dict(allowed_control(), num_frames=100)))
    m = _manager("odyssey_scene060", {"tl_control_path": str(path)})
    m.current_scene = {"dynamic_map_states": {
        "1": {"type": "TRAFFIC_LIGHT", "traffic_light_lane": "c1", "state": {"traffic_light_state": [R] * 50 + [G] * 50}}},
        "map_features": {}}
    m._configure_traffic_light_control()
    assert m._traffic_light_step(20, ["c1"], None, None) == {"A": 10, "B": 7}
    assert m._tl_unobserved_ok == {"B": 7} and m._tl_log[20]["unobserved_ok"] == {"B": 7}
    assert m._tl_log[20]["report"]["B"] == "rep_unobserved:red"
    assert m._traffic_light_step(70, ["c1"], None, None) == {"A": 60, "B": 62}
    assert m._tl_unobserved_ok == {} and "unobserved_ok" not in m._tl_log[70]


def _engine():
    engine = object.__new__(OmniReRenderEngine)
    calls = []
    engine.set_traffic_light_source_frames = lambda mapping, **kw: calls.append((mapping, kw))
    return engine, calls


def test_engine_forwards_the_allowance_and_is_unchanged_without_it():
    engine, calls = _engine()
    engine._apply_traffic_light_source_frames(RenderState({RenderState.TL_SOURCE_FRAMES: {"B": 7},
                                                           RenderState.TL_ALLOW_UNOBSERVED: {"B": 7}}))
    engine._apply_traffic_light_source_frames(RenderState({RenderState.TL_SOURCE_FRAMES: {"B": 7}}))
    assert calls == [({"B": 7}, {"allow_unobserved": {"B": 7}}), ({"B": 7}, {})]


def _node():
    node = object.__new__(OmniReTrafficLightSubModel)
    nn.Module.__init__(node)
    node.eval()
    node.head_ids = ("A", "B")
    node._observed = {"A": frozenset({1, 2}), "B": frozenset({100})}
    node.num_frames = 500
    return node


def test_node_accepts_exactly_the_allowed_unobserved_frame():
    node = _node()
    with pytest.raises(ValueError, match="B: source frame must be observed"):
        node.set_source_frames({"B": 70})                                     # default: strict as before
    node.set_source_frames({"A": 1, "B": 70}, allow_unobserved={"B": 70})
    assert node._source_overrides == {"A": 1, "B": 70}
    with pytest.raises(ValueError, match="B: source frame must be observed"):
        node.set_source_frames({"B": 71}, allow_unobserved={"B": 70})        # only that frame
    with pytest.raises(ValueError, match="A: source frame must be observed"):
        node.set_source_frames({"A": 70, "B": 70}, allow_unobserved={"B": 70})  # only that head
    with pytest.raises(ValueError, match="must be observed"):
        node.set_source_frames({"B": 600}, allow_unobserved={"B": 600})      # still a training frame
    with pytest.raises(KeyError):
        node.set_source_frames({}, allow_unobserved={"Z": 1})
