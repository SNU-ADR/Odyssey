from odyssey_renderer.base_renderer import RenderState
from odyssey_renderer.omnire.engine import OmniReRenderEngine
from odyssey.manager.render_manager import RenderManager


def _engine_with_recorder():
    engine = object.__new__(OmniReRenderEngine)
    calls = []

    def record(mapping, **kwargs):
        calls.append((mapping, kwargs))

    engine.set_traffic_light_source_frames = record
    return engine, calls


def test_render_state_names_traffic_light_source_frames():
    assert RenderState.TL_SOURCE_FRAMES == "tl_source_frames"


def test_engine_applies_mapping_in_strict_mode():
    engine, calls = _engine_with_recorder()
    mapping = {"tl_bulb_35949": 350}
    engine._apply_traffic_light_source_frames(
        RenderState({RenderState.TL_SOURCE_FRAMES: mapping}))
    assert calls == [(mapping, {})]


def test_engine_applies_empty_mapping_to_clear_previous_override():
    engine, calls = _engine_with_recorder()
    engine._apply_traffic_light_source_frames(
        RenderState({RenderState.TL_SOURCE_FRAMES: {}}))
    assert calls == [({}, {})]


def test_engine_is_noop_when_control_key_is_absent():
    engine, calls = _engine_with_recorder()
    engine._apply_traffic_light_source_frames(RenderState())
    assert calls == []


def test_render_trace_exposes_the_exact_signal_source_override_for_tlc():
    manager = object.__new__(RenderManager)
    manager._tl_ctl = object()
    manager._tl_log = {
        10: {'lane_match': {'signal_source_overrides': {'planned': 'actual'}}},
        11: {'lane_match': {'signal_source_overrides': {}}},
    }
    assert manager.traffic_light_signal_source_overrides == {
        10: {'planned': 'actual'},
    }
