"""tl_control_allow_excluded: opt-in that explicitly bypasses the audited exclusion list (TL_CONTROL_SCENE_EXCLUSIONS) per scene.

With the key missing or empty, behaviour is unchanged (excluded scenes play naturally, others as before).
"""
import json
import logging

from odyssey.manager.render_manager import RenderManager
from odyssey_renderer.omnire.tl_control import TL_CONTROL_SCENE_EXCLUSIONS


class _Agents:
    @staticmethod
    def traffic_light_source_row(connector, step):
        return int(step)


class _Engine:
    managers = {'agent_manager': _Agents()}


class _RM(RenderManager):
    global_config = property(lambda self: self._cfg)
    engine = property(lambda self: _Engine())


def _manager(scene, cfg):
    m = object.__new__(_RM)
    m._cfg = cfg
    m.current_scene_id = scene
    m.current_scene = {'dynamic_map_states': {}, 'map_features': {}}
    return m


def _control(tmp_path, scene):
    p = tmp_path / f'{scene}.json'
    p.write_text(json.dumps({
        'schema': 'tl_control/1', 'scene_id': scene, 'num_frames': 10,
        'heads': {'h': {'node': True, 'representatives': {'red': 3, 'green': 3}, 'unusable': {}, 'hold_frame': 3}},
        'connectors': {'c1': ['h']}, 'keep_natural': []}))
    return p


def test_an_audited_scene_is_on_the_exclusion_list():
    assert 'odyssey_scene024' in TL_CONTROL_SCENE_EXCLUSIONS and 'odyssey_scene032' not in TL_CONTROL_SCENE_EXCLUSIONS


def test_excluded_scene_without_the_key_is_still_excluded(tmp_path):
    for cfg in ({'tl_control_path': str(tmp_path / 'never_read.json')},
                {'tl_control_path': str(tmp_path / 'never_read.json'), 'tl_control_allow_excluded': []},
                {'tl_control_path': str(tmp_path / 'never_read.json'), 'tl_control_allow_excluded': ['odyssey_scene032']}):
        m = _manager('odyssey_scene024', cfg)
        m._configure_traffic_light_control()          # does not read the path (a missing file is fine)
        assert m._tl_ctl is None
        assert m.traffic_light_control_trace == {
            'schema': 'tl_render_control_trace/2', 'scene_id': 'odyssey_scene024',
            'control_path': str(tmp_path / 'never_read.json'), 'control_sha256': None, 'excluded': True,
            'exclusion_reason': TL_CONTROL_SCENE_EXCLUSIONS['odyssey_scene024'], 'missing_connectors': [], 'steps': []}


def test_excluded_scene_explicitly_allowed_builds_the_controller(tmp_path, caplog):
    path = _control(tmp_path, 'odyssey_scene024')
    m = _manager('odyssey_scene024', {'tl_control_path': str(path), 'tl_control_allow_excluded': ['odyssey_scene024']})
    with caplog.at_level(logging.WARNING, logger='odyssey.manager.render_manager'):
        m._configure_traffic_light_control()
    assert m._tl_ctl is not None and m._tl_ctl.node_heads
    reason = TL_CONTROL_SCENE_EXCLUSIONS['odyssey_scene024']
    lines = [r.getMessage() for r in caplog.records]
    assert (f'tl_control: scene odyssey_scene024 is on the exclusion list ({reason}) but explicitly allowed '
            f'by tl_control_allow_excluded') in lines
    trace = m.traffic_light_control_trace
    assert trace['exclusion_overridden'] is True and trace['exclusion_reason'] == reason
    assert 'excluded' not in trace and trace['control_sha256'] is not None


def test_non_excluded_scene_unaffected_by_the_key(tmp_path):
    path = _control(tmp_path, 'odyssey_scene032')
    traces = []
    for cfg in ({'tl_control_path': str(path)},
                {'tl_control_path': str(path), 'tl_control_allow_excluded': ['odyssey_scene024', 'odyssey_scene032']}):
        m = _manager('odyssey_scene032', cfg)
        m._configure_traffic_light_control()
        assert m._tl_ctl is not None
        traces.append(json.dumps(m.traffic_light_control_trace, sort_keys=True))
    assert traces[0] == traces[1]
    assert 'exclusion_overridden' not in traces[0] and 'exclusion_reason' not in traces[0]


def test_no_control_path_is_still_a_noop():
    m = _manager('odyssey_scene024', {'tl_control_path': None, 'tl_control_allow_excluded': ['odyssey_scene024']})
    m._configure_traffic_light_control()
    assert m._tl_ctl is None and m.traffic_light_control_trace is None


def test_allow_list_entries_without_effect_are_logged_once(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(RenderManager, '_TL_ALLOW_WARNED', set())
    path = _control(tmp_path, 'odyssey_scene032')
    cfg = {'tl_control_path': str(path), 'tl_control_allow_excluded': ['c71', 'C071', 'odyssey_scene032', 'odyssey_scene024']}
    with caplog.at_level(logging.WARNING, logger='odyssey.manager.render_manager'):
        for _ in range(2):                                          # configured twice, logged once
            m = _manager('odyssey_scene032', cfg)
            m._configure_traffic_light_control()
            assert m._tl_ctl is not None                            # behaviour unchanged
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "tl_control_allow_excluded: entry 'C071' has no effect (not a published scene name (odyssey_sceneNNN))",
        "tl_control_allow_excluded: entry 'c71' has no effect (not a published scene name (odyssey_sceneNNN))",
        "tl_control_allow_excluded: entry 'odyssey_scene032' has no effect (scene is not on the tl_control exclusion list)",
    ]


def test_pin_uncontrolled_default_off_and_opt_in(tmp_path, caplog):
    path = _control(tmp_path, 'odyssey_scene032')
    m = _manager('odyssey_scene032', {'tl_control_path': str(path)})
    with caplog.at_level(logging.INFO, logger='odyssey.manager.render_manager'):
        m._configure_traffic_light_control()
    assert m._tl_ctl.pin_uncontrolled is False and 'pin_uncontrolled' not in m.traffic_light_control_trace
    assert '[tl-control] pin_uncontrolled=on' not in [r.getMessage() for r in caplog.records]
    caplog.clear()
    m = _manager('odyssey_scene032', {'tl_control_path': str(path), 'tl_control_pin_uncontrolled': True})
    with caplog.at_level(logging.INFO, logger='odyssey.manager.render_manager'):
        m._configure_traffic_light_control()
    assert m._tl_ctl.pin_uncontrolled is True and m.traffic_light_control_trace['pin_uncontrolled'] is True
    assert '[tl-control] pin_uncontrolled=on' in [r.getMessage() for r in caplog.records]


# ---- warm-up (route unavailable) ----------------------------------------------------------------
def _warm_manager(tmp_path, cfg_extra):
    path = tmp_path / 'odyssey_scene032.json'
    path.write_text(json.dumps({
        'schema': 'tl_control/1', 'scene_id': 'odyssey_scene032', 'num_frames': 100,
        'heads': {'h': {'node': True, 'representatives': {'red': 3, 'green': 3}, 'unusable': {}, 'hold_frame': 9},
                  'g': {'node': True, 'representatives': {}, 'unusable': {}, 'hold_frame': 8}},
        'connectors': {'c1': ['h', 'g']}, 'keep_natural': []}))
    m = _manager('odyssey_scene032', dict({'tl_control_path': str(path)}, **cfg_extra))
    m.current_scene = {'dynamic_map_states': {'1': {'type': 'TRAFFIC_LIGHT', 'traffic_light_lane': 'c1',
                                                    'state': {'traffic_light_state': ['TRAFFIC_LIGHT_RED'] * 100}}},
                       'map_features': {}}
    m._configure_traffic_light_control()
    return m


def test_warmup_default_off_is_unchanged(tmp_path):
    m = _warm_manager(tmp_path, {})
    # Without a route, same as before: returns None and the log has only these four fields
    assert m._traffic_light_step(0, None, None, None) is None
    assert m._tl_log[0] == {'step': 0, 'mapping': {}, 'counts': {}, 'skipped': 'route_unavailable'}
    # With a route, the same fields as before (scope is the route, h is rep:red)
    frames = m._traffic_light_step(20, ['c1'], None, None)
    assert frames == {'h': 3}
    log = m._tl_log[20]
    assert sorted(log) == ['counts', 'lane_match', 'mapping', 'report', 'route_connectors', 'signal_scope', 'step']
    assert log['report'] == {'h': 'rep:red', 'g': 'cannot:red'} and log['counts'] == {'rep': 1, 'cannot': 1}
    assert log['signal_scope'] == ['c1'] and log['route_connectors'] == ['c1']
    assert json.dumps(m.traffic_light_control_trace, sort_keys=True) == json.dumps(
        dict(schema='tl_render_control_trace/2', scene_id='odyssey_scene032', control_path=m._tl_control_path,
             control_sha256=m._tl_control_sha256, missing_connectors=[],
             steps=[m._tl_log[0], m._tl_log[20]]), sort_keys=True)


def test_warmup_pin_uncontrolled_draws_pinned_heads_from_step_0(tmp_path):
    m = _warm_manager(tmp_path, {'tl_control_pin_uncontrolled': True})
    frames = m._traffic_light_step(0, None, None, None)
    assert frames == {'h': 3}                                   # g has no representative, so it plays naturally as before
    log = m._tl_log[0]
    assert log['skipped'] == 'route_unavailable' and log['report'] == {'h': 'pin:same'}
    assert log['signal_scope'] == [] and log['route_connectors'] == [] and log['counts'] == {'pin': 1}
    assert m.traffic_light_control_trace['pin_uncontrolled'] is True
