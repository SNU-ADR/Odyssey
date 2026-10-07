"""nuplan_idm_static_pin_and_predrop: parked vehicles are static; vehicles that would fail the gate never appear.

Pins the R (nuplan_idm_policy) decisions with small synthetic tracks:
  1. Only a VEHICLE valid on every frame with zero xy motion (or control_mode=static) is pinned.
  2. A pinned vehicle is an obstacle, not an IDM builder candidate, is never sector-replayed, and is labelled 'static'.
  3. A vehicle that would fail the handoff gate is predropped instead of using GT fallback, and has no warm-up pose.
  4. With the switch off, behaviour is unchanged (GT fallback).
  5. Parked vehicles remain after their source ends, **in NR too**. Parking is a property of the track alone,
     so _parked_vehicle_ids is filled regardless of the R switch. The simulation runs longer than the log,
     and removing these vehicles only in NR left an empty road after step 1000.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from odyssey.components.agents.policy import nuplan_idm_policy as policy_module
from odyssey.components.agents.policy.nuplan_idm_policy import NuPlanIDMBatch
from odyssey.manager import base_manager as base_manager_module
from odyssey.manager.agent_manager import BaseAgentManager, _is_pinned_static_vehicle


def _track(positions, valid=None, kind="VEHICLE", control_mode=None):
    positions = np.asarray(positions, dtype=float)
    valid = np.ones(len(positions), dtype=bool) if valid is None else np.asarray(valid)
    metadata = {} if control_mode is None else {"control_mode": control_mode}
    return {"type": kind, "metadata": metadata,
            "state": {"position": positions, "valid": valid}}


PARKED = [[5.0, 2.0, 620.0]] * 6


def test_predicate_pins_only_whole_episode_stationary_vehicles():
    assert _is_pinned_static_vehicle(_track(PARKED))
    # A single invalid frame means the vehicle is not parked for the whole episode.
    assert not _is_pinned_static_vehicle(_track(PARKED, valid=[1, 1, 1, 0, 1, 1]))
    # Any xy motion of 1 mm or more disqualifies it (z is ignored).
    moved = np.asarray(PARKED)
    moved[3, 0] += 0.01
    assert not _is_pinned_static_vehicle(_track(moved))
    z_only = np.asarray(PARKED)
    z_only[3, 2] += 0.5
    assert _is_pinned_static_vehicle(_track(z_only))
    # Vehicles only. Cones and pedestrians keep the existing path.
    assert not _is_pinned_static_vehicle(_track(PARKED, kind="TRAFFIC_CONE"))
    assert not _is_pinned_static_vehicle(_track(PARKED, kind="PEDESTRIAN"))
    # An explicit control_mode flag is followed as is.
    moving = [[float(i), 0.0, 0.0] for i in range(6)]
    assert _is_pinned_static_vehicle(_track(moving, control_mode="static"))
    assert not _is_pinned_static_vehicle(_track(moving, control_mode="idm_candidate"))
    assert not _is_pinned_static_vehicle({"type": "VEHICLE"})


class _Config(dict):
    __getattr__ = dict.__getitem__


def _manager(monkeypatch, pinned=frozenset(), policy="nuplan_idm_policy", parked=None):
    engine = SimpleNamespace(global_config=_Config(agent_policy=policy), _nuplan_idm_batch=None)
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._static_pinned_vehicle_ids = frozenset(pinned)
    # On reset the R set is derived from this parked set; in R with the switch on, the two are equal.
    manager._parked_vehicle_ids = frozenset(pinned if parked is None else parked)
    return manager


def test_pinned_vehicle_is_a_static_agent_and_never_sector_replay(monkeypatch):
    track = _track(PARKED)
    state = {"position": np.asarray(PARKED), "valid": np.ones(6, dtype=bool)}
    # Switch off: in R every vehicle is dynamic (IDM), the existing behaviour.
    assert not _manager(monkeypatch)._is_static_track("car", track, state)
    manager = _manager(monkeypatch, pinned={"car"})
    assert manager._is_static_track("car", track, state)
    manager._idm_spawn_clock = object()
    assert not manager._uses_reactive_sector_replay("car")


def test_parked_vehicle_survives_source_end_in_nr(monkeypatch):
    """A parked vehicle stays in place after the log ends, in NR (trajectory_policy) too.

    The simulation runs horizon_extension_factor times longer than the log. Removing parked
    vehicles at source end leaves an empty road afterwards; R already kept them, NR did not.
    """
    manager = _manager(monkeypatch, policy="trajectory_policy", parked={"car"})
    # The R-only switch set is empty in NR; only the parked-vehicle set is filled.
    assert manager._static_pinned_vehicle_ids == frozenset()

    manager._static_agents = {}
    # Before its window opens the vehicle is not spawned, so there is nothing to keep (spawning is another branch).
    assert not manager._survives_source_end("car")

    manager._static_agents = {"car": object()}
    assert manager._survives_source_end("car")
    # A moving vehicle disappears when its source ends.
    assert not manager._survives_source_end("bus")


@pytest.mark.parametrize("reason", ["rail_offset_4.00m", "source_path_le_3m"])
def test_predropped_vehicle_goes_through_the_spawn_drop_path(monkeypatch, reason):
    manager = _manager(monkeypatch)
    manager._spawn_overlap_dropped = set()
    manager._spawn_overlap_deferred = set()
    batch = SimpleNamespace(predropped_vehicles={"bad": reason})
    manager._drop_nuplan_predropped(batch, 0)
    manager._drop_nuplan_predropped(batch, 1)                 # recorded only once
    assert manager.spawn_deferred_tokens == frozenset({"bad"})  # the set the renderer receives
    assert manager.object_source_row("bad", 5) == -1
    assert [e["reason"] for e in manager.spawn_gate_events] == [f"predrop:{reason}"]
    # A batch without predrops (switch off, test stubs) does nothing.
    manager._drop_nuplan_predropped(SimpleNamespace(), 2)
    assert len(manager.spawn_gate_events) == 1


# --- batch ---------------------------------------------------------------------------------

def _obj(token, x, kind=TrackedObjectType.VEHICLE):
    return SimpleNamespace(
        track_token=token, tracked_object_type=kind,
        center=SimpleNamespace(x=x, y=0.0, heading=0.0,
                               point=SimpleNamespace(array=np.asarray([x, 0.0]))),
        velocity=SimpleNamespace(x=0.0, y=0.0))


class _Objects:
    def __init__(self, objects):
        self.tracked_objects = objects

    def get_tracked_objects_of_type(self, kind):
        return [o for o in self.tracked_objects if o.tracked_object_type == kind]

    def get_tracked_objects_of_types(self, kinds):
        return [o for o in self.tracked_objects if o.tracked_object_type in kinds]


class _Occupancy:
    def __init__(self, tokens):
        self.tokens = set(tokens)

    def contains(self, token):
        return token in self.tokens

    def remove(self, tokens):
        self.tokens -= set(tokens)

    def set(self, token, _geometry):
        self.tokens.add(token)


def _rail_agent(x):
    return SimpleNamespace(to_se2=lambda: SimpleNamespace(x=x, y=0.0, heading=0.0),
                           polygon=None)


def _batch(monkeypatch, *, predrop, gt_fallback=True):
    """Handoff-tick snapshot: pinned (parked), good (passes), far (5 m snap), refused (builder rejects)."""
    snapshot = SimpleNamespace(tracked_objects=_Objects([
        _obj("pinned", 0.0), _obj("good", 10.0), _obj("far", 20.0), _obj("refused", 30.0),
        _obj("cone", 40.0, TrackedObjectType.TRAFFIC_CONE),
    ]))
    seen = {}

    def fake_builder(*args):
        detections = args[6].initial_tracked_objects.tracked_objects
        seen["candidates"] = {o.track_token for o in
                              detections.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)}
        seen["obstacles"] = {o.track_token for o in
                             detections.get_tracked_objects_of_types(args[7])}
        built = {"good": _rail_agent(10.0), "far": _rail_agent(25.0)}
        if "pinned" in seen["candidates"]:
            built["pinned"] = _rail_agent(0.0)
        return built, _Occupancy(built)

    monkeypatch.setattr(policy_module, "build_idm_agents_on_map_rails", fake_builder)
    monkeypatch.setattr(policy_module, "IDMAgentManager",
                        lambda agents, occupancy, map_api: SimpleNamespace(
                            agents=agents, agent_occupancy=occupancy))
    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch.engine = SimpleNamespace(agent_manager=SimpleNamespace())
    batch._gt_warmup_steps = 3
    batch._scenario = SimpleNamespace(map_api=None,
                                      get_tracked_objects_at_iteration=lambda _i: snapshot)
    batch._ego_agent = SimpleNamespace(
        object_track={"position": np.asarray([[100.0, 0.0]] * 5), "heading": np.zeros(5)},
        length=4.0, width=2.0, height=1.5)
    batch._origin = np.zeros(2)
    batch._idm_params = {k: 1.0 for k in ("target_velocity", "min_gap_to_lead_agent",
                                          "headway_time", "accel_max", "decel_max",
                                          "minimum_path_length")}
    batch._obs = SimpleNamespace(_open_loop_detections_types=[TrackedObjectType.TRAFFIC_CONE],
                                 extra_open_loop_vehicle_tokens=set(),
                                 _idm_agent_manager=None)
    batch._unstable_short_track_tokens = set()
    batch._reactive_min_source_displacement_m = 0.0
    batch._parked_vehicle_fallback_enabled = False
    batch._handoff_smooth_merge_enabled = True
    batch._handoff_smooth_merge_max_offset_m = 3.0
    batch._handoff_smooth_merge_max_heading_deg = 45.0
    batch._install_handoff_merge_path = lambda track, agent: True
    batch._vehicle_gt_fallback_enabled = gt_fallback
    batch._retired = set()
    batch._handoff_smooth_merge_tokens = set()
    batch._handoff_gt_fallback_tokens = set()
    batch._handoff_routable_not_built_tokens = set()
    batch._handoff_dropped_routable_tokens = set()
    batch._seed_idm_initial_speed = lambda _token, _agent, _step: None
    batch._spawn_sector_waiting_tokens = set()
    batch._source_motion_open_loop_tokens = set()
    if predrop:
        batch._static_pin_and_predrop = True
        batch._static_vehicle_tokens = frozenset({"pinned"})
        batch._obs.extra_open_loop_vehicle_tokens.add("pinned")
        batch._predropped = {}
    batch._prebuild_idm_handoff()
    return batch, seen


def test_handoff_prebuild_pins_obstacles_and_predrops_gate_failures(monkeypatch):
    batch, seen = _batch(monkeypatch, predrop=True)
    # A parked vehicle is an obstacle with its logged box, not a builder candidate.
    assert seen["candidates"] == {"good", "far", "refused"}
    assert seen["obstacles"] == {"cone", "pinned"}
    assert set(batch._obs._idm_agent_manager.agents) == {"good"}
    # Gate failures are predropped, not GT fallback (even with gt_fallback=True), and never re-admitted.
    assert batch.predropped_vehicles.keys() == {"far", "refused"}
    assert batch.predropped_vehicles["far"].startswith("rail_offset_")
    assert batch._handoff_gt_fallback_tokens == set()
    assert {"far", "refused"} <= batch._retired
    assert not batch.is_gt_replay_vehicle("far")
    # Parked vehicle: always materialized, labelled static, never sector-replayed.
    batch._poses, batch._source_modes = {}, {}
    assert batch.can_materialize("pinned")
    assert batch.source_mode_for("pinned") == "static"
    assert not batch.is_gt_replay_vehicle("pinned")


def test_predropped_vehicles_are_absent_from_gt_warmup(monkeypatch):
    batch, _ = _batch(monkeypatch, predrop=True)
    batch._publish_gt_warmup(0)
    assert set(batch._poses) == {"pinned", "good", "cone"}
    assert not batch.can_materialize("far") and not batch.can_materialize("refused")


def test_switch_off_keeps_the_gt_fallback_behaviour(monkeypatch):
    batch, seen = _batch(monkeypatch, predrop=False)
    assert seen["candidates"] == {"pinned", "good", "far", "refused"}
    assert set(batch._obs._idm_agent_manager.agents) == {"pinned", "good"}
    assert batch.predropped_vehicles == {}
    assert batch._handoff_gt_fallback_tokens == {"far", "refused"}
    batch._publish_gt_warmup(0)
    assert set(batch._poses) == {"pinned", "good", "far", "refused", "cone"}
    assert batch.source_mode_for("pinned") == "gt_warmup"


@pytest.mark.parametrize("predrop", [False, True])
def test_gt_warmup_always_runs_before_the_idm_handoff(monkeypatch, predrop):
    """Warm-up (< _gt_warmup_steps) is pure GT replay regardless of the switch; IDM runs only after it.

    With the switch on there are only two differences: parked vehicles are labelled 'static' and
    predropped vehicles are absent. All other vehicles stay 'gt_warmup', never 'idm', during warm-up.
    """
    batch, _ = _batch(monkeypatch, predrop=predrop)
    batch._step, batch._idm_start_step, batch._solve_stride = -1, None, 1
    batch._scenario.set_initial_iteration = lambda _i: None
    solved = []
    batch._solve_once = lambda step, prev, ego_step: solved.append(step)

    for step in range(batch._gt_warmup_steps):
        batch._advance(step)
        assert solved == [], "IDM ran during GT warm-up"
        modes = {tok: batch.source_mode_for(tok) for tok in batch._poses}
        if predrop:
            assert modes == {"pinned": "static", "good": "gt_warmup", "cone": "gt_warmup"}
        else:
            assert set(modes.values()) == {"gt_warmup"}
            assert set(modes) == {"pinned", "good", "far", "refused", "cone"}
        assert all(mode != "idm" for mode in modes.values())

    batch._advance(batch._gt_warmup_steps)
    assert solved == [batch._gt_warmup_steps]


@pytest.mark.parametrize("pinned", [True, False])
def test_pinned_static_vehicle_persists_past_the_source_horizon(monkeypatch, pinned):
    """The log ends after 10 rows (0..9) but the episode continues. A pinned vehicle stays; other
    static objects (switch-off behaviour) still disappear when their source ends."""
    class _Agent:
        destroyed = False

        def step(self):
            pass

        def destroy(self):
            self.destroyed = True

    scene = {"object_track": {"car": _track(PARKED * 2)}}
    batch = SimpleNamespace(prepare_step=lambda _s: None, predropped_vehicles={},
                            can_materialize=lambda _t: True)
    engine = SimpleNamespace(
        episode_step=12, sim_dt=0.1, current_scene=scene, _nuplan_idm_batch=batch,
        global_config=_Config(agent_policy="nuplan_idm_policy", parallel_ego_idm=False))
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    monkeypatch.setattr(policy_module, "_batch", lambda _engine, _config: batch)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._static_pinned_vehicle_ids = frozenset({"car"} if pinned else ())
    manager._dynamic_agents, manager._static_agents = {}, {"car": _Agent()}
    manager._agent_valid_periods = {"car": [(0, 9)]}
    manager._replay = None
    manager._spawn_overlap_dropped = set()

    for step in (10, 11, 12):
        engine.episode_step = step
        manager.step()

    assert ("car" in manager.all_agents) is pinned


# --- IDM propagate failure ----------------------------------------------------------------

def _failing_batch(tmp_path, fail):
    """State that trips the self-intersection assert in propagate_agents.
      zero0000  a parked vehicle entered IDM with a 0x0 box -> EMPTY path buffer.
      offender  path at y=10 but occupancy box at the origin (8 m from the buffer).
      fine      on its own path -> must not be listed."""
    from shapely.geometry import box
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMap

    def agent(y_path, footprint, width=2.0):
        return SimpleNamespace(
            is_active=lambda _s: True, has_valid_path=lambda: True, width=width, length=width,
            get_path_to_go=lambda: [StateSE2(-10.0, y_path, 0.0), StateSE2(10.0, y_path, 0.0)],
            to_se2=lambda: StateSE2(0.0, 0.0, 0.0), polygon=footprint)

    occupancy = STRTreeOccupancyMap({})
    offender_box, fine_box = box(-2, -1, 2, 1), box(-2, 29, 2, 31)
    point_box = box(50, 0, 50, 0)
    occupancy.insert("zero0000pinned", point_box)
    occupancy.insert("offender1234", offender_box)
    occupancy.insert("fine", fine_box)
    mgr = SimpleNamespace(agents={"zero0000pinned": agent(0.0, point_box, width=0.0),
                                  "offender1234": agent(10.0, offender_box),
                                  "fine": agent(30.0, fine_box)},
                          agent_occupancy=occupancy)

    def boom(_step):
        raise AssertionError("Agent's baseline does not intersect the agent itself")

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._deferred_tokens = set()
    batch.converter = SimpleNamespace(convert_to_current_ego_state=boom)
    batch._obs = SimpleNamespace(_get_idm_agent_manager=lambda: mgr)
    batch._source_modes = {"offender1234": "idm", "zero0000pinned": "gt_warmup"}
    # Even with the switch off, pinned_static reports the scenario-track verdict.
    batch.scene = {"object_track": {"zero0000pinned": _track(PARKED),
                                    "offender1234": _track([[float(i), 0, 0] for i in range(6)])}}
    batch._static_vehicle_tokens = frozenset()
    batch._fail_on_propagate_error = fail
    batch._failure_output_dir = str(tmp_path)
    return batch


def test_idm_propagate_failure_fails_the_rollout_and_names_the_agent(tmp_path):
    import json
    batch = _failing_batch(tmp_path, fail=True)
    with pytest.raises(policy_module.IDMPropagateError) as raised:
        batch._solve_once(15, 14, 15)
    assert str(raised.value) == ("IDM_PROPAGATE_FAILED step 15: Agent's baseline does not "
                                 "intersect the agent itself (agent zero0000 +1)")
    record = json.loads((tmp_path / "idm_failure.json").read_text())
    assert record["step"] == 15 and record["exception_type"] == "AssertionError"
    assert record["summary"] == str(raised.value)
    assert "AssertionError" in record["traceback"]
    zero, offender = record["agents"]                 # 'fine' satisfies the invariant, so it is omitted
    assert zero["token"] == "zero0000pinned" and zero["asserted_on"] is True
    assert zero["reason"] == "zero_size_box" and zero["path_buffer_empty"] is True
    assert zero["box_length_width_m"] == [0.0, 0.0] and zero["occupancy_to_path_buffer_m"] is None
    assert zero["pinned_static"] is True and zero["flag_static_pin_on"] is False
    assert offender["token"] == "offender1234" and "asserted_on" not in offender
    assert offender["reason"] == "occupancy_off_path"
    assert offender["source_mode"] == "idm" and offender["pinned_static"] is False
    assert offender["occupancy_to_path_buffer_m"] == pytest.approx(8.0)
    assert offender["occupancy_area_m2"] == pytest.approx(8.0)
    assert len(offender["pose_xy_heading"]) == 3


def test_idm_propagate_failure_knob_off_keeps_the_legacy_swallow(tmp_path):
    batch = _failing_batch(tmp_path, fail=False)
    assert batch._solve_once(15, 14, 15) is None
    assert not (tmp_path / "idm_failure.json").exists()


def test_failing_on_propagate_error_is_the_default():
    import pathlib, yaml
    cfg = yaml.safe_load((pathlib.Path(policy_module.__file__).parents[3]
                          / "configs/default_runner.yaml").read_text(encoding="utf-8"))
    assert cfg["nuplan_idm_fail_on_propagate_error"] is True
    assert NuPlanIDMBatch._fail_on_propagate_error is True


# --- scenario builder: box sizes on checkpoint-only valid rows -----------------------------

def _sized_track(lengths, valid, render_status="positive_gaussian"):
    n = len(lengths)
    lengths = np.asarray(lengths, dtype=float).reshape(n, 1)
    return {"type": "VEHICLE",
            "metadata": {"render_status": render_status, "simulation_enabled": True},
            "state": {"position": np.arange(3 * n, dtype=float).reshape(n, 3),
                      "heading": np.linspace(0, 1, n), "velocity": np.ones((n, 2)),
                      "valid": np.asarray(valid, dtype=float),
                      "length": lengths.copy(), "width": lengths / 2,
                      "height": np.where(lengths > 0, 1.5, 0.0)}}



# --- <=3 m source-path admission / scored-snapshot exclusion ------------------------------

def test_source_path_predrop_uses_valid_cumulative_distance():
    def source(points, valid, *, mode="idm_candidate", enabled=True, kind="VEHICLE"):
        return {"type": kind, "metadata": {
            "control_mode": mode, "simulation_enabled": enabled,
        }, "state": {
            "position": np.asarray(points, dtype=float),
            "valid": np.asarray(valid, dtype=bool),
        }}

    scene = {"object_track": {
        "partial_stationary": source([[0, 0], [5, 0], [5, 0]], [0, 1, 1]),
        "at_limit": source([[0, 0], [3, 0]], [1, 1]),
        "above_limit": source([[0, 0], [3.01, 0]], [1, 1]),
        # Endpoint displacement is 0, but the actual source path is 4 m.
        "returning": source([[0, 0], [2, 0], [0, 0]], [1, 1, 1]),
        "pinned": source([[10, 0], [10, 0]], [1, 1]),
        "replay": source([[0, 0], [0, 0]], [1, 1], mode="replay"),
        "disabled": source([[0, 0], [0, 0]], [1, 1], enabled=False),
        "pedestrian": source([[0, 0], [0, 0]], [1, 1], kind="PEDESTRIAN"),
    }}
    lengths = NuPlanIDMBatch._build_source_vehicle_path_lengths(scene)
    assert lengths["partial_stationary"] == 0.0
    assert lengths["at_limit"] == pytest.approx(3.0)
    assert lengths["returning"] == pytest.approx(4.0)
    assert "replay" not in lengths and "disabled" not in lengths
    assert "pedestrian" not in lengths

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._static_pin_and_predrop = True
    batch._source_path_predrop_max_m = 3.0
    batch._static_vehicle_tokens = frozenset({"pinned"})
    batch._source_vehicle_path_lengths = lengths
    batch._retired = set()
    batch._predropped = {}
    batch._predrop_short_source_paths()
    assert batch.predropped_vehicles == {
        "partial_stationary": "source_path_le_3m",
        "at_limit": "source_path_le_3m",
    }
    assert set(batch._retired) == {"partial_stationary", "at_limit"}


def test_source_path_predrop_never_reaches_builder_or_gt_snapshot():
    snapshot = SimpleNamespace(tracked_objects=_Objects([
        _obj("pinned", 0.0), _obj("short", 5.0), _obj("moving", 10.0),
        _obj("cone", 20.0, TrackedObjectType.TRAFFIC_CONE),
    ]))
    view = policy_module._PinnedVehiclesAsObstacles(
        snapshot, frozenset({"pinned"}), {"short": "source_path_le_3m"}
    )
    assert {obj.track_token for obj in view.get_tracked_objects_of_type(
        TrackedObjectType.VEHICLE)} == {"moving"}
    assert {obj.track_token for obj in view.get_tracked_objects_of_types(
        [TrackedObjectType.TRAFFIC_CONE])} == {"pinned", "cone"}

    batch = NuPlanIDMBatch.__new__(NuPlanIDMBatch)
    batch._obs = SimpleNamespace(_idm_agent_manager=object())
    batch._scenario = SimpleNamespace(get_tracked_objects_at_iteration=lambda _step: snapshot)
    batch._origin = np.zeros(2)
    batch._predropped = {"short": "source_path_le_3m"}
    batch._publish_gt_warmup(0)
    assert "short" not in batch._poses
    assert set(batch._poses) == {"pinned", "moving", "cone"}
