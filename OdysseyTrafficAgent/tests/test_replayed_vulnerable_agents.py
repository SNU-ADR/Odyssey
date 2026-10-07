"""Reconstructed pedestrians and bicycles share the planner/scorer actor snapshot."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from odyssey.manager import agent_manager as agent_manager_module
from odyssey.manager import base_manager as base_manager_module
from odyssey.manager.agent_manager import BaseAgentManager
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD
from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import OdysseyToNuPlanConverter
from odyssey_renderer.omnire.smpl_object import OmniReSMPLSubModel


class _Config(dict):
    __getattr__ = dict.__getitem__



def test_smpl_root_translation_respects_unnormalized_point_weights():
    model = OmniReSMPLSubModel.__new__(OmniReSMPLSubModel)
    torch.nn.Module.__init__(model)
    model.gauss_params = {"means": torch.zeros(2, 3)}
    model._root_joint = torch.tensor([1., 0., 0.])
    model._lbs_weight_sum = torch.tensor([.5, 2.])
    model._frame = None
    model._posed = lambda frame: (None, None)
    half = np.pi / 4
    quat = torch.tensor([np.cos(half), 0., 0., np.sin(half)], dtype=torch.float32)

    got = model.get_means(quat, torch.tensor([10., 20., 0.]))

    torch.testing.assert_close(
        got, torch.tensor([[10.5, 19.5, 0.], [12., 18., 0.]]),
        rtol=0, atol=1e-6)


def test_simulation_disabled_actor_stays_in_scene_but_not_manager(monkeypatch):
    scene = {"metadata": {"actor_pose_source": "checkpoint"}, "object_track": {
        "ego": {"type": "VEHICLE", "metadata": {"simulation_enabled": True}},
        "zero": {"type": "VEHICLE", "metadata": {
            "simulation_enabled": False, "render_status": "zero_gaussian",
            "control_mode": "disabled"}},
    }}
    engine = SimpleNamespace(
        current_scene=scene, episode_step=0,
        global_config=_Config(max_step=2, visualize_BEV=False,
                              with_render_manager=True, agent_policy="trajectory_policy"))
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._BEV_vis = None
    manager._dynamic_agents = {}
    manager._static_agents = {}
    spawned = []
    monkeypatch.setattr(manager, "_process_agent_track", lambda token, track: ({}, [(0, 2)]))
    monkeypatch.setattr(manager, "spawn_agent", lambda token, track: spawned.append(token))

    manager.reset()

    assert set(scene["object_track"]) == {"ego", "zero"}
    assert spawned == ["ego"]
    assert set(manager._agent_valid_periods) == {"ego"}
    assert set(manager._trajectory_buffer) == {"ego"}


@pytest.mark.parametrize("pose_source,expected", [
    ("checkpoint", {"car", "ped", "bike"}),
    (None, {"car"}),
])
def test_rendered_reset_only_admits_verified_checkpoint_actors(monkeypatch, pose_source, expected):
    scene = {"metadata": {"actor_pose_source": pose_source}, "object_track": {
        "car": {"type": "VEHICLE"}, "ped": {"type": "PEDESTRIAN"},
        "bike": {"type": "BICYCLE"}}}
    engine = SimpleNamespace(
        current_scene=scene, episode_step=0,
        global_config=_Config(max_step=2, visualize_BEV=False,
                              with_render_manager=True, agent_policy="trajectory_policy"))
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._BEV_vis = None
    manager._dynamic_agents = {}
    manager._static_agents = {}
    spawned = []
    monkeypatch.setattr(manager, "_process_agent_track", lambda token, track: ({}, [(0, 2)]))
    monkeypatch.setattr(manager, "spawn_agent", lambda token, track: spawned.append(token))

    manager.reset()

    assert set(manager._agent_valid_periods) == expected
    assert set(spawned) == expected
    assert set(manager._trajectory_buffer) == expected


@pytest.mark.parametrize("kind", ["PEDESTRIAN", "CYCLIST", "BICYCLE"])
def test_vulnerable_actor_uses_source_replay_even_in_idm_run(monkeypatch, kind):
    created = []

    class FakeAgent:
        def __init__(self, token, state, name, config, traj_step):
            created.append(self)
            self.id = token
            self.config = config
            self.traj_step = traj_step

        def reset(self):
            pass

    scene = {"adv_object_id": None}
    engine = SimpleNamespace(
        episode_step=0,
        global_config=_Config(agent_policy="nuplan_idm_policy", agent_controller="log_play_controller"),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
    )
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    monkeypatch.setattr(agent_manager_module, "BaseAgent", FakeAgent)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._dynamic_agents = {}
    manager._static_agents = {}
    manager._agent_valid_periods = {}
    state = {"position": np.array([[1., 2.], [1.1, 2.]]),
             "valid": np.ones((2, 1))}
    monkeypatch.setattr(manager, "_process_agent_track", lambda token, track: (state, [(0, 1)]))

    manager.spawn_agent("actor", {"type": kind})

    assert len(created) == 1
    assert "actor" in manager._dynamic_agents
    assert created[0].config["agent_policy"] == "trajectory_policy"
    assert created[0].config["agent_controller"] == "log_play_controller"


def test_nuplan_admission_does_not_prune_source_replayed_actors(monkeypatch):
    engine = SimpleNamespace(current_scene={"object_track": {
        "ped": {"type": "PEDESTRIAN"}, "bike": {"type": "BICYCLE"},
        "car": {"type": "VEHICLE"}}})
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    destroyed = []
    manager._dynamic_agents = {
        token: SimpleNamespace(policy=SimpleNamespace(), destroy=lambda t=token: destroyed.append(t))
        for token in ("ped", "bike", "car")}
    manager._static_agents = {}
    manager._prune_unmaterialized_nuplan_proxies(
        SimpleNamespace(can_materialize=lambda token: False), 0)

    assert set(manager.all_agents) == {"ped", "bike"}
    assert destroyed == ["car"]


def test_late_pedestrian_spawn_does_not_require_vehicle_admission(monkeypatch):
    scene = {"log_length": 10}
    source = {"ped": {"type": "PEDESTRIAN"}}
    batch = SimpleNamespace(
        scene=scene, prepare_step=lambda step: None,
        can_materialize=lambda token: False)
    engine = SimpleNamespace(
        episode_step=2, current_scene={"object_track": source},
        global_config=_Config(agent_policy="nuplan_idm_policy"),
        managers={"scenario_manager": SimpleNamespace(current_scene=scene)},
        _nuplan_idm_batch=batch)
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = None
    manager._dynamic_agents = {}
    manager._static_agents = {}
    manager._agent_valid_periods = {"ped": [(2, 4)]}
    spawned = []
    monkeypatch.setattr(manager, "spawn_agent", lambda token, track, traj_step=None: spawned.append(token))

    manager.step()

    assert spawned == ["ped"]


def test_shared_actor_observation_keeps_vehicle_pedestrian_and_bicycle():
    kinds = {"car": "VEHICLE", "ped": "PEDESTRIAN", "bike": "BICYCLE"}
    agents = {
        token: SimpleNamespace(
            id=token, current_position=np.array([float(i), 0.]),
            current_heading=0., current_velocity=np.array([1., 0.]),
            _length=4. if token == "car" else 1., _width=1., _height=1.)
        for i, token in enumerate(kinds)}
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.scene = {"token": "scene", "sdc_id": "ego",
                       "object_track": {token: {"type": kind} for token, kind in kinds.items()},
                       "cadence": SimpleNamespace(sim_dt=0.1)}
    converter.engine = SimpleNamespace(agent_manager=SimpleNamespace(all_agents=agents), managers={})
    converter.base_timestamp = 0
    converter.initial_ego_center = np.zeros(2)

    tracks = converter.convert_to_detections_tracks_from_agent_input(3).tracked_objects
    observed = {obj.track_token: obj for obj in tracks.tracked_objects}

    assert set(observed) == set(kinds)
    assert all(isinstance(obj, Agent) for obj in observed.values())
    assert observed["ped"].tracked_object_type == TrackedObjectType.PEDESTRIAN
    assert observed["bike"].tracked_object_type == TrackedObjectType.BICYCLE


def test_scene_observation_excludes_simulation_disabled_actor():
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    state = {
        "position": np.array([[1., 1., 0.]]),
        "heading": np.zeros(1), "velocity": np.zeros((1, 2)),
        "length": np.ones((1, 1)) * 4., "width": np.ones((1, 1)) * 2.,
        "height": np.ones((1, 1)) * 1.5,
    }
    converter.scene = {
        "token": "scene", "sdc_id": "ego",
        "cadence": SimpleNamespace(sim_dt=0.1),
        "object_track": {
            "ego": {"type": "VEHICLE", "state": state, "metadata": {}},
            "live": {"type": "VEHICLE", "state": state, "metadata": {
                "simulation_enabled": True, "nuplan_id": "live"}},
            "zero": {"type": "VEHICLE", "state": state, "metadata": {
                "simulation_enabled": False, "nuplan_id": "zero"}},
        },
    }
    converter.base_timestamp = 0
    converter.initial_ego_center = np.zeros(2)

    tracks = converter.convert_to_detections_tracks_from_scene(0).tracked_objects

    assert [obj.track_token for obj in tracks.tracked_objects] == ["live"]


# --- spawn-overlap gate (trajectory_policy) --------------------------------------------

def _gate_manager(monkeypatch, *, gate, actor_xy, ego_xy=(0., 0.),
                  periods=((0, 11),), kind="VEHICLE", rows=None, step=5,
                  action="drop"):
    """A trajectory_policy manager with one ego and one candidate actor.

    Poses are in the scenario's own frame -- the same frame the track is stored in --
    because that is what the gate compares. ego 5.176 x 2.297 (pacifica), actor 4.5 x 2.0,
    so the boxes touch at 4.838 m of separation and overlap below it.
    """
    n = 12
    track = {"type": kind, "state": {
        SD.POSITION: np.array([[actor_xy[0], actor_xy[1], 0.]] * n),
        SD.HEADING: np.zeros(n),
        "length": np.full(n, 4.5), "width": np.full(n, 2.0),
        SD.VALID: np.ones(n, bool)}}
    engine = SimpleNamespace(
        episode_step=step, current_scene={"object_track": {"a": track}},
        global_config=_Config(agent_policy="trajectory_policy",
                              trajectory_spawn_ego_overlap_gate=gate,
                              trajectory_spawn_ego_overlap_action=action))
    monkeypatch.setattr(base_manager_module, "get_engine", lambda: engine)
    manager = BaseAgentManager.__new__(BaseAgentManager)
    manager._replay = (None if rows is None
                       else SimpleNamespace(step=lambda s, xy: dict(rows)))
    manager._idm_lifecycle_clock = None
    manager._dynamic_agents = {"ego": SimpleNamespace(
        current_position=np.array(ego_xy, dtype=float), current_heading=0.,
        length=5.176, width=2.297, step=lambda: None)}
    manager._static_agents = {}
    manager._agent_valid_periods = {"a": list(periods)}
    manager._spawn_overlap_deferred = set()
    manager._spawn_overlap_deferred_ever = set()
    manager._spawn_overlap_dropped = set()
    manager._spawn_overlap_deferral_ticks = 0
    spawned = []
    monkeypatch.setattr(manager, "spawn_agent",
                        lambda token, track, traj_step=None: spawned.append((token, traj_step)))
    return manager, spawned, engine


@pytest.mark.parametrize("policy", ["trajectory_policy", "nuplan_idm_policy"])
def test_both_policies_read_the_same_gate(monkeypatch, policy):
    """The gate must be the same for both arms.

    If R ignored this setting, gate=false would admit overlapping spawns only under NR, and the same
    config would give the two arms different worlds.
    """
    manager, _, _ = _gate_manager(monkeypatch, gate=False, actor_xy=(0.5, 0.))
    manager.engine.global_config["agent_policy"] = policy
    assert manager._spawn_gate_settings() == (False, True)

    manager.engine.global_config["trajectory_spawn_ego_overlap_gate"] = True
    assert manager._spawn_gate_settings() == (True, True)


def test_defer_under_idm_fails_loudly(monkeypatch):
    """defer is implemented only for trajectory_policy; under R it must not silently fall back to drop."""
    manager, _, _ = _gate_manager(monkeypatch, gate=True, actor_xy=(0.5, 0.), action="defer")
    assert manager._spawn_gate_settings() == (True, False)      # NR can still use it

    manager.engine.global_config["agent_policy"] = "nuplan_idm_policy"
    with pytest.raises(ValueError, match="defer"):
        manager._spawn_gate_settings()


def test_spawn_gate_off_lets_an_overlapping_actor_in(monkeypatch):
    """The flag is off by default and must reproduce the old behaviour exactly."""
    manager, spawned, _ = _gate_manager(monkeypatch, gate=False, actor_xy=(0.5, 0.))

    manager.step()

    assert [t for t, _ in spawned] == ["a"]
    assert manager.spawn_deferred_tokens == frozenset()


def test_an_actor_overlapping_the_ego_is_held_back_and_published_as_suppressed(monkeypatch):
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="defer")

    manager.step()

    assert spawned == []
    assert manager.spawn_deferred_tokens == frozenset({"a"})
    assert "a" not in manager.all_agents
    assert manager.spawn_overlap_stats == {
        "deferral_ticks": 1, "deferred_actors": 1, "dropped_actors": 0,
        "dropped_tokens": []}


def test_a_held_back_actor_enters_as_soon_as_the_ego_is_clear(monkeypatch):
    """Retry needs no queue: the actor is simply still absent next step."""
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="defer")

    manager.step()
    assert spawned == []

    manager._dynamic_agents["ego"].current_position = np.array([-30., 0.])
    engine.episode_step = 6
    manager.step()

    assert [t for t, _ in spawned] == ["a"]
    assert manager.spawn_deferred_tokens == frozenset()
    assert manager.spawn_overlap_stats["dropped_actors"] == 0


def test_an_actor_whose_window_closes_while_held_back_never_appears(monkeypatch):
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), periods=((0, 5),), action="defer")

    manager.step()                      # step 5: still valid, overlapping -> held
    assert manager.spawn_deferred_tokens == frozenset({"a"})

    engine.episode_step = 6             # window closed
    manager.step()

    assert spawned == []
    assert manager.spawn_overlap_stats["dropped_actors"] == 1
    # Still suppressed, not released: the set is what the renderer must not draw, and this
    # actor is now out for good. A track whose valid periods reopen must not come back
    # with it.
    assert manager.spawn_deferred_tokens == frozenset({"a"})


def test_drop_mode_removes_the_actor_from_the_rollout_for_good(monkeypatch):
    """`drop` abandons the entry rather than moving it: the ego clearing the way later
    does not bring the actor back, because the row the log chose has already passed."""
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="drop")

    manager.step()
    assert spawned == []
    assert manager.spawn_overlap_stats["dropped_actors"] == 1
    assert manager.spawn_overlap_stats["deferral_ticks"] == 0

    manager._dynamic_agents["ego"].current_position = np.array([-30., 0.])
    for step in (6, 7, 8):
        engine.episode_step = step
        manager.step()

    assert spawned == []
    assert manager.spawn_overlap_stats["dropped_actors"] == 1


def test_a_dropped_actor_stays_out_of_the_rendered_image_too(monkeypatch):
    """Absent from the simulation but drawn anyway is the ghost the render hook exists to
    prevent, and a drop lasts the whole rollout, so the suppression has to as well."""
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="drop")

    manager.step()
    assert manager.spawn_deferred_tokens == frozenset({"a"})

    manager._dynamic_agents["ego"].current_position = np.array([-30., 0.])
    engine.episode_step = 6
    manager.step()

    assert manager.spawn_deferred_tokens == frozenset({"a"})
    assert spawned == []


def test_the_stats_name_the_dropped_actors_not_just_count_them(monkeypatch):
    """A dropped actor leaves no box in the rollout, so the token is the only record that it
    was ever supposed to be there."""
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="drop")

    manager.step()

    stats = manager.spawn_overlap_stats
    assert stats["dropped_tokens"] == ["a"]
    assert stats["dropped_actors"] == len(stats["dropped_tokens"])


def test_a_clear_run_reports_no_dropped_tokens(monkeypatch):
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(30., 0.), action="drop")

    manager.step()

    assert manager.spawn_overlap_stats["dropped_tokens"] == []


def test_drop_mode_leaves_a_clear_spawn_untouched(monkeypatch):
    """The gate still only reads the one thing it claims to read."""
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(30., 0.), action="drop")

    manager.step()

    assert [t for t, _ in spawned] == ["a"]
    assert manager.spawn_deferred_tokens == frozenset()
    assert manager.spawn_overlap_stats["dropped_actors"] == 0


def test_tight_ahead_spawn_is_permanently_dropped_and_audited(monkeypatch):
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(7.0, 0.0), action="drop")
    engine.global_config["spawn_ego_tight_ahead_gate"] = True
    engine.global_config["spawn_ego_tight_ahead_gap_m"] = 5.0

    manager.step()

    assert spawned == []
    assert manager.spawn_overlap_stats["dropped_tokens"] == ["a"]
    assert manager.spawn_gate_events == ({
        "simulation_step": 5,
        "token": "a",
        "source_row": 5,
        "decision": "drop",
        "reason": "tight_ahead",
    },)


def test_drop_mode_does_nothing_while_the_gate_is_off(monkeypatch):
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=False, actor_xy=(0.5, 0.), action="drop")

    manager.step()

    assert [t for t, _ in spawned] == ["a"]
    assert manager.spawn_overlap_stats["dropped_actors"] == 0


@pytest.mark.parametrize("action", [None, "nonsense"])
def test_an_unset_or_unrecognised_action_drops(monkeypatch, action):
    """Drop is the default and the fallback: a config that predates the key, or carries a
    typo in it, must not quietly get the other behaviour."""
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="drop")
    if action is None:
        del manager.engine.global_config["trajectory_spawn_ego_overlap_action"]
    else:
        manager.engine.global_config["trajectory_spawn_ego_overlap_action"] = action

    manager.step()
    assert manager.spawn_overlap_stats["dropped_actors"] == 1

    manager._dynamic_agents["ego"].current_position = np.array([-30., 0.])
    engine.episode_step = 6
    manager.step()

    assert spawned == []


def test_defer_is_still_available_when_asked_for_by_name(monkeypatch):
    manager, spawned, engine = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), action="defer")

    manager.step()
    assert manager.spawn_overlap_stats["dropped_actors"] == 0

    manager._dynamic_agents["ego"].current_position = np.array([-30., 0.])
    engine.episode_step = 6
    manager.step()

    assert [t for t, _ in spawned] == ["a"]


@pytest.mark.parametrize("kind", ["TRAFFIC_CONE", "BARRIER", "PEDESTRIAN", "BICYCLE"])
def test_the_gate_reads_geometry_not_actor_type(monkeypatch, kind):
    """Cones and barriers are a quarter of the measured cases and land in _static_agents,
    so the gate has to run before that split and without a type test."""
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), kind=kind)

    manager.step()

    assert spawned == []
    assert manager.spawn_deferred_tokens == frozenset({"a"})


def test_an_actor_already_in_the_world_is_never_removed_by_the_gate(monkeypatch):
    """The gate refuses entries; it does not evict. Pulling a live actor out mid-episode
    would change the world the planner has already been reacting to."""
    manager, spawned, _ = _gate_manager(monkeypatch, gate=True, actor_xy=(0.5, 0.))
    live = SimpleNamespace(destroy=lambda: pytest.fail("a live actor was destroyed"),
                           step=lambda: None)
    manager._dynamic_agents["a"] = live

    manager.step()

    assert "a" in manager._dynamic_agents
    assert manager.spawn_deferred_tokens == frozenset()


def test_the_gate_measures_the_row_the_actor_would_enter_at(monkeypatch):
    """Under a replay clock the presence row is not the wall clock, and the box the gate
    tests has to be the box that would actually be placed."""
    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(0.5, 0.), rows={"a": 3.0})

    manager.step()

    assert spawned == []
    assert manager.spawn_deferred_tokens == frozenset({"a"})


def test_the_gate_is_inert_without_an_ego(monkeypatch):
    """No ego, no footprint to compare against -- admit rather than crash."""
    manager, spawned, _ = _gate_manager(monkeypatch, gate=True, actor_xy=(0.5, 0.))
    del manager._dynamic_agents["ego"]

    manager.step()

    assert [t for t, _ in spawned] == ["a"]


def test_the_gate_uses_the_ego_box_centre_not_the_rear_axle(monkeypatch):
    """Regression for a real 1.46 m error.

    `ds_states` stores the REAR AXLE pose, and an audit that drew the ego box there
    reported overlaps the simulator never saw and missed ones it did. The gate reads
    `ego.current_position`, which is the box centre (vehicle_utils.RearVehicle derives the
    axle from it, not the other way round). These coordinates are measured from the scene
    that exposed the bug: at the centre the boxes are clear, at the rear axle they are not.
    """
    rear_axle_to_centre = 5.176 / 2 - 1.127          # pacifica: half_length - rear_length
    assert round(rear_axle_to_centre, 3) == 1.461

    # The actor sits 5.0 m ahead, just clear of the ego's 4.838 m half-sum. Read the ego's
    # pose as a centre and the boxes do not touch; read the same number as an axle and the
    # centre belongs 1.461 m further forward, where they overlap by 2.6 m^2.
    manager, spawned, _ = _gate_manager(monkeypatch, gate=True, actor_xy=(5.0, 0.))
    manager.step()
    assert [t for t, _ in spawned] == ["a"], "centre-referenced ego box is clear here"

    manager, spawned, _ = _gate_manager(
        monkeypatch, gate=True, actor_xy=(5.0, 0.), ego_xy=(rear_axle_to_centre, 0.))
    manager.step()
    assert spawned == [], "the same pose read as an axle overlaps, and must be held back"
