"""Episode DS contract shared by live simulation, offline scoring and verification.

Dense observed states are separate from the legacy uniform PDM proposal buffers.
The serialized inputs include HD polygons and the directed SD graph: replay never
reconstructs a reference from today's scene/map/sidecar.
"""
import json
import logging
import math
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.wkb import loads as load_wkb
from .pinned_actors import UNSET_POSE_JUMP_M, drop_unset_pose_frames  # noqa: F401

VERSION = 'dense'

logger = logging.getLogger(__name__)

#: Values stored in older archives (driving_inputs_json) for what is now spelled differently.
#: replay() maps them before scoring, so every rollout_trajectory.npz written since the inputs
#: were first pinned rescores unchanged.
LEGACY_VALUES = {
    'term_reason': {'gt_reached': 'destination_arrival', 'budget_exhausted': 'time_limit',
                    'sd_route_departed': 'route_deviation', 'log_exhausted': 'log_end'},
    'lane_rule': {'v4_penalty': 'plc'},
    'rc_method': {'sd_sdf_prefix_1m_v1': 'hmm_prefix_1m'},
    'tl_version': {'tlc_pinned_events_v3': 'tl_events_v3', 'tlc_pinned_events_v2': 'tl_events_v2',
                   'tlc_pinned_events_v1': 'tl_events_v1'},
}


def normalize_legacy_inputs(data):
    """Rewrite pre-rename vocabulary inside pinned driving inputs, in place. Returns data."""
    data['term_reason'] = LEGACY_VALUES['term_reason'].get(data.get('term_reason'), data.get('term_reason'))
    rules = data.get('rules') or {}
    if rules.get('lane') in LEGACY_VALUES['lane_rule']:
        rules['lane'] = LEGACY_VALUES['lane_rule'][rules['lane']]
    rc = data.get('rc') or {}
    if rc.get('method') in LEGACY_VALUES['rc_method']:
        rc['method'] = LEGACY_VALUES['rc_method'][rc['method']]
    tlc = data.get('tlc') or {}
    if tlc.get('version') in LEGACY_VALUES['tl_version']:
        tlc['version'] = LEGACY_VALUES['tl_version'][tlc['version']]
    return data


def pack_graph(g):
    packed = {k: [x.tolist() if isinstance(x, (np.ndarray, np.generic)) else x
                  for x in getattr(g, k)]
              for k in ('geom', 'u', 'v', 'twin', 'length', 'hw')}
    packed['active'] = list(getattr(g, 'active', range(len(g))))
    packed['edge_ids'] = list(getattr(g, 'edge_ids', range(len(g))))
    return packed


def unpack_graph(data):
    from .sdroute_sdf import _module
    class PinnedGraph(_module('graph').SDGraph):
        def candidates(self, p, radius):
            hits = self.tree.query(Point(float(p[0]), float(p[1])).buffer(radius))
            return [self.active[int(k)] for k in np.atleast_1d(hits)]

    g = PinnedGraph.__new__(PinnedGraph)
    for k, value in data.items():
        if k in ('active', 'edge_ids'):
            continue
        setattr(g, k, [np.asarray(p) for p in value] if k == 'geom' else
                value if k in ('hw', 'name') else np.asarray(value))
    g.active = [int(i) for i in data.get('active', range(len(g.geom)))]
    g.edge_ids = [int(i) for i in data.get('edge_ids', range(len(g.geom)))]
    g.edge_index = {edge_id: i for i, edge_id in enumerate(g.edge_ids)}
    g.out = {}
    for i in g.active:
        g.out.setdefault(int(g.u[i]), []).append(i)
    g.lines = [LineString(p) for p in g.geom]
    g.tree = STRtree([g.lines[i] for i in g.active])
    g._acc = [np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))] for p in g.geom]
    g._dij = {}
    return g


def pack_frame(ego, tracks):
    d = ego.dynamic_car_state
    state = [ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading,
             d.rear_axle_velocity_2d.x, d.rear_axle_velocity_2d.y,
             d.rear_axle_acceleration_2d.x, d.rear_axle_acceleration_2d.y, 0., 0., 0., 0.]
    actors = []
    for obj in tracks.tracked_objects:
        b, v = obj.box, getattr(obj, 'velocity', None)
        actors.append([str(obj.track_token), obj.tracked_object_type.name,
                       b.center.x, b.center.y, b.center.heading, b.length, b.width, b.height,
                       v.x if v is not None else 0., v.y if v is not None else 0.])
    return state, actors


def pack_lane_graph(route_roadblock_dict, map_api):
    """Route lane graph. `lane_follow.py` scores from this alone.

    The lane verdict ("does this lane continue into the next connector") needs **connectivity**,
    not geometry, and the drivable polygons packed by pack_map carry no connectivity. The scorer
    therefore had to open the scenario pkl separately -- scoring was not self-contained in the
    rollout outputs.

    Type strings use **the same vocabulary** as the scenario builder
    (nuplan_utils.extract_map_features). One scorer has to read both sources, so no new names
    are coined here.

    Neighbour lists are not packed. Lane numbers are for reporting and verdicts use lane ids, so
    they are not needed, and left/right ordering has map-specific pitfalls
    (ego_agent._apply_start_lane_shift docstring).
    """
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    lanes = {}
    for rb_id, block in (route_roadblock_dict or {}).items():
        is_roadblock = map_api.get_map_object(str(rb_id), SemanticMapLayer.ROADBLOCK) is not None
        lane_type = 'LANE_SURFACE_STREET' if is_roadblock else 'LANE_SURFACE_UNSTRUCTURE'
        for lane in getattr(block, 'interior_edges', ()):
            polygon = [[round(x, 2), round(y, 2)]
                       for x, y in lane.polygon.exterior.coords]
            polyline = [[round(p.x, 2), round(p.y, 2)]
                        for p in lane.baseline_path.discrete_path]
            lanes[str(lane.id)] = dict(
                rb=str(rb_id), type=lane_type, polygon=polygon, polyline=polyline,
                exit=[str(e.id) for e in getattr(lane, 'outgoing_edges', ())])
    return lanes


def pack_map(drivable, lane_ids, vehicle, route_roadblock_dict=None, map_api=None):
    data = dict(tokens=drivable.tokens, types=[x.name for x in drivable.map_types],
                wkb=[g.wkb_hex for g in drivable._geometries], lane_ids=list(lane_ids),
                vehicle={k: getattr(vehicle, k) for k in (
                    'width', 'front_length', 'rear_length', 'cog_position_from_rear_axle',
                    'wheel_base', 'vehicle_name', 'vehicle_type', 'height')})
    # The lane graph is what P_PLC is judged on. A failure to pack it is only logged here; the run
    # then has no stop line to judge.
    if route_roadblock_dict is not None and map_api is not None:
        try:
            data['lane_graph'] = pack_lane_graph(route_roadblock_dict, map_api)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[lane_graph] could not pack route lane graph: {exc}")
    return data


def apply_goal_rc_bonus(row, reason):
    """Treat a verified SD-route arrival as complete without hiding measured RC.

    Arrival stops the rollout just short of the route end, so an otherwise clean
    finish cannot earn 1.0 from the measured prefix alone. Whether the ego arrived
    is base_env's verdict, carried here as goal_source == 'sd_route'; this function
    does not re-decide it. SDF is only an edge-set verdict, so the ordered-prefix
    and matching-quality guards still stand before the tail is granted.
    """
    if row.get('rc_method') != 'hmm_prefix_1m':
        return row

    def finite(key):
        try:
            value = float(row[key])
            return value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            return None

    measured = finite('RC_measured')
    if measured is None:
        measured = finite('RC')
    row['RC_goal_bonus'] = False
    if measured is None:
        return row
    row['RC_measured'] = measured
    row['RC'] = measured  # make repeated scoring deterministic
    # `goal_source == 'sd_route'` IS the arrival verdict: base_env.done_function only emits it
    # after progress >= SD_GOAL_PROGRESS_RATIO and end_dist <= SD_GOAL_END_DIST_M (base_env.py
    # done_function). Re-testing those two here against literal .99/10. was a second copy of a
    # threshold owned elsewhere -- it could only ever disagree with the runtime, never with the
    # ego. The arrival tolerance stays a single source, and
    # the guards that remain are the ones about MATCHING QUALITY -- which arrival does not imply.
    #
    # There is deliberately NO floor on `measured` such as .99: SD_GOAL_PROGRESS_RATIO is the
    # RUNTIME's progress ratio, and `measured` is the prefix-scan RC. Different quantities, no
    # reason to share a scale.
    #
    # The shortfall a completed arrival carries is SD_GOAL_END_DIST_M / route_length, because
    # done_function stops the ego as soon as it is within 10 m of the goal: the last 10 m are
    # never driven, so they are never credited. That is a fraction of the ROUTE, not a constant:
    # the ceiling is 1 - 10 / route_length, so any route shorter than 10/(1-.99) = 1000 m would sit
    # permanently under a .99 floor.
    #
    # So the gate is the arrival verdict plus the matching-quality guards, and nothing about how
    # far the prefix scan happened to credit. `RC_goal_bonus` still records whether the
    # value actually moved, so a run that measured 1.0 on its own is not reported as bonused.
    if (reason == 'destination_arrival' and row.get('goal_source') == 'sd_route'
            and finite('P_SD') == 1.
            and row.get('rc_status') == 'full_sdf_one'
            and row.get('rc_guard_reason') == 'rollout_end'
            and finite('rc_reference_suspect') == 0.):
        row['RC_goal_bonus'] = measured < 1.
        row['RC'] = 1.
    return row


def apply_termination(row, reason):
    row['term_reason'] = reason
    if reason == 'route_deviation':
        # Departure fails SDF, but P_offroad remains the measured distance ratio.
        row.update(P_SD=0, P_SD_status='termination')
    apply_goal_rc_bonus(row, reason)
    from .ds_formula import route_ds
    row['RouteDS'] = route_ds(row)
    return row


def traffic_fields(states, actor_frames, steps, sim_dt, rc):
    """Traffic Efficiency on every observed simulator step of the GT window, not the PDM grid.

    `rc` is the pinned RC snapshot: the window ends where the GT reference RC is scored
    against ends, so live scoring and replay cut at the same step. Like TLC, this needs a
    pinned input beyond the dense frames, so it is called beside score_dense, not inside it.
    """
    from .traffic_efficiency import score_dense_frames, gt_window_end
    result = score_dense_frames(states, actor_frames, steps, sim_dt, gt_window_end(rc))
    return {'Eff': result.efficiency, 'Eff_coverage': result.coverage,
            'Eff_status': result.reason, 'Eff_movers': result.n_movers,
            'Eff_low_coverage': int(result.low_coverage),
            'Eff_dt_s': float(sim_dt),
            'Eff_frames': result.n_frames}


def score_dense(states, actor_frames, steps, dt, map_data, capture=None):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
    from nuplan.common.actor_state.agent import Agent
    from nuplan.common.actor_state.static_object import StaticObject
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from odyssey.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap, PDMOccupancyMap
    from odyssey.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
    from odyssey.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex, EgoAreaIndex
    states, steps = np.asarray(states, float), np.asarray(steps, int)
    if len(states) < 2:
        return dict(score_method=VERSION, ds_status='insufficient_poses')
    if len(states) != len(actor_frames) or len(states) != len(steps) or not np.all(np.diff(steps) == 1):
        raise ValueError('Dense scoring requires aligned consecutive observed sim steps')
    vp = VehicleParameters(**map_data['vehicle'])
    initial = EgoState.build_from_rear_axle(StateSE2(*states[0, :3]), StateVector2D(*states[0, 3:5]),
        StateVector2D(*states[0, 5:7]), 0., TimePoint(0), vehicle_parameters=vp)
    drivable = PDMDrivableMap(map_data['tokens'], [SemanticMapLayer[t] for t in map_data['types']],
                              [load_wkb(g, hex=True) for g in map_data['wkb']])
    tracks, frames, objects = [], [], []
    for fi, actors in enumerate(actor_frames):
        objs = []
        from nuplan.common.actor_state.oriented_box import OrientedBox
        for ai, (token, kind, x, y, h, ln, wd, ht, vx, vy) in enumerate(actors):
            ty = TrackedObjectType[kind]
            box = OrientedBox(StateSE2(x, y, h), ln, wd, ht)
            meta = SceneObjectMetadata(timestamp_us=int(fi * dt * 1e6), token=token, track_token=token, track_id=ai)
            cls = Agent if kind in ('VEHICLE', 'PEDESTRIAN', 'BICYCLE') else StaticObject
            kw = dict(velocity=StateVector2D(vx, vy)) if cls is Agent else {}
            objs.append(cls(tracked_object_type=ty, oriented_box=box, metadata=meta, **kw))
        tracks.append(DetectionsTracks(TrackedObjects(objs)))
        objects.append({o.track_token: o for o in objs})
        frames.append(PDMOccupancyMap(list(objects[-1]), [o.box.geometry for o in objs]))

    class Observations:
        collided_track_ids = []
        red_light_token = 'red_light'
        def __getitem__(self, idx): return frames[idx]
        def object_at(self, idx, token): return objects[idx][token]

    scorer = PDMScorer(TrajectorySampling(num_poses=len(states)-1, interval_length=dt))
    lanes = dict.fromkeys(map_data['lane_ids'])
    scorer._reset(states[None], initial, Observations(), None, lanes, drivable, None)
    scorer._calculate_ego_area()
    scorer._calculate_no_at_fault_collision()
    scorer._calculate_drivable_area_compliance()
    scorer._calculate_offroad_distance()
    stats = scorer.offroad_stats(0)
    out = dict(score_method=VERSION, ds_status='ok', ds_pose_count=len(states), ds_final_step=int(steps[-1]),
        no_at_fault_collisions=float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0]),
        P_col=scorer.P_col(0),
        collision_count=scorer.collision_count(0),
        P_off=scorer.P_off(0))
    from .comfort import score_states as score_observed_comfort
    # This is a separate 0.1 s diagnostic; keep PDM's 0.5 s `comfort` and
    # `score` untouched so their historical contract remains interpretable.
    out.update(score_observed_comfort(states, steps, dt))
    for src, dst in [('offroad_distance','offroad_distance_m'), ('total_distance','driven_distance_m'),
                     ('offroad_ratio','offroad_ratio'), ('offroad_distance_nondrivable','offroad_distance_nondrivable_m'),
                     ('offroad_distance_offroute','offroad_distance_offroute_m')]:
        out[dst] = stats[src]
    nc = scorer._collision_time_idcs[0]
    nd = np.flatnonzero(scorer._ego_areas[0, :, EgoAreaIndex.NON_DRIVABLE_AREA])
    first = min(nc, nd[0] if len(nd) else np.inf)
    out['first_violation_step'] = int(steps[int(first)]) if np.isfinite(first) else None
    if capture is not None:
        capture['scorer'] = scorer
        capture.update(dense_series(scorer, steps))
    return out


def dense_series(scorer, steps):
    """The per-tick series behind the scalar penalties, as plain arrays.

    The scorer knows, for every tick, which actors the ego was touching and which area flags
    were set -- and then reports only the integrated scalars. An audit of the timeline must not
    re-derive them (reconstructing "at fault" from the types alone once gave 6 contacts where
    the scorer charged 4), so the scorer's own decisions are saved beside the pinned inputs.

    Returned as a flat COO triple rather than a dense (ticks x actors) matrix because contacts
    are sparse -- a few hundred entries against a few hundred thousand cells.

    - contact_step / contact_token: one row per (tick, actor) overlap the scorer saw at all.
    - contact_at_fault: whether that same row was charged as at-fault.
    - ego_area_flags: the EgoAreaIndex bit-plane per tick, so NON_DRIVABLE / ONCOMING /
      MULTIPLE_LANES / INTERSECTION are each recoverable without re-running point-in-polygon.
    """
    steps = np.asarray(steps, int)
    at_fault = set(map(tuple, scorer._at_fault_contacts[0]))
    contacts = sorted({(int(t), str(k)) for k, t in _iter_seen_contacts(scorer)}
                      | {(int(t), str(k)) for t, k in at_fault})
    return dict(
        contact_step=np.array([steps[t] for t, _ in contacts], dtype=np.int64),
        contact_token=np.array([k for _, k in contacts], dtype=object),
        contact_at_fault=np.array([(t, k) in at_fault for t, k in contacts], dtype=bool),
        ego_area_flags=np.asarray(scorer._ego_areas[0], dtype=bool))


def _iter_seen_contacts(scorer):
    """(token, tick) for every overlap the scorer examined, at fault or not.

    `_last_contact` only retains the LAST tick per token, so it cannot be replayed for a
    timeline. The full set comes from re-querying the same occupancy maps the scorer used --
    the same geometry, not a reconstruction of it.
    """
    for time_idx in range(scorer._proposal_sampling.num_poses + 1):
        for token in scorer._observation[time_idx].intersects(scorer._ego_polygons[0, time_idx]):
            if scorer._observation.red_light_token in token or token == 'ego':
                continue
            yield token, time_idx


#: Static obstacle classes. nuPlan is their only population source: the reconstruction has no
#: actor node for them (the scene export bakes vehicle/pedestrian/bicycle only).
STATIC_CLASSES = frozenset({'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT'})

#: Static obstacles (TRAFFIC_CONE, BARRIER, CZONE_SIGN, GENERIC_OBJECT) are not scored. The
#: reconstruction bakes no actor node for these classes (the scene export: vehicle,
#: pedestrian, bicycle only), so every one of them is an obstacle the model cannot see;
#: charging a contact with one would blame the model for a gap in the reconstruction.
#: GENERIC_OBJECT, nuPlan's catch-all, dominates these classes and is mostly boxes under 1 m2.
#:
#: Only the SCORED copy is filtered. driving_inputs_json['actors'] keeps every observed row, so
#: what was measured stays on record and a later rule can still choose otherwise on the same npz.
STATIC_MODE = 'none'


def drop_static_actors(actor_frames):
    """Remove every STATIC_CLASSES actor from the pinned actor frames. -> (frames, info).

    info counts dropped TOKENS per class. Frames without a static actor come back as the same
    object (idempotent).
    """
    dropped = {}
    for actors in actor_frames:
        for row in actors:
            kind = str(row[1])
            if kind in STATIC_CLASSES:
                dropped.setdefault(kind, set()).add(str(row[0]))
    if not dropped:
        return actor_frames, {'mode': STATIC_MODE, 'dropped': 0, 'by_class': {}}
    tokens = {t for toks in dropped.values() for t in toks}
    out = [[row for row in actors if str(row[0]) not in tokens] for actors in actor_frames]
    return out, {'mode': STATIC_MODE, 'dropped': len(tokens),
                 'by_class': {k: len(v) for k, v in sorted(dropped.items())}}


class PinnedInputs(dict):
    """The scoring inputs in the shape of a loaded rollout_trajectory.npz (``z[key]``, ``z.files``).

    Scene-end scoring builds one of these from exactly the values it is about to save, and hands it to
    replay() -- the same function a rescore calls on the saved file. So a live score can only read what
    the archive keeps: an input that is not pinned is invisible to both paths alike.
    """

    @property
    def files(self):
        return list(self)


# --- record-level rules: which signals are scored, whether lane choice is scored --------------------
#
# A run is scored under the rules pinned into its driving_inputs_json['rules'] when it was scored at the
# end of its scene. The benchmark fixes them (OdysseyBenchmark/odyssey_runtime/launch.py: tl_set / plc_rule) and the
# launcher passes them; nothing downstream decides them again from a scene name.

#: Lane-choice rules a run can be scored with (lane_follow.lane_penalty).
PLC_RULES = ('plc',)
#: Columns the traffic-light rule owns. A run scored without the rule carries them empty and
#: tl_rule='events' (the pinned-events metric alone), so an old row's 'timetable' can never sit
#: beside an events-only P_TL. tl_events_* keep the events metric's own values for the record.
TL_RULE_FIELDS = ('tl_rule', 'tl_set', 'tl_set_sha256', 'tl_events_penalty', 'tl_events_violation_count',
                  'tl_timetable_reason', 'tl_timetable_coverage')
#: Columns the lane rule owns. Empty when the rule is off: an absent P_PLC is "rule not applied".
PLC_RULE_FIELDS = ('plc_rule', 'P_PLC', 'plc_fail', 'plc_late', 'plc_reached')


def make_rules(tl_set=None, lane=None):
    """-> the rules dict to pin (empty = scored as before). Loud on an unknown set or lane rule.

    The set's manifest sha256 is pinned beside its name, so a later edit of the set files cannot
    silently change what an old run is scored against (replay refuses the mismatch).
    """
    rules = {}
    if tl_set not in (None, ''):
        from odyssey.manager.tlc_timetable_set import load_set
        from .tl_sets import timetable_dir
        s = load_set(timetable_dir(tl_set))
        rules.update(tl_set=s['name'], tl_set_sha256=s['sha256'])
    if lane not in (None, ''):
        if lane not in PLC_RULES:
            raise ValueError(f'lane rule {lane!r} must be one of {PLC_RULES}')
        rules['lane'] = str(lane)
    return rules


def rules_from_config(global_config):
    """The rules this run is scored with: tl_set / plc_rule (default null = scored as before)."""
    get = global_config.get if hasattr(global_config, 'get') else (lambda *_: None)
    return make_rules(get('tl_set'), get('plc_rule'))


def _tl_set_and_timetable(rules):
    from .tlc_timetable import load_timetable
    from odyssey.manager.tlc_timetable_set import load_set
    from .tl_sets import timetable_dir
    s = load_set(timetable_dir(rules['tl_set']))
    if rules.get('tl_set_sha256') and s['sha256'] != rules['tl_set_sha256']:
        raise ValueError(f"traffic-light set {rules['tl_set']}: manifest sha256 is {s['sha256'][:12]}, "
                         f"the run was scored against {rules['tl_set_sha256'][:12]} -- the set changed")
    return s, load_timetable(s['dir'])


def check_rules_drivable(rules, global_config, scene_id):
    """Raise at scene start if the timetable rule could never score this run.

    A designated signal scene is scored against the set's timetable, which is only fair if the run was
    DRIVEN with it (tlc_timetable_set = the same set). Scene end would refuse such a run anyway; this
    refuses it before the whole rollout is spent. Other scenes are not affected (P_TL 1.0 either way).
    """
    if not rules.get('tl_set'):
        return
    from odyssey.manager.tlc_timetable_set import resolve, scene_code
    s, timetable = _tl_set_and_timetable(rules)
    code = scene_code(scene_id)
    if code not in s['scenes'] or code not in timetable['files']:
        return
    preset = resolve(global_config, scene_id)
    if not preset or preset['name'] != s['name'] or not preset.get('in_set'):
        raise ValueError(f"{code} is a designated signal scene of {s['name']}: it is scored against that "
                         f"timetable, so it must be driven with tlc_timetable_set={s['name']} "
                         f"(got {None if not preset else preset['name']})")


def apply_rules(row, z, data, rules, capture=None):
    """Apply the pinned record-level rules to a scored row, in place. DS is recomputed afterwards
    by apply_termination, once, from the finished row.

    Traffic light (tl_set): a scene outside the set, or in it without a designated signal, is not a
    signal scene and gets P_TL 1.0; a designated scene gets the timetable's P_TL and must have been
    driven with that timetable. The events metric's own values stay in tl_events_*.
    Lane (lane): P_PLC = 0.7^fail x 0.9^late over the lane-choice stop lines the ego reached.
    """
    row.update(dict.fromkeys(TL_RULE_FIELDS), tl_rule='events')
    row.update(dict.fromkeys(PLC_RULE_FIELDS))
    if rules.get('tl_set'):
        from .tlc_metric import ego_polygons_from_states
        from .tlc_timetable import score as score_timetable
        from odyssey.manager.tlc_timetable_set import scene_code
        s, timetable = _tl_set_and_timetable(rules)
        code = scene_code(str(z['scene']))
        if code is None:
            raise ValueError(f"traffic-light set {s['name']}: scene {str(z['scene'])!r} is not a published scene name (odyssey_sceneNNN)")
        row.update(tl_set=s['name'], tl_set_sha256=s['sha256'],
                   tl_events_penalty=row.get('P_TL'),
                   tl_events_violation_count=row.get('tl_violation_count'))
        if code not in s['scenes']:
            row.update(tl_rule='not_signal_scene', P_TL=1.0, tl_violation_count=0)
        elif code not in timetable['files']:
            row.update(tl_rule='no_designated_signal', P_TL=1.0, tl_violation_count=0)
        else:
            if not data.get('tlc'):
                raise ValueError(f'{code}: no pinned TLC snapshot to score against the {s["name"]} timetable')
            ego = ego_polygons_from_states(z['ds_states'], data['map']['vehicle'])
            result, _ = score_timetable(data['tlc'], ego, z['ds_sim_steps'], code, timetable)
            check = result.get('online_check') or {}
            if result.get('patch_applied') != 'online' or not check.get('ok'):
                # Scoring against colours the run did not drive under would judge a different drive.
                raise ValueError(f"{code}: not driven with the {s['name']} timetable "
                                 f"(patch_applied={result.get('patch_applied')}, online_check ok={check.get('ok')})")
            if result.get('unwidened_connectors'):
                # The lateral margin is part of the rule: scoring a designated signal without it is another rule.
                raise ValueError(f"{code}: designated connectors {result['unwidened_connectors']} have no stop line "
                                 f"or exit edge on their pinned baseline, so the {result.get('lateral_margin_m')} m "
                                 f"lateral margin cannot be applied")
            row.update(tl_rule='timetable', P_TL=float(result['p_tl']),
                       tl_violation_count=int(result['charged_count']),
                       tl_timetable_reason=result.get('tlc_filter_reason'),
                       tl_timetable_coverage=(result.get('coverage') or {}).get('coverage'))
    if rules.get('lane'):
        from .lane_follow import lane_penalty, score_pinned
        lane = score_pinned(z)
        row.update(plc_rule=rules['lane'], P_PLC=lane_penalty(lane), plc_fail=lane.n_fail,
                   plc_late=lane.n_late, plc_reached=lane.n_reached)
        if capture is not None:
            capture['lane_result'] = lane
    return row


def replay(z, capture=None, rules=None):
    """Score a run from its pinned inputs. -> the scored row (RouteDS included).

    The one scoring path: scene end calls it on the inputs it is about to save (PinnedInputs), a
    rescore calls it on the saved npz. ``rules`` None = the rules pinned with the run (none for runs
    scored before rules were pinned); a dict overrides them. ``capture`` (a dict) receives the
    pre-lane-change result, the TLC and contact series, and the dense scorer for metric_manager.
    """
    from .sdroute_score import SDRouteMetric
    data = normalize_legacy_inputs(json.loads(str(z['driving_inputs_json'])))
    rules = (data.get('rules') or {}) if rules is None else rules
    dense_capture = capture if capture is not None else {}
    frames, unset_dropped = drop_unset_pose_frames(data['actors'])
    actors, static_info = drop_static_actors(frames)
    result = score_dense(z['ds_states'], actors, z['ds_sim_steps'],
                         data['sim_dt'], data['map'], dense_capture)
    result.update(traffic_fields(z['ds_states'], actors, z['ds_sim_steps'],
                                 data['sim_dt'], data['rc']))
    if 'tlc' in data:
        from .tlc_metric import score_tlc, ego_polygons_from_states
        # The exact same ego polygons and 0.5 s signal samples as live scoring.
        ego_polygons = (dense_capture['scorer']._ego_polygons[0]
                        if 'scorer' in dense_capture else
                        ego_polygons_from_states(z['ds_states'], data['map']['vehicle']))
        tlc_capture = {}
        result.update(score_tlc(data['tlc'], ego_polygons, z['ds_sim_steps'], capture=tlc_capture))
        dense_capture['tlc_series'] = tlc_capture
        result['score_method'] = 'dense'
    rc = SDRouteMetric.from_snapshot(data['rc'], z['rc_sim_steps'], z['rc_ego_xy'], z['rc_ego_heading'])
    result.update(rc.score(data['term_reason'], data['departure_distance_m']))
    result.update(data.get('goal') or {})
    apply_rules(result, z, data, rules, dense_capture)
    apply_termination(result, data['term_reason'])
    result.update(static_mode=static_info['mode'], static_dropped=static_info['dropped'],
                  unset_pose_dropped=unset_dropped,
                  # a scored row carries no failure: a rescore over a row whose scene-end scoring
                  # failed clears it
                  scoring_error=None)
    return result
