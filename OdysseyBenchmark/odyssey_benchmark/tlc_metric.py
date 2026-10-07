"""Pinned NAVSIM-style traffic-light contacts and event penalty.

The signal clock, route connector geometry, renderer flags and GT poses are
snapshotted during rollout. Live scoring and archive replay call score_tlc on
the same snapshot and the same observed ego polygons.

CADENCE. Contacts are tested on every simulation step (0.1 s), not on the 0.5 s
scoring grid: the geometry was always dense (ego footprints come from the dense
executed poses) and only the signal rows were subsampled, which pushed a
violation's first contact up to 0.5 s late and let a brief entry slip between
samples entirely. v1 snapshots stay on the 0.5 s grid and keep the v1 rules, so
an old archive never gets silently re-read under the new ones.

EVENTS. One violation per INTERSECTION, not per lane connector. Whatever the
cadence, and however many times the footprint leaves and re-enters while it is
red, the intersection is entered once and counts once -- which also keeps the
event count, and therefore the 0.7^n penalty, independent of the sampling rate.

Grouping by connector was the obvious reading of that rule and it was wrong.
nuPlan splits one movement across several LANE_CONNECTOR polygons -- a wide or
multi-lane turn is routinely two or three heavily overlapping shapes -- so a
single red-light entry produced one event per shape and 0.7 became 0.49 or
0.343. Such duplicate pairs overlap heavily (polygon IoU above 0.5) with
identical contact steps, while genuinely distinct connectors barely overlap.
The group key is the
connector's parent ROADBLOCK (its intersection), pinned in the snapshot by
pack_tlc rather than re-derived at scoring time: the map is not available to
replay, and a group key that depends on geometry thresholds is a second source
of truth for "is this the same intersection". v1 snapshots keep v1's rules.
"""
from __future__ import annotations

import json

import numpy as np
from shapely.wkb import loads as load_wkb
from .tlc_filter import load_red_renderable_flags


VERSION = 'tl_events_v3'
DENSE_VERSION = 'tl_events_v2'
LEGACY_VERSION = 'tl_events_v1'
LEGACY_SAMPLE_DT = 0.5          # v1 froze the signal on the scoring grid, by construction
CODE_TABLE = ('MISSING', 'UNKNOWN', 'GREEN', 'YELLOW', 'RED')
TURN_MIN_RAD = np.deg2rad(30.0)
BRANCH_LOOKAHEAD_S = 8.0
BRANCH_EXIT_TOLERANCE_M = 7.5


def _encode(rows, table):
    index = {name: code for code, name in enumerate(table)}
    return [[index[name] for name in row] for row in rows]


def _decode_rows(data, key):
    """Signal rows as strings, from either a v2 code matrix or v1 strings."""
    rows = data.get(key)
    table = data.get('signal_code_table')
    if rows is None or table is None:
        return rows
    return [[table[int(code)] for code in row] for row in rows]


def _signal_source_row(converter, connector, sim_step):
    resolver = getattr(converter, 'traffic_light_source_row', None)
    return int(sim_step) if resolver is None else int(resolver(connector, sim_step))


def capture_tlc_signal_snapshot(scene, converter, sim_step):
    """Capture the logical signals actually observed at one rollout step.

    This is the only place where TLC may consult the live simulator/converter. Scoring at
    rollout finalization must consume these immutable rows; it must never reconstruct past
    colours from ``dynamic_map_states``, ``tl_control.json`` or the then-current replay clock.
    ``tl_control.json`` is render-only metadata mapping a logical colour to a representative
    checkpoint frame. It is not, and must not become, a scoring timetable.
    """
    step = int(sim_step)
    records = {
        str(light.get('traffic_light_lane')): light
        for light in scene.get('dynamic_map_states', {}).values()
        if light.get('type') == 'TRAFFIC_LIGHT' and light.get('traffic_light_lane') is not None
    }
    states = {}
    for signal in converter.convert_to_traffic_lights(step):
        connector = str(signal.lane_connector_id)
        name = str(getattr(signal.status, 'name', signal.status)).rsplit('.', 1)[-1]
        if connector in states and states[connector] != name:
            raise ValueError(f'conflicting TLC status for {connector} at step {step}')
        states[connector] = name
    connector_ids = sorted(set(records).union(states))
    rows, inactive, held = {}, {}, {}
    for connector in connector_ids:
        row = _signal_source_row(converter, connector, step)
        source = records.get(connector, {}).get('state', {}).get('traffic_light_state', ())
        rows[connector] = row
        inactive[connector] = row < 0
        held[connector] = len(source) == 0 or row >= len(source)
        states.setdefault(connector, 'MISSING')
    snap = dict(step=step, states=states, source_rows=rows,
                inactive=inactive, held=held)
    if scene.get('tl_signal_patch'):
        # Opt-in signal patch: which of these colours came from the patch (keyed by the source row).
        from odyssey.manager.signal_patch import is_patched
        snap['patched'] = {c: is_patched(scene, c, rows[c]) for c in connector_ids}
    return snap


def pack_tlc(scene, route_lane_dict, dense_steps, sample_dt, scene_frame_stride,
             gt_rear_xy, gt_heading, vehicle, *, render_override=None, apply_gt_filter=False,
             route_roadblock_dict=None, map_location=None,
             apply_nearside_turn_filter=False, signal_source_overrides=None,
             signal_snapshots=None):
    """Freeze the signal, route polygons, renderer flags and GT on the dense step axis.

    ``gt_rear_xy``/``gt_heading`` are expected already aligned to ``dense_steps`` -- one GT pose
    per sampled step -- so nothing downstream has to divide a step by a stride to find one.
    ``scene_frame_stride`` converts a simulation step into a scene frame (cadence.upsample_n):
    the renderer sanity flags are per SCENE frame, and folding that conversion in here is what
    keeps score-side indexing free of the scene's own sampling rate.

    ``route_roadblock_dict`` maps roadblock id -> roadblock, the same dict the scorer's route
    already carries. It is what turns a lane connector into the intersection it belongs to, and
    it is pinned here because scoring must not consult the map (see the module docstring).

    ``signal_source_overrides`` is the renderer's per-step lane-match decision.  Its key remains
    the planned connector whose polygon gates intersection contact; its value is the connector
    whose colour/source row governed ego's actually occupied entry lane.  This intentionally
    changes TLC signal semantics without introducing a wrong-lane-turn penalty.
    """
    steps = [int(step) for step in dense_steps]
    if not steps or float(sample_dt) <= 0 or int(scene_frame_stride) <= 0:
        raise ValueError('invalid TLC sampling cadence')
    if type(apply_gt_filter) is not bool:
        raise ValueError('invalid TLC GT filter')
    if signal_snapshots is None:
        raise ValueError('TLC requires rollout-captured signal snapshots')
    snapshots = {int(row['step']): row for row in signal_snapshots}
    if len(snapshots) != len(signal_snapshots) or set(snapshots) != set(steps):
        raise ValueError('TLC signal snapshots do not exactly match the scored step axis')
    ordered = [snapshots[step] for step in steps]
    statuses = [{str(k): str(v) for k, v in row['states'].items()} for row in ordered]
    all_ids = {connector for row in statuses for connector in row}
    signal_ids = sorted(all_ids)
    lanes = {str(key): value for key, value in route_lane_dict.items()}
    route_ids = sorted(set(lanes).intersection(signal_ids))
    overrides = {
        int(step): {str(target): str(source) for target, source in row.items()}
        for step, row in (signal_source_overrides or {}).items()
    }
    missing_sources = sorted({
        source for row in overrides.values() for source in row.values()
        if source not in set(signal_ids)
    })
    if missing_sources:
        raise ValueError(f'TLC signal source connectors absent from scenario: {missing_sources}')

    def signal_source(step, connector):
        return overrides.get(int(step), {}).get(connector, connector)

    flags = load_red_renderable_flags(scene, render_override)
    polygons = {}
    paths = {}
    maneuvers = {}
    for connector in route_ids:
        lane = lanes[connector]
        polygon = lane.polygon
        if polygon is None or polygon.is_empty or not polygon.is_valid:
            raise ValueError(f'invalid TLC connector polygon: {connector}')
        polygons[connector] = polygon.wkb_hex
        discrete = list(getattr(getattr(lane, 'baseline_path', None), 'discrete_path', ()))
        xy = [[float(point.x), float(point.y)] for point in discrete]
        if len(xy) >= 2:
            paths[connector] = xy
            maneuvers[connector] = _path_maneuver(np.asarray(xy, dtype=float))
    # Connector -> its intersection. One entry per scored connector, or the key is absent and
    # the connector stands alone: a missing group must not silently merge unrelated connectors,
    # and standing alone is exactly v1's behaviour.
    groups = {}
    for block_id, block in (route_roadblock_dict or {}).items():
        for lane in getattr(block, 'interior_edges', ()):
            lane_id = str(lane.id)
            if lane_id in polygons:
                groups[lane_id] = str(block_id)
    gt = None if gt_rear_xy is None else np.asarray(gt_rear_xy, float).tolist()
    heading = None if gt_heading is None else np.asarray(gt_heading, float).tolist()
    if (gt is None) != (heading is None) or (gt is not None and len(gt) != len(heading)):
        raise ValueError('TLC GT positions/headings must be aligned')
    if gt is not None and len(gt) != len(steps):
        raise ValueError('TLC GT must carry one pose per sampled step')
    unknown = sorted({name for row in statuses for name in row.values()} - set(CODE_TABLE))
    if unknown:
        raise ValueError(f'unknown traffic-light status {unknown}')
    all_states = [[row.get(connector, 'MISSING') for connector in signal_ids] for row in statuses]
    source_connector_ids = [
        [signal_source(step, connector) for connector in route_ids]
        for step in steps
    ]
    route_states = [
        [row.get(source, 'MISSING') for source in source_row]
        for row, source_row in zip(statuses, source_connector_ids)
    ]
    source_rows_all = [[int(row['source_rows'][connector]) for connector in signal_ids]
                       for row in ordered]
    signal_inactive_all = [[bool(row['inactive'][connector]) for connector in signal_ids]
                           for row in ordered]
    all_held = [[bool(row['held'][connector]) for connector in signal_ids]
                for row in ordered]
    patched = any('patched' in row for row in ordered)
    # Renderer sanity flags are arrays over SCENE frames. Moving them onto the step axis once here
    # means scoring need not know the scene's sampling period (v1 wrongly did this conversion in
    # score frames).
    aligned_flags = {}
    for connector in route_ids:
        values = flags.get(connector, [])
        aligned_flags[connector] = [
            values[step // int(scene_frame_stride)]
            if step // int(scene_frame_stride) < len(values) else None for step in steps]
    packed = dict(version=VERSION, sample_dt=float(sample_dt),
                scene_frame_stride=int(scene_frame_stride), sim_steps=steps,
                signal_code_table=list(CODE_TABLE),
                signal_connector_ids=signal_ids,
                signal_states_all=_encode(all_states, CODE_TABLE),
                signal_source_rows_all=source_rows_all,
                signal_inactive_all=signal_inactive_all,
                signal_held_all=all_held,
                connector_ids=route_ids, connector_wkb=polygons,
                connector_roadblock=groups,
                connector_paths=paths, connector_maneuver=maneuvers,
                map_location=str(map_location or ''),
                apply_nearside_turn_filter=bool(apply_nearside_turn_filter),
                signal_states=_encode(route_states, CODE_TABLE),
                signal_state_connector_ids=source_connector_ids,
                signal_source_rows=[
                    [int(ordered[i]['source_rows'][source]) for source in sources]
                    for i, sources in enumerate(source_connector_ids)
                ],
                signal_inactive=[
                    [bool(ordered[i]['inactive'][source]) for source in sources]
                    for i, sources in enumerate(source_connector_ids)
                ],
                signal_held=[
                    [bool(ordered[i]['held'][source]) for source in sources]
                    for i, sources in enumerate(source_connector_ids)
                ],
                red_renderable=aligned_flags,
                gt_rear_xy=gt, gt_heading=heading,
                vehicle=vehicle, apply_gt_filter=apply_gt_filter)
    if patched:
        # Opt-in signal patch (tl_signal_patch_path): per cell, did the colour come from the patch.
        packed['signal_patched_all'] = [[bool(row.get('patched', {}).get(connector, False)) for connector in signal_ids]
                                        for row in ordered]
        packed['signal_patched'] = [[bool(ordered[i].get('patched', {}).get(source, False)) for source in sources]
                                    for i, sources in enumerate(source_connector_ids)]
        packed['signal_patch_sha256'] = (scene.get('tl_signal_patch') or {}).get('sha256')
    return packed


def _gt_polygon(xy, heading, vehicle):
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
    ego = EgoState.build_from_rear_axle(
        StateSE2(float(xy[0]), float(xy[1]), float(heading)),
        StateVector2D(0., 0.), StateVector2D(0., 0.), 0., TimePoint(0),
        vehicle_parameters=VehicleParameters(**vehicle))
    return ego.car_footprint.geometry


def ego_polygons_from_states(states, vehicle):
    """Build exact vehicle footprints when a too-short rollout has no dense scorer."""
    states = np.asarray(states, float)
    return [_gt_polygon(row[:2], row[2], vehicle) for row in states]


def score_tlc(data, ego_polygons, dense_steps, capture=None):
    """Score red-light events and optionally capture the exact dense trace.

    ``capture`` is populated from the same contacts used for the scalar score.  It
    exists so rollout visualization never has to intersect a second, sparse ego
    trajectory with the map and call that reconstruction the scored result.
    """
    legacy = data['version'] == LEGACY_VERSION
    if data['version'] not in (VERSION, DENSE_VERSION, LEGACY_VERSION):
        raise ValueError(f'unsupported TLC snapshot {data["version"]}')
    steps = np.asarray(dense_steps, int)
    if len(ego_polygons) != len(steps):
        raise ValueError('TLC ego polygons and dense steps must be aligned')
    by_step = {int(step): i for i, step in enumerate(steps)}
    if len(by_step) != len(steps):
        raise ValueError('TLC dense steps contain duplicates')
    connectors = data['connector_ids']
    polygons = {c: load_wkb(data['connector_wkb'][c], hex=True) for c in connectors}
    frames = data['sim_steps']
    signal_states = _decode_rows(data, 'signal_states')
    if len(frames) != len(signal_states) or len(frames) != len(data['signal_held']):
        raise ValueError('TLC snapshot rows are misaligned')
    sample_dt = LEGACY_SAMPLE_DT if legacy else float(data['sample_dt'])
    raw_contacts = []
    gt_contacts = []
    for row_index, (step, row) in enumerate(zip(frames, signal_states)):
        if step not in by_step or len(row) != len(connectors):
            raise ValueError('TLC signal row is not on the saved ego scoring grid')
        ego_polygon = ego_polygons[by_step[step]]
        # v1 stored GT only on the 0.5 s grid and looked it up by dividing the step by stride. v2
        # stores one GT pose per sampled step -- without the division, a cadence change still
        # sees the same pose.
        gt_idx = int(step) // data['score_stride'] if legacy else row_index
        gt_polygon = None
        if data['gt_rear_xy'] is not None and gt_idx < len(data['gt_rear_xy']):
            gt_polygon = _gt_polygon(data['gt_rear_xy'][gt_idx],
                                     data['gt_heading'][gt_idx], data['vehicle'])
        for connector, state in zip(connectors, row):
            if state != 'RED':
                continue
            polygon = polygons[connector]
            if ego_polygon.intersects(polygon):
                raw_contacts.append((row_index, int(step), connector))
            if gt_polygon is not None and gt_polygon.intersects(polygon):
                gt_contacts.append((row_index, int(step), connector))
    events = _group_events(raw_contacts, legacy, data.get('connector_roadblock'))
    renderer_waived = 0
    waived_contacts = 0
    unknown_contacts = 0
    for event in events:
        # Every connector in the group, so a merged event is waived only when NONE of the
        # polygons it covers could be shown red -- the same "missing never waives" rule, applied
        # to the whole intersection rather than to whichever connector happened to be first.
        members = event.get('connectors', [event['connector']])
        contact_rows = [(f, c) for f, _, c in raw_contacts if c in members
                        and event['first_frame'] <= f <= event['last_frame']]
        # v1 kept flags as scene-frame arrays and indexed them by score frame (correct only by
        # coincidence when the scene is 0.5 s). In v2, pack_tlc has already aligned them to steps.
        values = [_flag_at(data, data['red_renderable'].get(c, []), frames, f, legacy)
                  for f, c in contact_rows]
        waived_contacts += sum(value is False for value in values)
        unknown_contacts += sum(value is None for value in values)
        event['render_waived'] = bool(values) and all(value is False for value in values)
        renderer_waived += int(event['render_waived'])
        branch = _selected_branch(data, event, ego_polygons, sample_dt)
        event.update(branch)
        event['nearside_turn_waived'] = bool(
            not legacy and data.get('apply_nearside_turn_filter', False)
            and branch.get('selected_maneuver') == _nearside_maneuver(data.get('map_location'))
        )
    gt_waived = data['apply_gt_filter'] and bool(gt_contacts)
    turn_waived = sum(bool(event.get('nearside_turn_waived')) for event in events)
    penalized = ([] if gt_waived else
                 [e for e in events if not e['render_waived']
                  and not e.get('nearside_turn_waived', False)])
    count = len(penalized)
    if not events:
        reason = 'no_violation'
    elif gt_waived:
        reason = 'gt'
    elif not penalized:
        if renderer_waived and turn_waived:
            reason = 'rendering_sanity+nearside_turn'
        elif turn_waived:
            reason = 'nearside_turn'
        else:
            reason = 'rendering_sanity'
    else:
        reason = 'unwaived_violation'
    all_states = _decode_rows(data, 'signal_states_all')
    all_held = data.get('signal_held_all', data['signal_held'])
    events_json = json.dumps(events, separators=(',', ':'))
    result = dict(tlc_version=data['version'],
                  tlc_sample_dt=float(sample_dt),
                  traffic_light_compliance=float(not raw_contacts),
                  gt_traffic_light_compliance=(float(not gt_contacts)
                                               if data['gt_rear_xy'] is not None else None),
                  filtered_traffic_light_compliance=float(count == 0),
                  tlc_filter_reason=reason, tlc_red_contact_count=len(raw_contacts),
                  tlc_red_contact_seconds=float(len(raw_contacts) * sample_dt),
                  tlc_render_waived_contact_count=waived_contacts,
                  tlc_render_unknown_contact_count=unknown_contacts,
                  tlc_raw_event_count=len(events), tlc_render_waived_event_count=renderer_waived,
                  tlc_nearside_turn_waived_event_count=turn_waived,
                  tl_violation_count=count,
                  P_TL=float(0.7 ** count),
                  first_tlc_violation_step=raw_contacts[0][1] if raw_contacts else None,
                  first_tlc_penalized_step=penalized[0]['first_step'] if penalized else None,
                  traffic_light_status_frames=(sum(any(state != 'MISSING' for state in row)
                                                  for row in all_states) if all_states is not None else None),
                  red_light_status_frames=(sum('RED' in row for row in all_states)
                                           if all_states is not None else None),
                  relevant_red_light_frames=sum('RED' in row for row in signal_states),
                  traffic_light_status_records=(sum(sum(state != 'MISSING' for state in row)
                                                    for row in all_states)
                                                if all_states is not None else None),
                  tlc_signal_held_frames=sum(any(row) for row in all_held),
                  tlc_events_json=events_json)
    if capture is not None:
        contact_flags = []
        for row_index, _, connector in raw_contacts:
            value = _flag_at(data, data['red_renderable'].get(connector, []),
                             frames, row_index, legacy)
            contact_flags.append(-1 if value is None else int(bool(value)))
        capture.update(
            tlc_trace_version=np.int64(4),
            tlc_scorer_trace_valid=np.bool_(True),
            tlc_trace_sample_dt=np.float64(sample_dt),
            tlc_trace_sim_steps=np.asarray(frames, dtype=np.int64),
            tlc_signal_code_table=np.asarray(data.get('signal_code_table', CODE_TABLE), dtype=str),
            tlc_signal_connector_ids=np.asarray(data['signal_connector_ids'], dtype=str),
            tlc_signal_states=np.asarray(all_states, dtype=str),
            tlc_signal_source_rows=np.asarray(
                data.get('signal_source_rows_all', [
                    [step] * len(data['signal_connector_ids']) for step in frames
                ]), dtype=np.int64
            ),
            tlc_signal_inactive=np.asarray(
                data.get('signal_inactive_all', np.zeros_like(all_held, dtype=bool)),
                dtype=bool,
            ),
            tlc_signal_held=np.asarray(all_held, dtype=bool),
            tlc_route_connector_ids=np.asarray(connectors, dtype=str),
            # Route connector is the polygon used for contact.  Source connector is the signal
            # governing ego's actually occupied entry lane; they intentionally differ after a
            # lane deviation.  Archive both so the score can be audited without map-matching.
            tlc_route_signal_states=np.asarray(signal_states, dtype=str),
            tlc_route_signal_source_connector_ids=np.asarray(
                data.get('signal_state_connector_ids', [connectors] * len(frames)), dtype=str),
            tlc_route_signal_source_rows=np.asarray(data.get(
                'signal_source_rows', [[step] * len(connectors) for step in frames]),
                dtype=np.int64),
            tlc_route_signal_inactive=np.asarray(data.get(
                'signal_inactive', np.zeros((len(frames), len(connectors)), dtype=bool)),
                dtype=bool),
            tlc_route_signal_held=np.asarray(data.get('signal_held'), dtype=bool),
            tlc_contact_sim_steps=np.asarray([step for _, step, _ in raw_contacts], dtype=np.int64),
            tlc_contact_connector_ids=np.asarray([connector for _, _, connector in raw_contacts],
                                                 dtype=str),
            tlc_contact_render_flags=np.asarray(contact_flags, dtype=np.int8),
            tlc_events_json=np.asarray(events_json),
        )
    return result


def _path_maneuver(path):
    """Classify a connector by its entry/exit tangents, not by its bounding box."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        return 'UNKNOWN'
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    if cumulative[-1] < 1e-3:
        return 'UNKNOWN'
    span = min(5.0, cumulative[-1] * .35)
    entry_i = max(1, int(np.searchsorted(cumulative, span)))
    exit_i = min(len(path) - 2, int(np.searchsorted(cumulative, cumulative[-1] - span)) - 1)
    entry = path[entry_i, :2] - path[0, :2]
    exit_vector = path[-1, :2] - path[exit_i, :2]
    if np.linalg.norm(entry) < 1e-6 or np.linalg.norm(exit_vector) < 1e-6:
        return 'UNKNOWN'
    entry_heading = np.arctan2(entry[1], entry[0])
    exit_heading = np.arctan2(exit_vector[1], exit_vector[0])
    delta = (exit_heading - entry_heading + np.pi) % (2 * np.pi) - np.pi
    if delta >= TURN_MIN_RAD:
        return 'LEFT'
    if delta <= -TURN_MIN_RAD:
        return 'RIGHT'
    return 'STRAIGHT'


def _nearside_maneuver(map_location):
    """Near-side turn: left in Singapore's left-hand traffic, right on US maps."""
    location = str(map_location or '').lower()
    if location.startswith('sg-') or 'singapore' in location:
        return 'LEFT'
    if location.startswith('us-'):
        return 'RIGHT'
    return None


def _selected_branch(data, event, ego_polygons, sample_dt):
    """Map-match the executed exit branch after a raw red-polygon contact.

    The entry polygons of several movements can overlap.  We therefore wait for the ego to
    leave the fork and select the connector whose directed baseline it actually completes.
    Failure to observe an exit is deliberately conservative: it grants no waiver.
    """
    paths = data.get('connector_paths') or {}
    maneuvers = data.get('connector_maneuver') or {}
    if not paths or not data.get('apply_nearside_turn_filter', False):
        return {}
    group = event.get('roadblock')
    connector_groups = data.get('connector_roadblock') or {}
    candidates = [connector for connector in data.get('connector_ids', ())
                  if (connector_groups.get(connector) == group if group is not None
                      else connector in event.get('connectors', (event['connector'],)))
                  and connector in paths]
    if not candidates:
        return {}
    first = int(event['first_frame'])
    last = min(len(ego_polygons), int(event['last_frame']) + 1
               + int(np.ceil(BRANCH_LOOKAHEAD_S / float(sample_dt))))
    points = np.asarray([[polygon.centroid.x, polygon.centroid.y]
                         for polygon in ego_polygons[first:last]], dtype=float)
    if len(points) < 2:
        return {}
    matches = []
    for connector in candidates:
        path = np.asarray(paths[connector], dtype=float)[:, :2]
        if len(path) < 2:
            continue
        cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        if cumulative[-1] < 1e-3:
            continue
        distances = np.linalg.norm(points[:, None, :] - path[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        progress = cumulative[nearest] / cumulative[-1]
        exit_distance = float(np.min(np.linalg.norm(points - path[-1], axis=1)))
        progress_gain = float(np.max(progress) - progress[0])
        if (exit_distance > BRANCH_EXIT_TOLERANCE_M or np.max(progress) < .8
                or progress_gain < .2):
            continue
        path_error = float(np.mean(np.min(distances, axis=1)))
        matches.append((exit_distance, path_error, connector))
    if not matches:
        return {}
    exit_distance, path_error, connector = min(matches)
    return dict(selected_connector=connector,
                selected_maneuver=maneuvers.get(connector, 'UNKNOWN'),
                selected_exit_distance_m=round(exit_distance, 3),
                selected_path_error_m=round(path_error, 3))


def _group_events(raw_contacts, legacy, groups=None):
    """One event per INTERSECTION. v1 snapshots keep v1's per-connector frame-adjacency rule.

    An intersection is entered once. Splitting that entry whenever the footprint momentarily
    clears the polygon made the count -- and with it the 0.7^n penalty -- a function of the
    sampling rate: at 0.1 s a single jittering edge became several events. Splitting it across
    the several overlapping LANE_CONNECTOR polygons nuPlan uses for one movement made the count
    a function of how finely the map happened to cut that turn, which is the same failure one
    step up. ``groups`` maps a connector to its roadblock; a connector with no entry is its own
    group, so an unpinned snapshot degrades to the old per-connector count rather than merging
    connectors it knows nothing about.

    ``connector`` on the event is the FIRST connector contacted in that group, and
    ``connectors`` lists all of them, so an audit can still see every polygon involved.
    """
    events = []
    if legacy:
        last = {}
        for frame, step, connector in raw_contacts:
            event = last.get(connector)
            if event is None or event['last_frame'] != frame - 1:
                event = dict(connector=connector, first_frame=frame, last_frame=frame,
                             first_step=step, last_step=step)
                events.append(event)
            else:
                event['last_frame'], event['last_step'] = frame, step
            last[connector] = event
        return events
    groups = groups or {}
    by_group = {}
    for frame, step, connector in raw_contacts:
        key = groups.get(connector, connector)
        event = by_group.get(key)
        if event is None:
            event = dict(connector=connector, connectors=[connector], roadblock=groups.get(connector),
                         first_frame=frame, last_frame=frame,
                         first_step=step, last_step=step, contacts=0)
            by_group[key] = event
            events.append(event)
        elif connector not in event['connectors']:
            event['connectors'].append(connector)
        event['last_frame'], event['last_step'] = frame, step
        event['contacts'] += 1
    return events


def _flag_at(data, flags, frames, row_index, legacy):
    index = frames[row_index] // data['score_stride'] if legacy else row_index
    return flags[index] if 0 <= index < len(flags) else None
