"""Resolving a scene's nuPlan map location.

Two scene-prep conventions disagree about what ``scene["map"]`` holds:

  MTGS ``vis_tokens_*``  ->  the nuPlan CITY, e.g. "us-nv-las-vegas-strip"
  OmniRe scenes          ->  the source LOG TOKEN, e.g.
                             "2021.09.29.19.02.14_veh-28_02451_02708",
                             with the city in ``metadata["nuplan_map_location"]``

``get_maps_api`` accepts either without complaint -- it builds the api lazily -- so a log
token only fails much later, and unrecognisably. The first roadblock lookup reaches
``gpkg_mapsdb.get_version()``, whose ``self._metadata[location]["version"]`` raises KeyError
on the token; ``nuplan_map.get_map_object()`` catches that and re-raises as

    ValueError: Object representation for layer: ROADBLOCK object: 49153 is unavailable

which reads as "that roadblock id is missing from the map" when in fact no map was ever
opened and the id was fine.

So resolve the city in ONE place, and fail at resolution time naming both candidates.
"""
import os

__all__ = ["resolve_map_location", "route_map_radius", "route_roadblock_ids",
           "DEFAULT_MAP_RADIUS", "MAP_RADIUS_MARGIN_M"]


def resolve_map_location(scene, nuplan_map_root):
    """Return the nuPlan city for ``scene``, or raise naming what was tried.

    ``metadata["nuplan_map_location"]`` wins when present (it is always the city); otherwise
    ``scene["map"]``, which is the city under the older convention.
    """
    metadata_location = (scene.get("metadata") or {}).get("nuplan_map_location")
    location = metadata_location or scene.get("map")

    try:
        available = sorted(d for d in os.listdir(nuplan_map_root)
                           if os.path.isdir(os.path.join(nuplan_map_root, d)))
    except OSError as exc:
        raise ValueError("nuplan map root %r is not readable: %s" % (nuplan_map_root, exc))

    if location not in available:
        raise ValueError(
            "nuplan map location %r not found under %s (available: %s). "
            "scene['map']=%r, metadata['nuplan_map_location']=%r"
            % (location, nuplan_map_root, available,
               scene.get("map"), metadata_location)
        )
    return location


#: A successor centroid can sit slightly behind the final pose while the block continues forward.
_HEADROOM_BEHIND_TOLERANCE_M = 5.0

#: Below this span the final-pose difference is dominated by stationary jitter.
_HEADROOM_MIN_HEADING_SPAN_M = 1.0

#: Fallback when a scene carries no usable GT ego track -- the value the PDM map radius was
#: hardcoded to before it became route-derived.
DEFAULT_MAP_RADIUS = 100.0
#: Head-room past the GT extent, for the closed-loop ego drifting off the GT path.
MAP_RADIUS_MARGIN_M = 100.0


def _gt_ego_xy(scene):
    """Ego GT track as (N, 2), or None. Two scene-prep conventions -- see resolve_map_location."""
    import numpy as np

    ot = scene.get("object_track") or {}
    try:
        if "ego" in ot and isinstance(ot["ego"], dict):
            return np.asarray(ot["ego"]["state"]["position"], dtype=np.float64)[:, :2]
        if "position" in ot:
            return np.asarray(ot["position"], dtype=np.float64)[:, :2]
    except Exception:
        pass
    return None


def route_map_radius(scene) -> float:
    """Radius the PDM drivable-area / observation queries need to cover this scene's whole drive.

    The drivable-area map is built ONCE, around the ego state at the handoff frame, and every
    later step is scored against that one snapshot. A fixed radius therefore has to cover the
    WHOLE rollout, not a neighbourhood: any pose past it has no lane polygon under it and is
    treated as off-road.

    A fixed 100 m is enough for 20 s rollouts that cover a few tens of metres, but not for the
    80 s OmniRe scenes or longer tokens, whose GT extents can exceed 100 m and would be cut off.

    So derive it from the GT track the scene ships. The margin covers the closed-loop ego leaving
    the GT path; the floor keeps short scenes at the 100 m default.

    Lives here rather than in metric_manager so the PDM *policy* can size its planner the same
    way without a policy -> manager import.
    """
    import numpy as np

    xy = _gt_ego_xy(scene)
    if xy is None or len(xy) < 2:
        return DEFAULT_MAP_RADIUS
    extent = float(np.linalg.norm(xy - xy[0], axis=1).max())
    return max(DEFAULT_MAP_RADIUS, extent + MAP_RADIUS_MARGIN_M)


def route_roadblock_ids(object_track_positions, initial_ego_center, map_api, logger=None):
    """Roadblocks the logged ego drive runs through, in order, as nuPlan map ids.

    Queried against the nuPlan map api in the GLOBAL frame, NOT read off
    ``navigation.checkpoint_lanes``. Reading the navigation route looks correct -- those lanes
    carry real nuPlan roadblock ids, so every consumer resolves them without complaint -- but
    it mixes the simulator's two coordinate frames:

      * navigation builds its checkpoints with get_closest_lane_index() over object_track
        positions, i.e. against the SCENE-LOCAL ScenarioMap built from ``map_features``;
      * everything downstream (the EgoState from OdysseyToNuPlanConverter, the scorer's drivable
        polygons) lives in GLOBAL UTM.

    Where the two maps are aligned this is invisible. Where they are not, the wrong lanes are
    picked and nothing raises, because the ids are real. Two consumers are affected:

      PDM planner   _get_starting_lane() intersected a global ego pose against a scene-local
                    route -- 587 km apart -- fell through to its nearest-lane fallback and
                    built the centerline from an arbitrary lane, so the car barely moved
                    while collision / drivable-area / direction all read a perfect 1.0.
      PDM scorer    the route roadblocks become the scorer's ON-ROUTE drivable polygon set.
                    EgoAreaIndex.ONCOMING_TRAFFIC is not about direction at all -- it is set
                    when the ego centre is in no on-route polygon (pdm_scorer
                    _calculate_ego_area) -- and it feeds the Driving Score's off-road
                    distance. So a wrong route marks a car that drove perfectly inside its
                    lane as off-route. On scenes whose map_features origin is offset from
                    their object_track origin, the off-route distance grows with that offset
                    even when the ego never touches non-drivable area.

    Sampling the ego's own logged path -- rather than trusting a lane graph built in the other
    frame -- also lets us map-match continuously.  Intersections often contain several
    overlapping lane connectors. Picking ``lanes[0]`` independently at each pose made the route
    jump between unrelated connectors and even revisit the same intersection roadblock several
    times. PDM then either built a looping centerline or treated an adjacent movement's red light
    as its own. Keep the current lane while it still contains the ego, prefer its graph successor
    at transitions, and use path-heading agreement only as the fallback/tie-breaker.

    :param object_track_positions: the ego's logged track, SCENE-LOCAL, (N, >=2).
    :param initial_ego_center: the converter's first-frame ego centre; local + this = global.
    :param map_api: nuPlan map api, i.e. the GLOBAL-frame map.
    :param logger: optional; warned when the walk finds no lane under any pose.
    """
    import numpy as np
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    positions = np.asarray(object_track_positions, dtype=np.float64)[:, :2]
    center = np.asarray(initial_ego_center, dtype=np.float64).reshape(-1)[:2]

    if len(positions) == 0:
        if logger is not None:
            logger.warning("route is empty: the logged ego track has no poses")
        return []

    def _objects_at(point, layer):
        """map_api lookup that tolerates the layer being absent from this map."""
        try:
            return map_api.get_all_map_objects(point, layer)
        except (KeyError, ValueError):
            return []

    def _roadblock_successors(roadblock_id):
        """Successor roadblock IDs, or None when this map cannot resolve the object."""
        roadblock = None
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            try:
                roadblock = map_api.get_map_object(str(roadblock_id), layer)
            except (AttributeError, KeyError, ValueError):
                roadblock = None
            if roadblock is not None:
                break
        if roadblock is None:
            return None
        return {str(edge.id) for edge in getattr(roadblock, "outgoing_edges", [])}

    def _roadblock_path(start_id, target_id, max_depth=8):
        """Return a short directed roadblock path, including both endpoints.

        Logged poses do not necessarily cover every tiny connector polygon.  Treating the next
        observed roadblock as invalid unless it is an *immediate* successor permanently froze the
        matcher at the first such map gap.  A bounded BFS preserves graph continuity while filling
        only the unobserved intermediate roadblocks; it cannot jump to a merely nearby road.
        """
        start_id, target_id = str(start_id), str(target_id)
        if start_id == target_id:
            return [start_id]
        queue = [(start_id, [start_id])]
        visited = {start_id}
        while queue:
            current_id, path = queue.pop(0)
            if len(path) - 1 >= max_depth:
                continue
            successors = _roadblock_successors(current_id)
            if successors is None:
                continue
            for successor_id in sorted(successors):
                if successor_id == target_id:
                    return path + [successor_id]
                if successor_id not in visited:
                    visited.add(successor_id)
                    queue.append((successor_id, path + [successor_id]))
        return None

    # Centered path tangents disambiguate crossing connectors. Fill stationary samples from the
    # nearest moving segment so a stop line does not erase the direction just where it matters.
    global_positions = positions + center
    tangents = np.empty_like(global_positions)
    if len(global_positions) == 1:
        tangents[0] = (1.0, 0.0)
    else:
        tangents[0] = global_positions[1] - global_positions[0]
        tangents[-1] = global_positions[-1] - global_positions[-2]
        if len(global_positions) > 2:
            tangents[1:-1] = global_positions[2:] - global_positions[:-2]
        moving = np.linalg.norm(tangents, axis=1) > 1e-6
        if moving.any():
            moving_idcs = np.flatnonzero(moving)
            for idx in np.flatnonzero(~moving):
                tangents[idx] = tangents[moving_idcs[np.argmin(abs(moving_idcs - idx))]]
        else:
            tangents[:] = (1.0, 0.0)
    path_headings = np.arctan2(tangents[:, 1], tangents[:, 0])

    ids = []
    previous_lane = None
    last_accepted_xy = None
    for global_xy, path_heading in zip(global_positions, path_headings):
        # The same local -> global shift the converter applies to the ego state, so the query
        # and the pose it is meant to describe live in one frame.
        gx, gy = global_xy
        point = Point2D(float(gx), float(gy))
        # get_all_map_objects, not get_one_map_object: the latter ASSERTS when a point sits in
        # more than one lane (overlapping connectors at an intersection), which would abort at
        # exactly the places a route matters most.
        lanes = (_objects_at(point, SemanticMapLayer.LANE)
                 + _objects_at(point, SemanticMapLayer.LANE_CONNECTOR))

        def _lane_errors(lane):
            nearest = lane.baseline_path.get_nearest_pose_from_position(point)
            heading_error = abs(
                (float(nearest.heading) - path_heading + np.pi) % (2 * np.pi) - np.pi
            )
            centerline_distance = np.hypot(float(nearest.x) - gx, float(nearest.y) - gy)
            return heading_error, centerline_distance

        # Polygon containment alone is unreliable at map seams.  In Boston scene 63 the ego
        # starts 2.2 m from its forward lane rail but just inside an opposite-direction lane's
        # broad polygon.  The only containing candidate was therefore 175 degrees backwards,
        # and that single bad first roadblock made the matcher reject the remaining 99 seconds.
        # If containment offers no directionally plausible lane, admit only nearby rails that
        # satisfy the same 3 m / 45 degree handoff gate used by the simulator.
        containing_has_aligned_lane = any(
            _lane_errors(lane)[0] <= np.pi / 4 for lane in lanes
        )
        if not containing_has_aligned_lane:
            try:
                nearby = map_api.get_proximal_map_objects(
                    point, 3.0,
                    [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
                )
            except (AttributeError, KeyError, ValueError):
                nearby = {}
            for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
                for lane in nearby.get(layer, []):
                    heading_error, centerline_distance = _lane_errors(lane)
                    if heading_error <= np.pi / 4 and centerline_distance <= 3.0:
                        lanes.append(lane)

        if not lanes:
            continue

        # Some map releases expose the same object through both layers; keep selection stable by
        # ID before applying graph continuity.
        lanes_by_id = {str(lane.id): lane for lane in lanes}
        def _alignment_score(lane):
            heading_error, centerline_distance = _lane_errors(lane)
            # Heading dominates; distance only breaks same-direction parallel candidates.
            return heading_error + 0.05 * centerline_distance

        # Do not unconditionally keep ``previous_lane`` merely because its polygon still
        # overlaps the point.  Connector polygons overlap for several metres after a fork.  In
        # that region the logged heading is the evidence that tells us which branch was taken;
        # pinning the old connector hid the correct sibling until it was too late to reconnect.
        ranked_lanes = sorted(lanes_by_id.values(), key=_alignment_score)

        # A fallback selection can briefly land on a geometrically overlapping but disconnected
        # connector.  Neither the route nor the continuity seed may follow that connector: once
        # ``previous_lane`` was replaced by such a decoy, its own successors kept winning and
        # the matcher never returned to the logged movement.  That produced routes of only 2--6
        # roadblocks for 100 s tracks and PDM stopped hundreds of metres before the GT endpoint.
        # Advance the continuity seed only after the selected lane is connected to the accepted
        # route (or remains inside its current roadblock).
        for selected_lane in ranked_lanes:
            roadblock_id = selected_lane.get_roadblock_id()
            if not roadblock_id:
                continue
            roadblock_id = str(roadblock_id)
            accept_lane = False
            if not ids:
                ids.append(roadblock_id)
                accept_lane = True
            elif ids[-1] == roadblock_id:
                accept_lane = True
            else:
                lane_connected = previous_lane is None or str(selected_lane.id) in {
                    str(edge.id) for edge in getattr(previous_lane, "outgoing_edges", [])
                }
                successor_roadblocks = _roadblock_successors(ids[-1])
                route_connected = (
                    lane_connected if successor_roadblocks is None
                    else roadblock_id in successor_roadblocks
                )
                if route_connected:
                    ids.append(roadblock_id)
                    accept_lane = True
                elif len(ids) >= 2:
                    # One-pose look-ahead for overlapping intersection branches.  The first few
                    # samples can favour the wrong sibling by centimetres; once the headings
                    # separate, replace that last choice if the new candidate is another direct
                    # successor of the previously confirmed roadblock.  This is backtracking,
                    # not a graph jump: every adjacent pair in the returned route stays connected.
                    predecessor_successors = _roadblock_successors(ids[-2])
                    if (predecessor_successors is not None
                            and roadblock_id in predecessor_successors):
                        ids[-1] = roadblock_id
                        accept_lane = True
                if not accept_lane:
                    # The log can cross a very short connector between two sampled/valid map
                    # poses.  Keep the route directed and deterministic, but bridge that map gap
                    # instead of rejecting every remaining pose in the scene.
                    bridge = _roadblock_path(ids[-1], roadblock_id)
                    if bridge is not None:
                        ids.extend(bridge[1:])
                        accept_lane = True
                if not accept_lane:
                    # Some nuPlan map seams have no directed graph edge despite a continuous,
                    # well-aligned logged drive (scene 83 crosses several such seams).  After a
                    # real spatial gap, retain the new GT-supported roadblock as route membership
                    # rather than truncating every later signal/on-route polygon.  The 5 m
                    # hysteresis prevents a momentary overlapping connector from becoming a
                    # reset; PDM's long-window centerline uses the continuous GT geometry.
                    heading_error, centerline_distance = _lane_errors(selected_lane)
                    travelled_since_accept = (
                        np.inf if last_accepted_xy is None
                        else float(np.linalg.norm(global_xy - last_accepted_xy))
                    )
                    if (
                        heading_error <= np.pi / 4
                        and centerline_distance <= 3.0
                        and travelled_since_accept >= 5.0
                    ):
                        ids.append(roadblock_id)
                        accept_lane = True
            if accept_lane:
                previous_lane = selected_lane
                last_accepted_xy = np.asarray(global_xy, dtype=np.float64).copy()
                break

    # The arrival test may end one simulator tick after the GT endpoint. Give the scorer one
    # roadblock of forward head-room so that this legal stopping distance is not charged as
    # oncoming/off-route. This is deliberately one graph level, direction-filtered, and is not
    # part of the GT matcher above.
    if ids and len(global_positions) >= 2:
        end_xy = global_positions[-1]
        end_dir = np.zeros(2, dtype=np.float64)
        span = 0.0
        for back in range(1, len(global_positions)):
            end_dir = end_xy - global_positions[-1 - back]
            span = float(np.linalg.norm(end_dir))
            if span >= _HEADROOM_MIN_HEADING_SPAN_M:
                break
        if span >= _HEADROOM_MIN_HEADING_SPAN_M:
            end_dir /= span
            for successor_id in sorted(_roadblock_successors(ids[-1]) or ()):
                if successor_id in ids:
                    continue
                block = None
                for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
                    try:
                        block = map_api.get_map_object(str(successor_id), layer)
                    except (AttributeError, KeyError, ValueError):
                        block = None
                    if block is not None:
                        break
                if block is None:
                    continue
                centroid = np.asarray(block.polygon.centroid.coords[0], dtype=np.float64)
                if float((centroid - end_xy) @ end_dir) < -_HEADROOM_BEHIND_TOLERANCE_M:
                    continue
                headings = []
                for lane in block.interior_edges:
                    try:
                        path = lane.baseline_path.discrete_path
                        vector = np.array(
                            [path[-1].x - path[0].x, path[-1].y - path[0].y],
                            dtype=np.float64,
                        )
                        length = float(np.linalg.norm(vector))
                        if length > 1e-6:
                            headings.append(float((vector / length) @ end_dir))
                    except (AttributeError, IndexError, ValueError):
                        continue
                if headings and max(headings) <= 0.0:
                    continue
                ids.append(successor_id)

    if not ids and logger is not None:
        logger.warning(
            "route is empty: no map lane under any of the %d logged ego poses. Check that the "
            "scene's map location and initial_ego_center agree with the nuPlan map.",
            len(positions))
    return ids
