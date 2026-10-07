"""Post-rollout SDF and first-zero 1m prefix RC on the same executed window.

The route and the executed poses come from the simulator's tracker
(odyssey.manager.sdroute_metric.SDRouteTracker), which also decides arrival and departure while
the run drives. SDRouteMetric adds the SD-graph edges and the scoring.
"""
import numpy as np

from odyssey.manager.sdroute_metric import SDRouteTracker
from odyssey.manager.sdroute_reference import build_coverage_reference
from odyssey.utils.cadence import SD_GOAL_END_DIST_M
from .sdroute_prefix import build_edge_reference
from .sdroute_prefix_scan import MatcherCache, scan_rc
from .sdroute_sdf import METHOD, VERSION, graph, matcher


#: The arrival radius: the same value base_env ends a rollout with (defines.SD_GOAL_END_DIST_M).
ARRIVAL_ZONE_M = SD_GOAL_END_DIST_M


def arrival_zone_keep(xy, end, radius=ARRIVAL_ZONE_M, min_keep_m=5.0):
    """Number of leading poses to keep for SDF/RC matching in a run that ended by arrival.

    The arrival check treats the route as finished once the ego is within `radius` of the end of
    the scored segment. Zeroing SDF again because of which branch the ego took inside that radius
    would give one run both "arrived" and "departed". This happens when a fork lies just before
    the end point: a run that arrives a few metres past it is close to both branches, and the
    matcher's heading cost can attach it to the wrong one -- the difference is when the wheel was
    turned, not the position.

    So, walking back from the end of the trajectory, keep poses up to the first pose of the
    contiguous stretch that stayed within `radius` of the end point, and drop the rest. Only the
    contiguous stretch counts, so that an early pass is not cut in scenes whose route first brushes
    past the end point. If the kept length is shorter than `min_keep_m`, the matcher has
    nothing to judge, so nothing is trimmed. Nothing is trimmed if the last pose is outside the
    radius.
    """
    P = np.asarray(xy, float).reshape(-1, 2)
    if len(P) < 2:
        return len(P)
    d = np.linalg.norm(P - np.asarray(end, float).reshape(2), axis=1)
    if d[-1] > radius:
        return len(P)
    k = len(P) - 1
    while k > 0 and d[k - 1] <= radius:
        k -= 1
    keep = k + 1
    if keep >= len(P):
        return len(P)
    kept = float(np.linalg.norm(np.diff(P[:keep], axis=0), axis=1).sum())
    return keep if kept >= min_keep_m else len(P)


class SDRouteMetric(SDRouteTracker):
    """SDRouteTracker plus the SD-graph edges of its sidecar and the RC/SDF scoring."""
    def __init__(self, scene, track, handoff_step, stride, rac, map_api=None):
        self.topology = None
        self.result = None
        self.edges = []
        self._manual_graph = False
        self.map_location = None
        super().__init__(scene, track, handoff_step, stride, rac, map_api)
        if self.status not in ('ok', 'reference_suspect'):
            return          # the tracker stopped before the reference was built: no edges to read
        # RC no longer queries HD polygons or bridges. Keep map_api in the
        # constructor for callers using the former adapter signature.
        with np.load(self.sidecar, allow_pickle=False) as z:
            if 'sd_edges' in z.files:
                self.edges = [int(e) for e in z['sd_edges']]
            if 'map_location' in z.files:
                self.map_location = str(z['map_location'])
        self._manual_graph = any(e < 0 for e in self.edges)
        if self._manual_graph and self.map_location:
            indexed = graph(self.map_location, include_manual=True).edge_index
            self.edges = [indexed[e] for e in self.edges]

    def score(self, term_reason=None, departure_distance_m=30.):
        steps, xy, heading, timestamps = self.arrays()
        departed = term_reason == 'route_deviation'
        # Only runs that ended by arrival drop the trajectory inside the arrival radius
        # (arrival_zone_keep). Runs that did not arrive are not trimmed: the distance they covered
        # toward the end must still count toward RC.
        trim_m = 0.
        if (term_reason == 'destination_arrival' and self.reference is not None
                and not self.reference.quality['suspect']):
            keep = arrival_zone_keep(xy, self.reference.crop_xy()[-1])
            if keep < len(xy):
                trim_m = float(np.linalg.norm(np.diff(xy[keep - 1:], axis=0), axis=1).sum())
                steps, xy, heading, timestamps = (
                    steps[:keep], xy[:keep], heading[:keep], timestamps[:keep])
        row = dict(RC=float('nan'), rc_method=METHOD, rc_status=self.status,
                   rc_sidecar=self.sidecar or '', rc_pose_count=len(xy),
                   rc_final_step=int(steps[-1]) if len(steps) else -1,
                   rc_arrival_trim_m=trim_m,
                   rc_topology_status='not_used', rc_prefix_spacing_m=1.,
                   rc_prefix_checks=0, sdf_version=VERSION,
                   P_SD=0 if departed else float('nan'),
                   P_SD_status='termination' if departed else 'unavailable')
        if self.reference is None:
            return row
        if not len(xy) or steps[0] != self.handoff:
            row['rc_status'] = 'missing_handoff_pose'
            return row
        row.update(rc_reference_m=self.reference.length,
                   rc_reference_suspect=bool(self.reference.quality['suspect']))
        if not self.edges or not self.map_location:
            row['rc_status'] = 'missing_sdf_reference'
            return row
        if len(xy) < 2 or len(matcher().resample(xy)) < 2:
            # No driven distance is known zero RC, not an invented SDF pass.
            # Not only a full stop: every drive the matcher cannot resolve lands here. The matcher
            # resamples at STEP (5 m) spacing, so anything shorter collapses to one point and
            # match() returns None per its "query too short" contract -- not a match failure, but
            # nothing to ask in the first place. The result is already determined: none of the
            # route was covered, so RC and SDF are both 0.
            # Runs that ended by departure are already 0/termination; do not overwrite that label.
            # Measured: 4.99 m -> 1 resampled point, 5.00 m -> 2. Some runs moved only 2.8-3.3 m
            # in 200 s.
            row.update(RC=0., rc_covered_m=0., rc_status='no_motion')
            if not departed:
                row.update(P_SD=0., P_SD_status='no_motion')
            return row
        try:
            g, m = getattr(self, '_pinned_graph', None), matcher()
            if g is None:
                g = (graph(self.map_location, include_manual=True) if self._manual_graph
                     else graph(self.map_location))
            if any(e < 0 or e >= len(g) for e in self.edges):
                row['rc_status'] = 'unsupported_reference_edge'
                return row
            cache = MatcherCache(g, self.edges, m)
            value, _, _, _ = cache.verdict(xy)
            if not departed:
                row.update(P_SD=value if value is not None else float('nan'),
                           P_SD_status='ok' if value is not None else 'match_failed')
            er = build_edge_reference(g, self.edges, self.reference.route.xy)
            self.result = scan_rc(g, self.edges, er, self.reference, xy, timestamps, m,
                                  departed=departed, cache=cache,
                                  departure_distance_m=departure_distance_m)
        except (ValueError, KeyError, OSError, ImportError) as exc:
            # Preserve the rollout and PDM metrics, but never fall back to the
            # old RC or a fabricated SDF=1. Launchers can inspect these statuses.
            row.update(rc_status='scoring_error', rc_error=f'{type(exc).__name__}: {exc}')
            return row
        result = self.result
        row.update(RC=result['rc'] if result['rc'] is not None else float('nan'),
                   rc_status=result['status'], rc_covered_m=result.get('credited_m'),
                   rc_prefix_checks=result['prefix_checks'],
                   rc_cutoff_model_m=result['cutoff_model_m'],
                   rc_cutoff_time_s=result['cutoff_time_s'],
                   rc_first_zero_model_m=result['first_zero_model_m'],
                   rc_first_zero_time_s=result['first_zero_time_s'],
                   rc_first_zero_edge=(getattr(g, 'edge_ids', range(len(g)))[result['first_zero_edge']]
                                       if result['first_zero_edge'] is not None else None),
                   rc_guard_reason=result['rc_guard_reason'])
        return row

    def snapshot(self):
        """Pin the actual reference, not a path to a mutable sidecar/scenario."""
        from .driving_metrics import pack_graph
        return dict(handoff=self.handoff, stride=self.stride, sim_dt=self.sim_dt,
                    offset=self.offset.tolist(), route=None if self.route_xy is None else self.route_xy.tolist(),
                    gt=None if self.reference_gt is None else self.reference_gt.tolist(),
                    edges=self.edges, manual_graph=self._manual_graph,
                    map_location=self.map_location, sidecar=self.sidecar,
                    status=self.status,
                    graph=pack_graph(graph(self.map_location, include_manual=True)
                                     if self._manual_graph else graph(self.map_location))
                    if self.edges and self.map_location else None)

    @classmethod
    def from_snapshot(cls, data, steps, xy, heading):
        from .driving_metrics import unpack_graph
        obj = cls.__new__(cls)
        obj.handoff, obj.stride, obj.sim_dt = data['handoff'], data['stride'], data['sim_dt']
        obj.offset = np.asarray(data['offset'])
        obj.route_xy, obj.reference_gt = data['route'], data['gt']
        obj.edges, obj.map_location, obj.sidecar = data['edges'], data['map_location'], data['sidecar']
        obj._manual_graph = bool(data.get('manual_graph', False))
        obj.status, obj.reference, obj.topology, obj.result = data['status'], None, None, None
        if data['route'] is not None and data['gt'] is not None and data['status'] in ('ok', 'reference_suspect'):
            obj.reference = build_coverage_reference(data['route'], data['gt'], gt_start_index=obj.handoff // obj.stride)
        obj._pinned_graph = unpack_graph(data['graph']) if data['graph'] is not None else None
        # Saved XY already includes the configured origin offset.
        obj.samples = {int(s): (np.asarray(p), float(h)) for s, p, h in zip(steps, xy, heading)}
        obj.last = None
        return obj
