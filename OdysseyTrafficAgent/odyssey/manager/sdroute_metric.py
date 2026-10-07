"""The SD route a run is held to while it drives.

The route file (load_scene_route) and the tracker built on it (SDRouteTracker: the GT-cropped
reference and the executed poses) are what base_env ends a rollout on -- departure from the
route and arrival at its end. Scoring the same route (SDF and the 1 m prefix RC) is the
benchmark's: its SDRouteMetric subclasses SDRouteTracker and reaches MetricManager through the
configured scorer.
"""
import os

import numpy as np

from .sdroute_reference import build_coverage_reference


def load_scene_route():
    # ODYSSEY_ROUTE_FILE is the route.npz of the scene's own release folder (the launcher checks
    # the folder against scenes.csv). The file's `tokens` are the frames it was built from; a
    # hand-edited route can come from a neighbouring window of the same drive, so they are not
    # compared with the scene's token.
    match = os.environ.get('ODYSSEY_ROUTE_FILE', '')
    if not match or not os.path.isfile(match):
        return None, None
    with np.load(match, allow_pickle=False) as z:
        route = np.asarray(z['route_xy'], dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError(f'Invalid SD polyline: {match}')
    return route, match


class SDRouteTracker:
    """Keep score-grid poses plus the exact terminal pose, with real timestamps.

    PDM retains its uniform 0.5 s arrays. This separate buffer never inserts an
    irregular terminal sample into them or borrows a future model prediction.
    """
    def __init__(self, scene, track, handoff_step, stride, rac, map_api=None):
        self.handoff = int(handoff_step)
        self.stride = int(stride)
        self.sim_dt = float(scene['cadence'].sim_dt)
        self.samples = {}
        self.last = None
        self.reference = None
        self.status = 'no_sidecar'
        # The route and the poses share one world frame. The (zero) offset is kept in the saved
        # scoring state.
        self.offset = np.zeros(2)
        route, self.sidecar = load_scene_route()
        self.route_xy = route
        self.reference_gt = None
        if route is None:
            return
        origin = scene['metadata'].get('old_origin_in_current_coordinate')
        if origin is None:
            self.status = 'missing_scene_origin'
            return
        if self.handoff % self.stride:
            raise ValueError('GT handoff must be on the scoring grid')
        # Built once and thinned by each consumer: RC uses the 0.5 s grid, TLC every step. Applying
        # the same transform twice would let the two metrics see different GT if any one of
        # rac/origin/offset differed.
        xy_dense = np.asarray(track['position'], float)[:, :2]
        heading_dense = np.asarray(track['heading'], float)
        rear_dense = xy_dense - rac * np.column_stack((np.cos(heading_dense), np.sin(heading_dense)))
        rear_dense = rear_dense - np.asarray(origin, float).reshape(2) + self.offset
        self.reference_gt_dense = rear_dense
        self.gt_heading_dense = heading_dense
        rear = rear_dense[::self.stride]
        heading = heading_dense[::self.stride]
        self.reference_gt = rear
        start = self.handoff // self.stride
        if len(rear) <= start + 1 or np.linalg.norm(np.diff(rear[start:], axis=0), axis=1).sum() < .5:
            self.status = 'insufficient_gt_motion'
            return
        try:
            self.reference = build_coverage_reference(route, rear, gt_start_index=start)
        except ValueError as exc:
            # Only geometric absence of a denominator is a valid unscorable case.
            if 'GT spans too little' not in str(exc) and 'stationary GT' not in str(exc):
                raise
            self.status = 'insufficient_sd_span'
            return
        self.status = 'reference_suspect' if self.reference.quality['suspect'] else 'ok'

    def observe(self, step, rear_global, heading):
        step = int(step)
        if step < self.handoff:
            return
        pose = (np.asarray(rear_global, float).copy() + self.offset, float(heading))
        if not np.isfinite(pose[0]).all() or not np.isfinite(pose[1]):
            raise ValueError('Non-finite executed pose')
        if self.last is not None and step < self.last[0]:
            raise ValueError('Executed steps must be monotonic')
        self.last = (step, pose)
        # This selects the input path geometry, not a temporal RC scoring rate:
        # SDF later resamples by 5 m, and SDF-zero RC checks 1 m path prefixes.
        # Cadence A/B: using the already archived 0.1 s ego poses
        # instead of this 0.5 s grid + exact terminal pose changed neither RC,
        # SDF, nor first-zero distance in 31 closed-loop runs (5 scenes) and
        # 70 GT replays. The GT crop/graph/termination were held fixed, so this
        # does not assess a denser GT reference. Closed-loop intermediate poses
        # stayed within 0.15 m of the coarse chords; sharp leave-and-rejoin or
        # intersection maneuvers remain unproven. Dense poses are preserved in
        # ds_states/ds_sim_steps for future comparisons.
        if step % self.stride == 0:
            self.samples[step] = pose

    def arrays(self):
        samples = dict(self.samples)
        if self.last is not None:
            samples[self.last[0]] = self.last[1]
        steps = np.array(sorted(samples), dtype=int)
        xy = np.array([samples[s][0] for s in steps], float).reshape(-1, 2)
        heading = np.array([samples[s][1] for s in steps], float)
        return steps, xy, heading, steps * self.sim_dt
