"""The log's driving_command, indexed by where the ego IS along the logged drive instead of by time.

Why this exists. In closed loop the ego is rarely where the log was at the same tick: it is slower
or faster, and a lane-width to the side. Two ways of coping were measured on the recorded frames of
207 finished rollouts:

  * recompute the label at the LIVE pose with OpenScene's own function. Exact at logged poses,
    but the definition is pose-sensitive: it ties the ego to
    the route only within 3 m / 45 deg of a lane centerline. 27% of closed-loop samples sit >= 3 m
    off the logged track; there it agrees with the log on 57% and says `unknown` on 31%.
  * read the label the log stores for the point of the drive the ego has REACHED. OpenScene's
    function returns the stored label at every logged pose (14000 / 14000), so this is the same
    thing as calling it with the ego snapped onto the logged track -- without a map query, a lane
    graph or a re-derived rule at run time, and without the 3 m cliff.

This module is the second one. A navigation command says where the route goes; it should not flip
because the ego is a lane to the side.

`unknown` is FILLED from the GT track, because the models cannot represent it. The log stores it
where nuPlan's route was too damaged for OpenScene's own pass to resolve a command (roughly 10% of
keyframes, concentrated in a few scenes rather than spread). navtrain carries no `unknown`, so a
planner conditioned on that slot would get a one-hot it never saw in training.

So this module derives a replacement, with OpenScene's own rule applied to the logged track instead
of to a lane centerline: snap to the track, read the point +LOOKAHEAD_M along it in the track's own
frame, threshold its lateral offset at +-LATERAL_THRESHOLD_M. Measured against the log's own labels,
that reproduces the label on about 95% of frames with the full 20 m ahead.

It is a SUBSTITUTE, and it says so: every filled frame carries SOURCE_TRACK / SOURCE_TRACK_HELD in
`driving_command_source`, next to the value, so no reader can mistake it for the log's own answer.
`self.onehot` keeps the
log's labels untouched for audit; `self.filled` is what the mode hands out.

Near the end of the track the lookahead cannot shorten. With a fixed +-2 m threshold the lateral
offset shrinks with the distance, so a turn flattens into `straight`: agreement falls 92% (15-20 m
left) -> 80% (10-15) -> 64% (5-10) -> 54% (2-5), and the errors are almost all right/left ->
straight. Scaling the threshold with the distance does not fix it either (it recovers the 5-10 m
band but collapses to 42% under 2 m and invents turns from track wobble). Holding the last command
derived with a FULL lookahead beats both, and beats it monotonically in the floor: taper agreement
69.6% (no hold) -> 73.5% (10 m) -> 76.8% (20 m). Hence the rule below: derive only where the whole
LOOKAHEAD_M is on the track, hold it everywhere after.
"""
import numpy as np
from pyquaternion import Quaternion

#: where a frame's command came from; written to the frame as `driving_command_source`.
SOURCE_LOG = "log_label"          # the label the log stores for that frame, verbatim
SOURCE_TRACK = "gt_lookahead"     # log said `unknown`; derived from the GT track at this frame
SOURCE_TRACK_HELD = "gt_lookahead_held"   # ditto, but < LOOKAHEAD_M of track is left: last one held

UNKNOWN = 3
#: the GT-track lookahead that fills `unknown`. Same numbers as OpenScene's get_driving_command
#: (distance=20, lateral_offset=2) so a filled frame is thresholded exactly like a logged one.
LOOKAHEAD_M = 20.0
LATERAL_THRESHOLD_M = 2.0
#: chord window for the track tangent, so a coarse polyline vertex does not swing the frame
TANGENT_WINDOW_M = 4.0
#: how far the ego's progress may move between two consecutive queries [m]. Backward covers
#: reversing / projection jitter, forward covers a 0.5 s step at motorway speed with margin.
SEARCH_BACK_M = 10.0
SEARCH_AHEAD_M = 30.0
#: logged frames this close to the best match count as the same place (a standstill)
SAME_PLACE_M = 0.05


class DrivingCommandTable:
    """Per-scene: logged ego positions, arc length along them, and one known command per frame."""

    def __init__(self, xy, onehot, source):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.cum = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])
        self.onehot = np.asarray(onehot)
        self.source = list(source)
        #: what the mode hands out: the log's label, or the track-derived one where it said
        #: `unknown`. `self.onehot` stays the log's, untouched, so the two can be compared.
        self.filled, self.source = self._fill_unknown()

    def _point_at(self, s):
        """Track point at arc length ``s``, clamped to the ends."""
        s = min(max(float(s), 0.0), float(self.cum[-1]))
        j = min(max(int(np.searchsorted(self.cum, s) - 1), 0), len(self.xy) - 2)
        seg = self.xy[j + 1] - self.xy[j]
        step = self.cum[j + 1] - self.cum[j]
        return self.xy[j] + ((s - self.cum[j]) / max(step, 1e-6)) * seg

    def _lookahead_command(self, s0):
        """OpenScene's rule on the GT track: one-hot, or None with < LOOKAHEAD_M of track left.

        The frame is the TRACK point at ``s0`` plus the TRACK tangent there -- not the ego's own
        pose -- so the answer describes where the drive goes, not how the ego happens to be
        sitting. Returns None rather than a shortened lookahead: see the module docstring for why
        a shortened one flattens turns into `straight`.
        """
        if float(self.cum[-1]) - s0 < LOOKAHEAD_M:
            return None
        origin = self._point_at(s0)
        tangent = (self._point_at(s0 + 0.5 * TANGENT_WINDOW_M)
                   - self._point_at(s0 - 0.5 * TANGENT_WINDOW_M))
        if np.hypot(tangent[0], tangent[1]) < 1e-6:
            j = min(max(int(np.searchsorted(self.cum, s0) - 1), 0), len(self.xy) - 2)
            tangent = self.xy[j + 1] - self.xy[j]
        theta = np.arctan2(tangent[1], tangent[0])
        target = self._point_at(s0 + LOOKAHEAD_M)
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        lateral = -np.sin(theta) * dx + np.cos(theta) * dy      # track frame, left = +
        onehot = np.zeros(4, dtype=self.onehot.dtype)
        onehot[0 if lateral >= LATERAL_THRESHOLD_M
               else 2 if lateral <= -LATERAL_THRESHOLD_M else 1] = 1
        return onehot

    def _fill_unknown(self):
        """-> (labels, sources). Replace `unknown` with the track-derived command.

        Walked in TRACK order, not in episode order: the held value is "the command derived at the
        last point of the track that still had a full lookahead", which is a property of the track
        and so is the same whichever way the ego drives it. An episode-order hold would make the
        answer depend on the route the ego took to get there.

        A track with nowhere to derive from (shorter than LOOKAHEAD_M end to end) fills nothing and
        leaves `unknown` standing -- there is no honest replacement, and inventing `straight` there
        would be exactly the silent substitution this module refuses.
        """
        labels = self.onehot.copy()
        sources = [SOURCE_LOG] * len(self.xy)
        if len(self.xy) < 2:
            return labels, sources
        held = None
        for i in range(len(self.xy)):
            derived = self._lookahead_command(float(self.cum[i]))
            if derived is not None:
                held = derived
            if np.argmax(labels[i]) != UNKNOWN or held is None:
                continue
            labels[i] = derived if derived is not None else held
            sources[i] = SOURCE_TRACK if derived is not None else SOURCE_TRACK_HELD
        return labels, sources

    def locate(self, ego_xy, anchor_s, time_index):
        """Index of the logged frame the ego has reached.

        Nearest logged position, searched only within [anchor_s - SEARCH_BACK_M, anchor_s +
        SEARCH_AHEAD_M] of arc length so a drive that loops back past itself cannot snap onto the
        wrong pass. ``anchor_s`` is the progress found at the previous query (at the first one,
        the progress of the frame the clock points at). While the log stands still many frames
        share one position; among those the one nearest the clock is taken.
        """
        lo = int(np.searchsorted(self.cum, anchor_s - SEARCH_BACK_M, side="left"))
        hi = int(np.searchsorted(self.cum, anchor_s + SEARCH_AHEAD_M, side="right"))
        lo, hi = min(lo, len(self.xy) - 1), max(hi, lo + 1)
        dist = np.linalg.norm(self.xy[lo:hi] - np.asarray(ego_xy, dtype=np.float64)[:2], axis=1)
        same = np.flatnonzero(dist <= dist.min() + SAME_PLACE_M) + lo
        return int(same[np.argmin(np.abs(same - int(time_index)))])


def build_driving_command_table(infos):
    """Table for one scene from its openscene frame infos.

    `self.onehot` is the log's labels verbatim; `self.filled` is those with `unknown` replaced from
    the GT track (see the module docstring and `_fill_unknown`). The two are kept side by side so a
    reader can always recover what the log actually said.

    An earlier version of the fill re-ran OpenScene's function on the scorer's ROUTE and, failing
    that, borrowed the nearest filled frame's label. Neither is what happens now: there is no route
    and no map query here -- the fill reads the logged track, which is in the scene and cannot be
    mis-baked, and it holds rather than borrows a neighbour.

    :param infos: the scene's openscene_data_infos_dict values, in log order.
    """
    infos = list(infos)
    if not infos:
        raise ValueError("scene has no openscene frame infos to read a driving_command from")
    xy = np.array([np.asarray(i["ego2global_translation"], dtype=np.float64)[:2] for i in infos])
    onehot = np.array([np.asarray(i["driving_command"]) for i in infos])
    source = [SOURCE_LOG] * len(infos)
    return DrivingCommandTable(xy, onehot, source)


def logged_pose(info):
    """(x, y, yaw) of a frame info's ego2global, global UTM -- what OpenScene labelled it at."""
    x, y = np.asarray(info["ego2global_translation"], dtype=np.float64)[:2]
    return float(x), float(y), float(Quaternion(info["ego2global_rotation"]).yaw_pitch_roll[0])
