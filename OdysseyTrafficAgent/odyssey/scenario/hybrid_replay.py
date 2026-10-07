"""Keying background traffic on ego progress instead of the wall clock.

A closed-loop benchmark hands ego control to a policy, so the ego reaches any given
point of the route at a time the recording never saw. The background actors are still
replayed from that log, and `trajectory_policy` indexes them by `episode_step` -- the
log's own clock -- so the ego and the traffic decouple: drive slower than the recording
and the traffic is gone before the ego arrives, drive faster and it has not appeared yet.

This module keys them on **ego route progress** instead:

* **ambient** actors -- everything that is not a candidate for interaction -- share one
  clock, `tau(s)`: the log time at which the recorded ego held arc length `s`. A monotone
  reparametrisation of log time preserves their relative geometry exactly, so ambient
  traffic is always in a configuration the recording contained, and the ego meets it at
  the same point of the road the recording did.

* **interactive** actors -- the ones that share the ego's road and could be reacted to --
  are released on that clock and then **free-run** on the sim clock, so a collision is
  possible. If they stayed on `tau(s)` they would stop when the ego stops, and nothing
  could ever hit anything.

* interactive actors that belong to the same **platoon** are released together, or the
  rear one is let go at a different time from the one ahead and drives through it.
  Platoon membership is post-encroachment time: two actors are coupled when they ever
  occupied the same patch of road within a few seconds of each other.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# The scale is a lane: traffic sharing the ego's lane has to land on one side of this
# line and oncoming traffic on the other, because the two classes run on different
# clocks and anything left on the wrong side is driven through by the other side.
TAU_INTERACT = 4.0        # m from the ego route
# A parked box jitters in place, so its net travel stays near zero while a car creeping
# through a queue accumulates. Net displacement, not largest excursion.
STATIC_MIN = 3.0          # m of net displacement
# Post-encroachment time: how long after one actor left a patch of road the other
# reached it. Finite for a platoon holding station, which is what time to collision is
# not -- two cars 20 m apart at 15 m/s never close, so their TTC is infinite, yet 1.3 s
# of release skew puts one through the other. 3 s is the conflict threshold the
# traffic-safety literature already uses.
PET_S = 3.0               # s
PET_R = 5.0               # m -- the patch is about a vehicle length across
# Pairwise PET is O(n^2) in actors and O(n*m) in frames. Scenes resampled to 10 Hz make
# that unaffordable at reset for no gain: the thresholds were validated at 2 Hz.
PET_MAX_HZ = 2.0
# How much log time one sector spans. The longer it is the longer the traffic runs at the
# speed it was recorded at, and the rarer -- but larger -- the re-anchoring step at the
# boundary. Cutting on log time rather than on distance puts the boundaries where the
# recording spent its time, so a jam becomes many short sectors. Where the recorded ego
# was exactly stationary those boundaries land on the same arc length and are crossed
# together, and the last one wins: that stretch of the log is skipped rather than
# replayed. This is the viewer's behaviour and is kept.
SECTOR_LEN_S = 20.0
# Spawn lead distance: a sector's actors are put on the road this far before the sector start.
# At 0 they appear the moment the boundary is reached, so they pop out right beside the ego, and
# sometimes where the ego has already passed.
SECTOR_LEAD_M = 15.0
# Intersection-based split. Boundaries go midway between intersection crossings, so no
# intersection is cut by a boundary. Closely spaced intersections are merged into one sector by
# the minimum length.
SECTOR_INTER_MIN_M = 50.0       # m; boundaries closer than this are not placed
# Intersection lane polygons do not always cover where the ego drove (especially at three-way
# junctions). If the route comes within this distance, the intersection counts as crossed.
SECTOR_INTER_APPROACH_M = 5.0   # m


def _polyline_project(route_xy, cum, pts):
    """Nearest point on the route for each query: (arc length, distance)."""
    a = route_xy[:-1]
    d = route_xy[1:] - a
    denom = np.einsum("ij,ij->i", d, d)
    denom[denom == 0] = 1e-9
    rel = pts[:, None, :] - a[None, :, :]
    u = np.clip(np.einsum("nij,ij->ni", rel, d) / denom[None, :], 0.0, 1.0)
    proj = a[None, :, :] + u[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)
    j = np.argmin(dist, axis=1)
    n = np.arange(len(pts))
    return cum[j] + u[n, j] * np.sqrt(denom[j]), dist[n, j]


class EgoProgressClock:
    """Map the live ego pose to a monotone row of its source trajectory.

    This is deliberately only a clock.  It does not classify actors, change their
    motion, or decide whether a spawn is safe.  A lifecycle consumer can substitute
    the returned source row for ``episode_step`` and keep every existing spawn,
    retry, despawn, and retention rule unchanged.
    """

    def __init__(self, route_xy, rows=None):
        self.route_xy = np.asarray(route_xy, dtype=np.float64)
        if self.route_xy.ndim != 2 or self.route_xy.shape[1] < 2:
            raise ValueError("ego progress route must have shape [N, >=2]")
        self.route_xy = self.route_xy[:, :2]
        if len(self.route_xy) < 2:
            raise ValueError("ego progress route needs at least two points")
        segment_lengths = np.linalg.norm(np.diff(self.route_xy, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if rows is None:
            rows = np.arange(len(self.route_xy), dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.float64).reshape(-1)
        if len(self.rows) != len(self.route_xy):
            raise ValueError("ego progress rows and route must have the same length")
        if np.any(np.diff(self.rows) < 0):
            raise ValueError("ego progress rows must be monotone")
        self._s_max = 0.0

    def progress(self, ego_xy):
        """Return the non-rewinding arc length the live ego has reached on the route."""
        progress, _ = _polyline_project(
            self.route_xy,
            self.cum,
            np.asarray(ego_xy, dtype=np.float64).reshape(1, 2),
        )
        self._s_max = max(self._s_max, float(progress[0]))
        return self._s_max

    def source_row(self, ego_xy):
        """Return a non-rewinding source row for the current live ego pose."""
        return float(np.interp(self.progress(ego_xy), self.cum, self.rows))


class HybridReplay:
    """Per-episode replay clock. Built once at reset, stepped once per sim step.

    `step(episode_step, ego_xy)` returns `{object_id: log row index}` for every non-ego
    actor. The caller writes those onto the agents; validity, spawning and despawning
    then follow from the same index, because an index outside an actor's valid periods
    means it is not on the road yet or not any more.
    """

    def __init__(self, route_xy, route_heading, tracks, *,
                 tau_interact=TAU_INTERACT, static_min=STATIC_MIN,
                 pet_s=PET_S, pet_r=PET_R, src_dt=0.5):
        """`tracks` maps object id -> (frame indices, xy) over that actor's valid frames."""
        self._clock = EgoProgressClock(route_xy)
        self.route_xy = self._clock.route_xy
        self.cum = self._clock.cum
        self.n_frames = len(self.route_xy)
        self.src_dt = float(src_dt)
        self._rows = np.arange(self.n_frames, dtype=np.float64)

        heading = np.unwrap(np.asarray(route_heading, dtype=np.float64))
        # Classification and PET are both quadratic -- actor frames against route points,
        # then actor against actor. A scene resampled to 10 Hz makes that unaffordable at
        # reset for no gain, because every threshold here was validated at 2 Hz. Both
        # tests therefore run on a strided view; only tau(s), which is one point against
        # the route per step, reads the dense arrays.
        self._stride = max(1, int(round(1.0 / (PET_MAX_HZ * self.src_dt))))
        self._route_lo = self.route_xy[::self._stride]
        self._cum_lo = self.cum[::self._stride]
        heading_lo = heading[::self._stride]

        self.k0 = {}            # object id -> first valid frame
        self.interactive = {}   # object id -> bool
        for oid, (k, xy) in tracks.items():
            self.k0[oid] = int(k[0])
            self.interactive[oid] = self._is_interactive(
                k[::self._stride], xy[::self._stride], heading_lo,
                tau_interact, static_min, xy)

        self.bundle = self._bundle(tracks, pet_s, pet_r)
        self.bundle_k0 = {}
        for oid, b in self.bundle.items():
            self.bundle_k0[b] = min(self.bundle_k0.get(b, 1 << 30), self.k0[oid])

        self._released = {}     # bundle id -> episode_step at which it was let go

        n_int = sum(1 for v in self.interactive.values() if v)
        logger.info("HYBRID_REPLAY actors=%d interactive=%d bundles=%d",
                    len(tracks), n_int, len(set(self.bundle.values())))

    # -- classification ------------------------------------------------------
    def _is_interactive(self, k, xy, heading, tau_interact, static_min, xy_full):
        """Three rules, each with one sentence of justification.

        The actor has to share the road the ego drives on for a collision to be
        possible at all; it has to actually move, since a parked car replays the same
        either way and only muddies the accounting; and it has to have been ahead of
        the ego at some point, because a follower cannot be yielded to, and if it runs
        into the ego that is the follower's fault.
        """
        _, dist = _polyline_project(self._route_lo, self._cum_lo, xy)
        if float(np.min(dist)) >= tau_interact:
            return False
        # net displacement over the whole track, not the strided view
        if float(np.linalg.norm(xy_full[-1] - xy_full[0])) < static_min:
            return False
        # The recorded ego held route_xy[k] at frame k, so the separation the two of
        # them actually had is read off the frames they share.
        kc = np.clip(k // self._stride, 0, len(self._route_lo) - 1)
        rel = xy - self._route_lo[kc]
        h = heading[kc]
        ahead = float(np.max(np.cos(h) * rel[:, 0] + np.sin(h) * rel[:, 1]))
        return ahead > 0.0

    # -- bundling ------------------------------------------------------------
    def _bundle(self, tracks, pet_s, pet_r):
        """Union-find over interactive actors, joined by post-encroachment time.

        Transitive by construction, so a queue of five chains into one bundle rather
        than four overlapping pairs.
        """
        inter = [oid for oid, v in self.interactive.items() if v]
        parent = {o: o for o in inter}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        stride = self._stride
        sub = {o: (tracks[o][0][::stride], tracks[o][1][::stride]) for o in inter}
        for i in range(len(inter)):
            ka, pa = sub[inter[i]]
            for j in range(i + 1, len(inter)):
                kb, pb = sub[inter[j]]
                ra, rb = find(inter[i]), find(inter[j])
                if ra == rb:
                    continue
                d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
                m = d < pet_r
                if not m.any():
                    continue
                pet = float((np.abs(ka[:, None] - kb[None, :]) * self.src_dt)[m].min())
                if pet < pet_s:
                    parent[ra] = rb

        out = {}
        roots = {}
        for o in inter:
            r = find(o)
            out[o] = roots.setdefault(r, len(roots))
        return out

    # -- runtime -------------------------------------------------------------
    def tau(self, ego_xy):
        """Ego position -> the log row at which the recorded ego held this progress.

        Clamped monotone: if the ego backs up or a lateral excursion drags the
        projection backwards, the background must not rewind. Freezing at the furthest
        point reached is itself an artifact, but a rewinding one is worse.
        """
        return self._clock.source_row(ego_xy)

    def step(self, episode_step, ego_xy):
        """The log row every non-ego actor should be showing at this sim step."""
        tau = self.tau(ego_xy)
        out = {}
        for oid, is_int in self.interactive.items():
            if not is_int:
                out[oid] = tau
                continue
            b = self.bundle[oid]
            if b not in self._released:
                if tau < self.bundle_k0[b]:
                    out[oid] = -1.0       # not on the road yet
                    continue
                self._released[b] = episode_step
            # Every member of a bundle reads the same row: a member's own logged offset
            # from the bundle's earliest appearance is already baked into its track, and
            # a row before its first valid frame simply reads as not yet present.
            out[oid] = self.bundle_k0[b] + (episode_step - self._released[b])
        return out


def _ring_contains(ring, pts):
    """Ray casting: whether each point is inside the polygon."""
    x, y = pts[:, 0], pts[:, 1]
    xi, yi = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    inside = np.zeros(len(pts), dtype=bool)
    for i in range(len(ring)):
        dy = yj[i] - yi[i]
        if dy == 0.0:
            continue
        crosses = (yi[i] > y) != (yj[i] > y)
        xint = (xj[i] - xi[i]) * (y - yi[i]) / dy + xi[i]
        inside ^= crosses & (x < xint)
    return inside


def _ring_distance2(ring, pts):
    """Squared minimum distance from each point to the polygon's **edges** (segments, not vertices)."""
    a = ring
    d = np.roll(ring, -1, axis=0) - a
    length = np.einsum("ij,ij->i", d, d)
    length = np.where(length > 0, length, 1e-12)
    t = np.clip(np.einsum("nij,ij->ni", pts[:, None, :] - a[None], d) / length[None], 0.0, 1.0)
    proj = a[None] + t[:, :, None] * d[None]
    return ((pts[:, None, :] - proj) ** 2).sum(axis=2).min(axis=1)


def intersection_bounds(route_xy, cum, polygons, *, approach_m=SECTOR_INTER_APPROACH_M,
                        min_m=SECTOR_INTER_MIN_M):
    """Return sector boundaries (arc length) midway between intersection crossings.

    The point of this split is to put boundaries **outside** intersections. If the time offset
    changed in the middle of an intersection, the traffic through it would be split across the
    boundary.

    Always includes 0; each following boundary is accepted only if it is at least `min_m` past the
    previous one, so closely spaced intersections are merged into one sector. If the last sector
    would be shorter than that, its boundary is dropped (the tail joins the previous sector).
    """
    inside = np.zeros(len(route_xy), dtype=bool)
    r2 = float(approach_m) ** 2
    for poly in polygons:
        ring = np.asarray(poly, dtype=np.float64)[:, :2]
        if len(ring) < 3:
            continue
        lo, hi = ring.min(axis=0) - approach_m, ring.max(axis=0) + approach_m
        near = ((route_xy >= lo) & (route_xy <= hi)).all(axis=1) & ~inside
        if not near.any():
            continue
        pts = route_xy[near]
        hit = _ring_contains(ring, pts) | (_ring_distance2(ring, pts) <= r2)
        idx = np.flatnonzero(near)
        inside[idx[hit]] = True

    spans = []                                  # merge contiguous stretches
    i = 0
    while i < len(inside):
        if not inside[i]:
            i += 1
            continue
        start = i
        while i < len(inside) and inside[i]:
            i += 1
        spans.append((cum[start], cum[i - 1]))

    keep = [0.0]
    for (_, s1), (s0, _) in zip(spans, spans[1:]):
        middle = (s1 + s0) / 2.0
        if middle - keep[-1] >= min_m:
            keep.append(float(middle))
    if len(keep) > 1 and cum[-1] - keep[-1] < min_m:
        keep.pop()
    return np.asarray(keep, dtype=np.float64)


class SectorReplay:
    """Hold the offset fixed across a sector of the route, so the log plays at log speed.

    A port of the sector mode in the odyssey_nr_trigger viewer, on its log-time split.
    The route is cut into sectors spanning a fixed stretch of log time; when the ego
    first enters sector k the offset is fixed once, `delta_k = gt_k - sim_k`, the log row
    that sector starts at minus the step the ego entered it on. An actor belongs to the
    earliest sector touched by its valid trajectory and reads that sector's offset, so
    `row_j(t) = t + delta_k`.

    Inside a sector the row advances one per step, so the traffic keeps the speed it was
    recorded at rather than the ego's. That is what `HybridReplay` costs: there `tau(s)`
    is continuous in the ego's position, so an ego at 0.6x puts the city in slow motion
    and a stopped ego freezes it.

    This clock therefore has no use for the ambient/interactive split, the release, or
    the PET bundles. Those exist so that traffic which would otherwise freeze along with
    the ego can still run into something; here nothing freezes. The viewer has none of
    them either.

    With a lead the offset is anchored entirely at the spawn line -- the log row the
    recording held there, minus the step the live ego reached it on -- so an ego that
    drives the recording still reproduces it exactly. Taking the row from the sector line
    and the step from the spawn line would leave the lead's traversal time in the offset.

    **The one place the closed loop cannot follow the viewer.** There `sim_k` is known
    for every sector before the episode starts, because the ego is driven along the
    recorded path at a set multiple of its speed, so a track is drawn from step 0 with an
    offset read off a future the viewer already has. Here the ego is driven by a policy
    and `sim_k` exists only once the ego is there. Until then the sector has no offset
    and its actors are not on the road -- the same answer the viewer gives for a sector
    the ego never reaches. Assigning actors by the earliest progress their trajectory
    reaches is the conservative closed-loop adaptation of the viewer's first-sighting
    rule: oncoming traffic is available before it travels back through an already-passed
    junction instead of materializing there when its first-sighting sector eventually opens.
    """

    def __init__(self, route_xy, tracks, *, signals=None,
                 sector_len_s=SECTOR_LEN_S, lead_m=0.0,
                 intersections=None, inter_min_m=SECTOR_INTER_MIN_M,
                 inter_approach_m=SECTOR_INTER_APPROACH_M, src_dt=0.5):
        """`tracks` maps object id -> (frame indices, xy) over that actor's valid frames.

        If `intersections` is given, boundaries go between intersection crossings. Otherwise log
        time is cut every `sector_len_s`.
        """
        self._clock = EgoProgressClock(route_xy)
        self.cum = self._clock.cum
        self.src_dt = float(src_dt)
        if intersections:
            bound_s = intersection_bounds(self._clock.route_xy, self.cum, intersections,
                                          approach_m=inter_approach_m, min_m=inter_min_m)
            # boundary arc length -> first log row that reached that point
            self.bound_rows = np.searchsorted(self.cum, bound_s, side="left")
        else:
            rows_per_sector = max(1, int(round(float(sector_len_s) / self.src_dt)))
            self.bound_rows = np.arange(0, len(self.cum), rows_per_sector)
        # The arc length the recorded ego held at each boundary: what the live ego's
        # progress is compared against to decide it has entered the next sector.
        self.bound_s = self.cum[self.bound_rows]

        self.sector = {}        # object id -> earliest route sector touched by its track
        self.actor_valid_bounds = {}
        for oid, (valid_rows, xy) in tracks.items():
            valid_rows = np.asarray(valid_rows, dtype=int)
            self.actor_valid_bounds[oid] = (int(valid_rows[0]), int(valid_rows[-1]))
            # First-sighting assignment is not safe for traffic moving against the ego
            # route.  Such an actor can first be observed beyond the next boundary, then
            # travel back through the ego's current junction while its sector is still
            # closed.  Opening that sector materializes the actor beside (or behind) the
            # ego in a burst.  Own the actor by the earliest route progress reached by any
            # valid pose instead.  This is monotone-conservative: a track may be admitted
            # earlier, but can never be withheld past a place its recorded trajectory
            # actually occupies.
            track_s, _ = _polyline_project(self._clock.route_xy, self.cum, xy)
            earliest_s = float(np.min(track_s))
            k = int(np.searchsorted(self.bound_s, earliest_s, side="right")) - 1
            self.sector[oid] = max(0, k)

        # A signal connector belongs to the sector containing the representative point
        # serialized in the scenario PKL.  The point is approximately 8 m along the lane
        # connector, not a lamp-head coordinate; it is used only to select the already
        # existing sector clock.  Missing/non-finite points remain deliberately unassigned
        # and therefore read UNKNOWN instead of silently falling back to wall time.
        self.signal_sector = {}
        for connector_id, position in (signals or {}).items():
            point = np.asarray(position, dtype=np.float64).reshape(-1)
            if len(point) < 2 or not np.all(np.isfinite(point[:2])):
                continue
            signal_s, _ = _polyline_project(
                self._clock.route_xy, self.cum, point[None, :2]
            )
            sector = int(np.searchsorted(
                self.bound_s, float(signal_s[0]), side="right"
            ) - 1)
            self.signal_sector[str(connector_id)] = int(np.clip(
                sector, 0, len(self.bound_s) - 1
            ))

        # Spawn boundary: `lead` metres before the sector start. When the ego reaches it, that
        # sector's actors come onto the road and the offset is fixed once, at that point.
        self.lead_m = float(lead_m)
        self.spawn_s = np.maximum(self.bound_s - self.lead_m, 0.0)
        # Both terms of the offset must come from the **same point**: the row at which the recorded
        # ego passed the spawn line minus the step at which the live ego passed it. Subtracting the
        # spawn line's step from the sector line's row leaves the lead traversal time in the
        # offset, so the log is not replayed exactly even when the ego drives the recording.
        # With lead 0 the spawn line is the sector line, so its row is used as is -- where the
        # recorded ego was stopped, several rows share one arc length, and searching again would
        # return an earlier row.
        self.spawn_rows = (self.bound_rows if self.lead_m == 0.0
                           else np.searchsorted(self.cum, self.spawn_s, side="left"))

        self._delta = []        # sector -> offset in rows, appended as the ego enters it
        # Needed for retrospective TLC packing.  Once later sectors have opened, a query for
        # an earlier simulation step must still report those sectors as inactive.
        self._opened_at = []    # sector -> first simulation step using that sector clock

        logger.info("SECTOR_REPLAY actors=%d sectors=%d split=%s lead=%.1fm",
                    len(tracks), len(self.bound_rows),
                    ("intersections" if intersections else "%.1fs" % float(sector_len_s)),
                    self.lead_m)

    def step(self, episode_step, ego_xy):
        """The log row every non-ego actor should be showing at this sim step."""
        progress = self._clock.progress(ego_xy)
        # The offset is fixed once, when the actors come onto the road (= the spawn line), and
        # never changes after that. Nothing happens at the sector line itself -- re-aligning there
        # would make the row jump at that moment, and actors whose valid period spans it would pop
        # up at the boundary.
        #
        # With lead 0 the spawn line is the sector line, matching the previous sector behaviour.
        # With a lead both terms still come from the spawn line, so driving the recording makes
        # every offset 0 and the log replays exactly -- the counterpart of the viewer having the
        # same property at the sector line.
        spawned = int(np.searchsorted(self.spawn_s, progress, side="right"))
        for k in range(len(self._delta), spawned):
            self._delta.append(float(self.spawn_rows[k]) - episode_step)
            self._opened_at.append(int(episode_step))

        out = {}
        for oid, k in self.sector.items():
            out[oid] = -1.0 if k >= len(self._delta) else episode_step + self._delta[k]
        return out

    def signal_rows(self, episode_step):
        """Return the source row used by every assigned connector at ``episode_step``.

        This is a historical query, not merely a view of the clock's current state.  A
        connector whose sector had not opened at the requested step returns ``-1`` even if
        that sector opened later in the rollout.
        """
        return {
            connector_id: self.signal_row(connector_id, episode_step)
            for connector_id in self.signal_sector
        }

    def signal_row(self, connector_id, episode_step):
        """Direct form of :meth:`signal_rows` for per-connector consumers."""
        sector = self.signal_sector.get(str(connector_id))
        step = int(episode_step)
        if sector is None or sector >= len(self._delta) or step < self._opened_at[sector]:
            return -1.0
        return step + self._delta[sector]

    def is_eligible(self, object_id):
        """Whether this actor's spawn line has been crossed.

        Reactive IDM uses this monotone gate for admission. Its source-valid lifetime
        uses actor_lifecycle_row, while motion after admission remains IDM-integrated.
        NR replay and traffic lights continue to use the shared sector row above.
        """
        sector = self.sector.get(str(object_id))
        return sector is not None and sector < len(self._delta)

    def actor_lifecycle_row(self, object_id, episode_step):
        """R-only source row: gate-open time plus the actor's original valid duration.

        NR replay and traffic-light rows keep their shared sector offset unchanged.
        """
        token = str(object_id)
        sector = self.sector.get(token)
        step = int(episode_step)
        if (sector is None or sector >= len(self._opened_at)
                or step < self._opened_at[sector]):
            return -1
        first, last = self.actor_valid_bounds[token]
        row = first + step - self._opened_at[sector]
        return row if row <= last else -1
