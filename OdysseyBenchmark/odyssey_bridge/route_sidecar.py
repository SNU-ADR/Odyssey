"""Build route_centerline by re-projecting a baked global route polyline onto the live ego (no map).

StaticRouteCenterline reads the training cache's sdroute, baked ahead of time into one polyline
per scene, and at runtime only projects the ego onto it. No HD map API, OSM matching or PDM route
search runs at all.

Sampling is copied from the training code (route_centerline.py). Only the polyline source
differs; ego projection -> P points at 1 m spacing -> ego-local transform -> vec=diff(pos) ->
mask=real[i]&real[i+1] are identical.

Note -- the baked route is fixed. If ego leaves the route the polyline stays the same, so the
"nearest point" projection keeps returning an answer.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

def _arc(p):
    """Cumulative arc length of a polyline."""
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _yaw(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# Max advance along the route in one step = ADVANCE_SLACK_M + ADVANCE_FACTOR * (distance ego
# actually moved). Derived from observed motion instead of introducing a speed limit. The factor
# allows along-route progress to exceed straight-line motion when the route is offset from the lane
# centre or cuts across a curve; the slack lets the projection move about one segment even while
# stopped (zero motion).
ADVANCE_FACTOR = 3.0
ADVANCE_SLACK_M = 10.0


class RouteCenterlineMissing(RuntimeError):
    """Raised in strict mode when the route centerline for a frame cannot be built.

    """


class RouteExhausted(RouteCenterlineMissing):
    """Ego has passed the end of the baked route and no route remains ahead.

    Unlike a missing sidecar (a data defect), this is a **driving outcome**. If the planner process
    died, the sim would wait for the plan timeout and mark the whole run failed, but the part driven
    so far must be scored normally -- the planner worker catches only this exception, answers
    "route exhausted" for the step and keeps serving.

    It subclasses RouteCenterlineMissing, so every existing except clause catches it as well.
    """


class StaticRouteCenterline:
    """route_centerline generator from baked sidecars; drop-in replacement for RouteCenterlineBuilder."""

    def __init__(self, config, sidecar_dir: str = "", strict: bool = True):
        self._config = config
        self._strict = strict
        self._num_points = int(getattr(config, "route_cl_num_points", 120))
        self._horizon = float(getattr(config, "route_cl_horizon", 120.0))
        # one scene's route file, or a directory of baked sidecars
        self._dir = sidecar_dir or os.environ.get("ODYSSEY_ROUTE_FILE", "")
        self._single = None
        self._n_built = 0
        self._n_failed = 0
        self._logged_once = False
        self._max_lat = 0.0

        self._index: Dict[str, str] = {}      # legacy scene token (= file name) -> npz path
        self._scene_index: Dict[str, str] = {}  # NPZ scene_name -> npz path (exact match)
        self._scene_dups: Dict[str, list] = {}  # scene_names baked by more than one file (see _resolve)
        self._tokens: list = []               # longest first; legacy substring-match candidates
        self._load_index()
        self._sel = None                      # polyline used by this rollout (fixed once chosen)
        self._sel_path = ""
        self._last_s0 = None                  # monotonic gate (see _sample); same lifetime as _sel
        self._last_xy = None                  # previous ego position; basis of the advance limit (_sample)

    # ----------------------------------------------------------------- index -- #
    def _load_index(self) -> None:
        if os.path.isfile(self._dir):
            # One scene's own route file (ODYSSEY_ROUTE_FILE): it is the route. Its `tokens` are
            # not compared with the scene's token (a hand-edited route can come from a
            # neighbouring window of the same drive); the launcher picks the file by scene folder.
            self._single = self._dir
            self._index[os.path.basename(self._dir)[:-4]] = self._dir
            self._tokens = list(self._index)
            return
        if not os.path.isdir(self._dir):
            raise SystemExit(
                f"route file not found: {self._dir!r}. Each published scene ships its route as "
                f"<scene>/route.npz; the launcher passes it as ODYSSEY_ROUTE_FILE.")
        for fn in sorted(os.listdir(self._dir)):
            if not fn.endswith(".npz"):
                continue
            path = os.path.join(self._dir, fn)
            try:
                with np.load(path, allow_pickle=False) as z:
                    _ = z["route_xy"]         # reading it is the corruption check
                    baked_scene = str(z["scene_name"]) if "scene_name" in z.files else ""
                # New sidecars bake the actual scenario package key into scene_name.
                # The file-name token is a fallback for the old ``<log>-<token>`` naming scheme.
                self._index[fn[:-4]] = path
                if baked_scene:
                    # If several files bake the same scene_name, that scene cannot be chosen by
                    # name. Do not exit here -- duplicates of **other** scenes in the directory
                    # would kill unrelated rollouts as soon as the planner starts. Ambiguous
                    # names are only recorded; _resolve exits when that scene is actually run.
                    if baked_scene in self._scene_dups:
                        self._scene_dups[baked_scene].append(path)
                    elif baked_scene in self._scene_index and self._scene_index[baked_scene] != path:
                        self._scene_dups[baked_scene] = [self._scene_index.pop(baked_scene), path]
                    else:
                        self._scene_index[baked_scene] = path
            except Exception as e:
                print(f"[odyssey_bridge] sidecar {fn} unreadable ({type(e).__name__}: {e}); skipped",
                      flush=True)
        if not self._index:
            raise SystemExit(f"no usable route sidecars in {self._dir}")
        # Check longer tokens first: if one token is a prefix of another, the longer one is right.
        self._tokens = sorted(self._index, key=len, reverse=True)

    def _resolve(self, frame: dict):
        """Choose this rollout's polyline by scene identity only, without guessing from coordinates.

        New sidecars keep ``scene_name``, the scenario package key at bake time, inside the NPZ.
        An exact match against the runtime ``frame["scene_name"]`` comes first. Sets whose package
        key is ``..._f1038_1238`` while the file name is a separate label token (e.g. label141)
        are found this way too.

        Older sidecars without that field are found by the file-name token being contained in the
        scene name ``<log>-<scene token>``. Over 1,407 existing rollouts this fallback was always
        unique.

        The frame token (``frame["token"]``) is not used. Closed-loop frame tokens are synthetic
        (``<hex>-000``) and unrelated to the scene token; even with the suffix stripped they can
        point to another scene's sidecar.

        If nothing matches, return None and the caller (build) exits. There is deliberately no
        nearest-route-by-position fallback: in cities several scenes share the same road, so
        start-point distance cannot separate routes that diverge later.
        """
        if self._single is not None:
            return self._select(self._single, "route_file")
        scene = str(frame.get("scene_name") or "")
        if scene in self._scene_dups:
            raise SystemExit(f"ambiguous sidecar scene_name {scene!r}: {self._scene_dups[scene]}")
        path = self._scene_index.get(scene)
        if path is not None:
            how = "scene_name_exact"
        else:
            source_token = str(frame.get("route_sidecar_token") or "")
            path = self._index.get(source_token)
            if path is not None:
                how = "source_lidar_token_exact"
            else:
                hits = [t for t in self._tokens if t and t in scene]
                if len(hits) != 1:
                    return None
                path, how = self._index[hits[0]], "scene_token_legacy"
        return self._select(path, how)

    def _select(self, path: str, how: str):
        with np.load(path, allow_pickle=False) as z:
            xy = np.asarray(z["route_xy"], float)
            segs = []
            if "seg_off" in z.files:
                sxy = np.asarray(z["seg_xy"], float)
                off = np.asarray(z["seg_off"], int)
                anc = np.asarray(z["seg_anchor"], float)
                for k in range(len(off) - 1):
                    p_ = sxy[off[k]:off[k + 1]]
                    if len(p_) >= 2:
                        segs.append(dict(xy=p_, s=_arc(p_), anchor=anc[k]))
            self._sel = dict(xy=xy, s=np.asarray(z["route_s"], float),
                             segs=segs,
                             # Which city this scene is in. Unrelated to the route itself, but
                             # the sidecar is already fixed at one per scene and
                             # route_sidecar_build.py bakes map_location straight from the
                             # scenario metadata. Closed-loop frame dicts have no map_location
                             # (synthetic tokens, so the scenario cannot be reopened), so
                             # city-conditional signals such as traffic side are read here
                             # -- planners/ltf_sdroute.py:LTFSDRouteStatusTrafficSideQueryPlanner.
                             # Old sidecars may lack the field, so it defaults to an empty
                             # string: the reader must then fail rather than silently continue.
                             map_location=str(z["map_location"]) if "map_location" in z.files else "",
                             verdict=str(z["verdict"]), n_hit=int(len(z["src_tokens"])))
        self._sel_path = path
        self._last_s0 = None      # new polyline -> restart the monotonic gate too (see _sample)
        self._last_xy = None
        print(f"[odyssey_bridge] route sidecar: {os.path.basename(path)} via {how} "
              f"({len(xy)} pts / {self._sel['s'][-1]:.0f} m, {self._sel['n_hit']} cached frames, "
              f"verdict={self._sel['verdict']}, map={self._sel['map_location'] or '?'})",
              flush=True)
        return self._sel

    def map_location(self, frame: dict) -> str:
        """City this rollout runs in (`sg-one-north` etc.). Empty string if undetermined.

        Read from the same npz and the same selection as the route, so the city cannot disagree
        with the scene the route came from. Called by planners that use city-conditional signals
        (traffic side); an empty string means the caller **must fail**: silently falling back to
        right-hand traffic would score Singapore scenes under a condition never seen in training.
        """
        sel = self._sel or self._resolve(frame)
        return str(sel["map_location"]) if sel else ""

    # ------------------------------------------------------------- sampling -- #
    def _project(self, poly: np.ndarray, s: np.ndarray, p: np.ndarray,
                 s_min: Optional[float] = None, heading: Optional[float] = None,
                 s_max: Optional[float] = None) -> Tuple[float, float]:
        """Project point p onto the polyline -> (arc length, lateral distance).

        With s_min, only segments reaching that arc length or beyond are candidates. The route is
        ordered start -> destination, so increasing s is the direction of travel and spans already
        passed have no reason to be candidates again. Without the limit, where the route comes back
        near itself (a road turning off at an intersection, an out-and-back road) and ego is far
        off, two different s values become nearly equidistant, argmin flips and s0 jumps back into
        the past -- measured at frame 291->292 of one run: ego moved 0.3 m forward, but candidate
        distances 31.69 m and 31.85 m (0.16 m apart) flipped and s0 fell back 63 m (88.9 -> 25.7 m).

        Blocking candidates up front differs from correcting after the choice in lat. A correction
        only rolls back s0 and still returns lat to the wrong candidate (s=25 above), so s0 and lat
        point at different places. Blocking here keeps both from the same segment.

        With heading, segments running against the direction of travel (more than 90 degrees from
        the tangent) are excluded. This stops the opposite-lane branch of an out-and-back road from
        snapping in at equal distance -- it is the only condition that applies on the **first
        frame**, which has no s_min (measured in one run: on the first frame s=5 m and s=430 m were
        both 3.0 m away, the far end was picked, and the monotonic gate locked it in so only 48
        points came out for all 200 frames).

        With s_max, segments starting beyond that arc length are excluded. The advance along the
        route in one step is bounded by how far ego actually moved -- needed for spans driven twice
        in the same direction (turnaround loops), which heading cannot separate (measured in one
        run: s0 skipped 117 m, 31 -> 148.6 m, within 0.5 s and then stuck).

        Omitting s_min/heading/s_max behaves exactly as before -- the diagnostic lat measurement
        (build) must measure the true nearest point, so it is unconstrained.
        """
        a, b = poly[:-1], poly[1:]
        ab = b - a
        L2 = (ab ** 2).sum(1).clip(1e-9)
        t = (((p - a) * ab).sum(1) / L2).clip(0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        # All three limits below only remove candidates. For why they do not correct after the
        # choice, see the docstring above (s0 and lat would point at different places).
        blocked = np.zeros(len(d), dtype=bool)
        if s_min is not None:
            # If segment i's end arc length s[i+1] is below s_min, the whole segment is passed.
            # Segments straddling s_min are kept -- when t clamps to the s_min end, s0 naturally
            # becomes s_min; dropping that segment would make s0 jump forward in normal following.
            blocked |= s[1:] < s_min
        if s_max is not None:
            # A segment starting beyond s_max cannot be reached this step. Straddling segments
            # are kept for the same reason as on the s_min side.
            blocked |= s[:-1] > s_max
        if heading is not None:
            ang = np.arctan2(ab[:, 1], ab[:, 0])
            blocked |= np.abs((ang - heading + np.pi) % (2 * np.pi) - np.pi) > 0.5 * np.pi
        if blocked.any() and not bool(blocked.all()):   # if everything is blocked, drop the limits
            d = np.where(blocked, np.inf, d)
        i = int(np.argmin(d))
        return float(max(s[i] + t[i] * np.sqrt(L2[i]), s_min if s_min is not None else -np.inf)), \
            float(d[i])

    def _pick_segment(self, sel, ox: float, oy: float):
        """Use route_xy as-is, since the bake step already **stitched** it together.

        A per-frame segment is not picked: a segment is only 118 m, so whichever one is picked runs
        out as ego advances, and switching makes the route length jump and leaves the area ahead of
        ego empty. The bake step instead trims the overlaps and stitches one polyline (appending only
        the new tail instead of a median average, so the route is not smeared).

        seg_* remain in the npz for diagnostics and are not used here.
        """
        return sel

    def _sample(self, sel, ox: float, oy: float, oh: float):
        """Same sampling as route_centerline.py. -> (rc (P,5), mask (P,), lat)."""
        import torch

        P = self._num_points
        poly, s = sel["xy"], sel["s"]
        length = float(s[-1])
        rc = torch.zeros(P, 5, dtype=torch.float32)
        mask = torch.zeros(P, dtype=torch.bool)

        # The route is ordered start -> destination, so increasing s is the direction of travel.
        # Excluding passed spans from the candidates makes this function's premise ("forward from
        # the nearest point") actually hold -- without it argmin can flip to a past span, and the
        # window, although it extends forward, covers road ego has already driven (measurements
        # in _project).
        #
        # s0 sticking while stopped or creeping backwards is intended -- the window reaches 120 m
        # ahead of s0, so even when stuck it still covers where ego actually is.
        #
        # Reset per scene: _sel is one per rollout, so this matches its lifetime.
        # Max advance along the route this step, derived from how far ego actually moved (no new
        # constant such as a speed limit). The factor and slack allow for a route offset from the
        # lane or cutting across curves. The first frame has no previous position, so no limit.
        p_now = np.array([ox, oy], float)
        s_max = None
        if self._last_s0 is not None and self._last_xy is not None:
            moved = float(np.linalg.norm(p_now - self._last_xy))
            s_max = self._last_s0 + ADVANCE_SLACK_M + ADVANCE_FACTOR * moved
        s0, lat = self._project(poly, s, p_now, s_min=self._last_s0, heading=oh, s_max=s_max)
        self._last_s0 = s0
        self._last_xy = p_now
        if length - s0 < 1.0:
            return rc, mask, lat

        spacing = self._horizon / P
        dist = s0 + np.arange(P) * spacing
        real = dist <= length
        q = np.clip(dist, s0, length)
        gx = np.interp(q, s, poly[:, 0])
        gy = np.interp(q, s, poly[:, 1])
        # heading is the polyline tangent. Training uses PDMPath's discrete_path heading, which
        # equals the tangent on a polyline resampled at 1 m. Interpolate after unwrap to avoid
        # jumps at the +-pi boundary.
        th = np.unwrap(np.arctan2(np.gradient(poly[:, 1], s), np.gradient(poly[:, 0], s)))
        gh = np.interp(q, s, th)

        c, sn = np.cos(oh), np.sin(oh)
        dx, dy = gx - ox, gy - oy
        pos = np.stack([c * dx + sn * dy, -sn * dx + c * dy], axis=1)   # global -> ego
        heading = gh - oh
        vec = np.zeros_like(pos)
        vec[:-1] = np.diff(pos, axis=0)

        valid = real.copy()
        valid[:-1] &= real[1:]
        valid[-1] = False
        pos[~valid] = 0.0
        vec[~valid] = 0.0
        heading[~valid] = 0.0
        if not valid.any():
            return rc, mask, lat

        rc[:, 0:2] = torch.from_numpy(pos).float()
        rc[:, 2:4] = torch.from_numpy(vec).float()
        rc[:, 4] = torch.from_numpy(heading).float()
        mask[:] = torch.from_numpy(valid)
        return rc, mask, lat

    # ----------------------------------------------------------------- build -- #
    def build(self, frame: dict, nav_pose=None) -> Optional[Tuple["object", "object"]]:
        """frame = latest raw simulator .pkl dict -> (route_centerline (1,P,5), mask (1,P)).

        nav_pose = (x, y, heading) global, optional. When given, it is used **only to sample the
        route**; everything else (segment choice below) keeps using the true pose. None = the
        true pose.
        """
        sel = self._sel or self._resolve(frame)
        if sel is None:
            return self._fail(
                f"scene {frame.get('scene_name')!r}: no exact baked scene_name or unique legacy "
                f"sidecar token match (dir={self._dir}, {len(self._index)} sidecars). There is "
                f"deliberately no nearest-route-by-position fallback: it can pick the route of "
                f"another scene on the same road.")

        # Sidecar route_xy is in the same world frame as ego2global_translation.
        trans = np.asarray(frame["ego2global_translation"], float)[:2]
        oh = _yaw(frame["ego2global_rotation"])
        tx, ty = float(trans[0]), float(trans[1])                       # real_ego (UTM world)
        if nav_pose is None:
            nx, ny, nh = tx, ty, oh
        else:
            nx, ny, nh = float(nav_pose[0]), float(nav_pose[1]), float(nav_pose[2])
        # If per-frame segments exist, pick the one matching the live ego. The merged polyline
        # (sel) is a fallback for old sidecars without segments -- when frames disagree, merging
        # builds a line matching neither route, which caused cache reproduction error
        # (14 DISAGREE cases: median 2.07 m, worst 29.6 m).
        pick = self._pick_segment(sel, tx, ty)                          # true pose
        rc, mask, lat = self._sample(pick, nx, ny, nh)                  # route the model sees = nav pose
        if nav_pose is not None:
            _, lat = self._project(pick["xy"], pick["s"], np.array([tx, ty], float))
        self._max_lat = max(self._max_lat, lat)

        if not bool(mask.any()):
            return self._fail(
                f"frame {frame.get('token')!r}: baked route exhausted (ego is past its end; "
                f"route is {sel['s'][-1]:.0f} m from {sel['n_hit']} cached frames, "
                f"verdict={sel['verdict']}). The training cache does not cover the whole log — "
                f"see the end_m column in the sidecar summary.",
                exc=RouteExhausted)

        self._n_built += 1
        if not self._logged_once:
            self._logged_once = True
            print(f"[odyssey_bridge] route centerline ON (static sidecar): {int(mask.sum())}/"
                  f"{self._num_points} valid points over {self._horizon} m, ego lat {lat:.2f} m",
                  flush=True)
        return rc.unsqueeze(0), mask.unsqueeze(0)

    def _fail(self, msg: str, exc=RouteCenterlineMissing):
        self._n_failed += 1
        if self._strict:
            raise exc(
                msg + " Refusing to plan with the route conditioning silently disabled; pass "
                      "--route-centerline off to run this checkpoint without it anyway.")
        print(f"[odyssey_bridge] WARNING: {msg} Planning WITHOUT route conditioning this step.",
              flush=True)
        return None

    @property
    def stats(self) -> str:
        return (f"route centerline (static sidecar {os.path.basename(self._sel_path) or '-'}): "
                f"{self._n_built} built, {self._n_failed} failed, max ego lat {self._max_lat:.1f} m")

    @property
    def branch(self):
        """--force-branch needs the map-based builder (branch geometry comes from the lane graph)."""
        return None
