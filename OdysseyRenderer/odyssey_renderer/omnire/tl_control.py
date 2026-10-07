"""Decide, per sim step, which observed frame's appearance each traffic-light head is drawn with.

Inputs
  control      per-scene tl_control.json (built by build.py from labels, map and manifest).
               This is not a logical signal timetable or TLC ground truth. It is a render-only
               file that only chooses which checkpoint representative frame draws an
               already-decided RED/GREEN.
  timetable    {connector_id: [per-step state]} -- the same as dynamic_map_states in the
               scenario pkl. States may be "TRAFFIC_LIGHT_RED/GREEN/UNKNOWN" or "red"/"green"/None.
  source_row   (connector_id, step) -> scenario row. Pass the simulator's
               agent_manager.traffic_light_source_row unchanged. Identity if absent.
Outputs
  Resolution.mapping  {head_id: source_frame}. Passed to
                      OmniReTrafficLightSubModel.set_source_frames in the default (strict) mode.
                      Heads not listed replay naturally.
  Resolution.report   {head_id: reason} -- for logs and assertions. Enabling control and having
                      nothing change is exactly the failure this guards against, so the reason
                      behind each head's decision is recorded.

Rules (carried over as the labellers defined them)
  1. A head's target state = the timetable state (row chosen by the clock) of the connectors it
     governs. If route_connectors is given, candidates are first restricted by the ego route
     topology. If that route also contains parallel lanes, the actual entry lane is chosen from
     the ego position and heading and swapped for that lane's connector on the same head, so the
     signal actually faced is drawn even when the planned route and the driven lane differ.
  2. keep_natural heads (verdict 'keep') are left untouched while the clock is the identity: the
     labeller confirmed the image already matches the timetable. Once the clock moves rows
     (sector replay) it no longer matches, so they fall through to the representative frame.
  3. If a representative frame of the needed colour exists, use it (colours marked unusable are
     never used).
  4. Otherwise replay naturally inside the training window and use hold_frame (the last observed
     frame) outside it. This case is reported as 'cannot'.
  5. Outside the training window strict mode requires a frame for every head, so node heads with
     no decision this step are filled with hold_frame too.
  6. (opt-in, pin_uncontrolled=True, off by default) Node heads not mapped by the rules above that
     have a representative frame and are not keep_natural are always drawn with a representative
     frame instead of natural replay / hold_frame:
       a. if the red and green representatives are the same frame, that frame (pin:same)
       b. if the DB colour of the connectors this head governs in the top-level connectors map
          (regardless of scope, same clock row) is unique and has a representative, that one
          (pin:db:<colour>)
       c. otherwise the representative frame this head was last drawn with (pin:hold)
       d. if there is none yet, the representative of the first colour the timetable will require
          (pin:lookahead); if no such colour, whichever representative exists, green before red
          (pin:lookahead:any).
          If a sector has not opened yet, scan from the row where it will open -- but the real
          rollout's sector clock returns -1 even for future steps until it opens (opening depends
          on ego progress), so future openings cannot be seen and this falls back to
          pin:lookahead:any. The scan length is capped by LOOKAHEAD_MAX_STEPS.
     Only representative frames are used, so the observed-frame requirement ("source frame must be
     observed") still holds. When off, this path never runs (results are byte-identical).

Unobserved representative frames (scene-specific exception: ALLOW_UNOBSERVED_SCENES only)
  If a labelled head has allow_unobserved {colour: frame}, that head's representative frame for
  that colour need not be an observed frame. Accepted only in labels of ALLOW_UNOBSERVED_SCENES;
  any other scene raises ValueError at construction. The frame must equal that
  colour's representative. When mapped, it is reported as rep_unobserved:<colour> and recorded in
  Resolution.unobserved_ok {head: frame}, so the renderer (set_source_frames) accepts just that
  frame. Labels without this key change nothing.

Scene restrictions (manual audit)
  Some scenes have defective representative-frame verdicts and are not used with tl_control.
  Scenes where only some events are usable cannot guarantee signal consistency for the whole
  rollout either, so they are excluded the same way. The list and each scene's reason are in
  TL_CONTROL_SCENE_EXCLUSIONS below.

Pure Python, no torch. On the simulator side the manager calls this; the renderer only applies
the mapping it receives.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

_RAW = {"TRAFFIC_LIGHT_RED": "red", "TRAFFIC_LIGHT_GREEN": "green",
        "red": "red", "green": "green"}

_ENTRY_ACQUIRE_DISTANCE_M = 6.0
# Scenes allowed unobserved representative frames (an audited exception; not to be generalised).
ALLOW_UNOBSERVED_SCENES = frozenset({"odyssey_scene060", "odyssey_scene075"})
# pin_uncontrolled: max steps scanned ahead for an unopened sector to open (only before a head's
# first pin).
LOOKAHEAD_MAX_STEPS = 300
_ENTRY_HOLD_DISTANCE_M = 35.0
_ENTRY_SWITCH_MARGIN_M = 0.75
_ENTRY_MAX_HEADING_ERROR_RAD = math.radians(60.0)

# Scenes the manual audit judged only partially usable cannot guarantee signal consistency for a
# whole rollout either. This table is a runtime denylist, not documentation: RenderManager checks
# it before reading the control JSON and falls back to natural signal replay on any host.
TL_CONTROL_SCENE_EXCLUSIONS = {
    "odyssey_scene011": "partial only: f966 representative control; f44 unlabeled",
    "odyssey_scene021": "partial only: f136 cannot change",
    "odyssey_scene024": "f36/f702 keep-natural only",
    "odyssey_scene056": "f195/f841 cannot change; f357 keep; f394 signal unknown",
    "odyssey_scene065": "f278 representative control; f893 cannot change",
    "odyssey_scene081": "partial only: f128/f497/f687/f860 unlabeled; f576 signal unknown",
}


def excluded_scene_reason(scene_id) -> Optional[str]:
    """Return the audited reason why tl_control must remain disabled for this scene."""
    match = re.search(r"odyssey_scene\d{3}", str(scene_id))
    return TL_CONTROL_SCENE_EXCLUSIONS.get(match.group(0)) if match else None


def normalize(status) -> Optional[str]:
    """Map one timetable entry to 'red' | 'green' | None. YELLOW never appears in timetables."""
    return _RAW.get(status) if status is not None else None


@dataclass
class Resolution:
    step: int
    in_window: bool
    mapping: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    unobserved_ok: dict = field(default_factory=dict)    # {head: frame}: allowed unobserved reps

    def counts(self) -> dict:
        out: dict = {}
        for reason in self.report.values():
            key = reason.split(":", 1)[0]
            out[key] = out.get(key, 0) + 1
        return out


class TrafficLightController:
    def __init__(self, control: Mapping, timetable: Mapping[str, list],
                 source_row: Optional[Callable[[str, int], int]] = None,
                 map_features: Optional[Mapping] = None, pin_uncontrolled: bool = False):
        if not str(control.get("schema", "")).startswith("tl_control/"):
            raise ValueError("not a tl_control.json (no schema)")
        self.control = control
        self.heads = control["heads"]
        self.connectors = {str(c): list(hs) for c, hs in control["connectors"].items()}
        self.keep_natural = set(control.get("keep_natural", ()))
        self.num_frames = int(control["num_frames"])
        self.timetable = {str(c): list(v) for c, v in timetable.items()}
        self.source_row = source_row or (lambda connector, step: int(step))
        self.map_features = {str(k): v for k, v in (map_features or {}).items()}
        self.node_heads = sorted(h for h, v in self.heads.items() if v.get("node"))
        self.allow_unobserved = self._parse_allow_unobserved()
        self.connector_entry_lanes = {
            connector: tuple(str(x) for x in
                             (self.map_features.get(connector, {}).get("entry_lanes") or ()))
            for connector in self.connectors
        }
        self.entry_connectors = {}
        for connector, entries in self.connector_entry_lanes.items():
            for entry in entries:
                self.entry_connectors.setdefault(entry, []).append(connector)
        missing = sorted(c for c in self.connectors if c not in self.timetable)
        # Connectors missing from the timetable are always UNKNOWN. Passing over them silently
        # would leave "a representative was given but nothing changes" to be chased through the
        # code, so expose them at construction.
        self.missing_connectors = missing
        # opt-in: always draw unmapped node heads with a representative frame too (rule 6). When
        # off, representative frames are not even read -- construction is unchanged (a malformed
        # label still fails only when it is used).
        self.pin_uncontrolled = bool(pin_uncontrolled)
        self.head_connectors = {}
        self.pin_heads = []
        self._last_pin = {}
        if self.pin_uncontrolled:
            for connector, hs in self.connectors.items():
                for h in hs:
                    self.head_connectors.setdefault(str(h), []).append(connector)
            self.pin_heads = sorted(h for h in self.node_heads
                                    if h not in self.keep_natural and self._usable_reps(h))

    @classmethod
    def from_scenario(cls, control, scenario: Mapping, source_row=None, pin_uncontrolled: bool = False):
        """Build the timetable from dynamic_map_states of a scenario pkl (dict)."""
        table = {}
        for obj in (scenario.get("dynamic_map_states") or {}).values():
            if obj.get("type") != "TRAFFIC_LIGHT":
                continue
            table[str(obj["traffic_light_lane"])] = list(obj["state"]["traffic_light_state"])
        return cls(control, table, source_row,
                   map_features=scenario.get("map_features") or {}, pin_uncontrolled=pin_uncontrolled)

    def _parse_allow_unobserved(self):
        """Label allow_unobserved -> {head: {frame: colour}}.

        ValueError outside the allowed scenes or when a frame differs from the representative.
        """
        out = {}
        for h, spec in self.heads.items():
            allow = (spec or {}).get("allow_unobserved")
            if allow is None:
                continue
            match = re.search(r"odyssey_scene\d{3}", str(self.control.get("scene_id", "")))
            scene = match.group(0) if match else None
            if scene not in ALLOW_UNOBSERVED_SCENES:
                raise ValueError(f"{h}: allow_unobserved is only allowed for scenes "
                                 f"{sorted(ALLOW_UNOBSERVED_SCENES)}, not {scene!r}")
            reps = spec.get("representatives") or {}
            if not isinstance(allow, Mapping) or not allow or not spec.get("node"):
                raise ValueError(f"{h}: allow_unobserved must be a non-empty {{colour: frame}} on a node head")
            for colour, frame in allow.items():
                if colour not in ("red", "green") or type(frame) is not int or reps.get(colour) != frame:
                    raise ValueError(f"{h}: allow_unobserved {colour}={frame!r} must equal the {colour} "
                                     f"representative {reps.get(colour)!r}")
                out.setdefault(str(h), {})[frame] = colour
        return out

    # ------------------------------------------------- pin_uncontrolled (opt-in)
    def _usable_reps(self, head):
        spec = self.heads.get(head) or {}
        reps = spec.get("representatives") or {}
        unusable = spec.get("unusable") or {}
        return {c: int(reps[c]) for c in ("red", "green") if reps.get(c) is not None and not unusable.get(c)}

    def _opening_row(self, connector, step, horizon):
        """Current row; if the sector has not opened yet (row < 0), the row at the first future
        step where it opens; None if it does not open within horizon.

        The opening row is only found when source_row reports future openings in advance (identity
        or fixed clock, dry run). The real rollout's sector clock returns -1 for future steps too
        until the ego reaches the sector, so this yields None -> lookahead:any.
        """
        row = int(self.source_row(connector, step))
        if row >= 0:
            return row
        for s in range(step + 1, step + 1 + horizon):
            row = int(self.source_row(connector, s))
            if row >= 0:
                return row
        return None

    def _lookahead(self, head, step, reps):
        """Frame of the first known colour that has a representative, scanning this head's connector
        timetables forward from the current row (or the opening row if the sector is not open)."""
        best = None
        for connector in self.head_connectors.get(head, ()):
            series = self.timetable.get(connector)
            if not series:
                continue
            row0 = self._opening_row(connector, step, min(max(len(series), self.num_frames), LOOKAHEAD_MAX_STEPS))
            if row0 is None:
                continue
            for i in range(row0, len(series)):
                state = normalize(series[i])
                if state in reps:
                    if best is None or i - row0 < best[0]:
                        best = (i - row0, reps[state])
                    break
        return None if best is None else best[1]

    def _pin(self, head, step):
        """Rule 6: frame and reason for an unmapped node head. -> (frame, reason) | (None, None)."""
        reps = self._usable_reps(head)
        if not reps:
            return None, None
        if reps.get("red") is not None and reps.get("red") == reps.get("green"):
            return reps["red"], "same"
        states = set()
        for connector in self.head_connectors.get(head, ()):
            state, _ = self._state(connector, step)
            if state is not None:
                states.add(state)
        if len(states) == 1:
            (state,) = states
            if state in reps:
                return reps[state], f"db:{state}"
        if head in self._last_pin:
            return self._last_pin[head], "hold"
        frame = self._lookahead(head, step, reps)
        if frame is not None:
            return frame, "lookahead"
        for c in ("green", "red"):
            if c in reps:
                return reps[c], "lookahead:any"
        return None, None

    # ------------------------------------------------------ driven-lane scope
    @staticmethod
    def _wrap_angle(value):
        return (float(value) + math.pi) % (2 * math.pi) - math.pi

    @classmethod
    def _polyline_match(cls, feature, xy, heading):
        """Return lateral distance and tangent error at the nearest polyline segment."""
        points = feature.get("polyline") if isinstance(feature, Mapping) else None
        if points is None or len(points) < 2:
            return None
        px, py = float(xy[0]), float(xy[1])
        best = None
        for first, second in zip(points[:-1], points[1:]):
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx, dy = bx - ax, by - ay
            denom = dx * dx + dy * dy
            if denom <= 1e-12:
                continue
            alpha = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
            qx, qy = ax + alpha * dx, ay + alpha * dy
            distance = math.hypot(px - qx, py - qy)
            error = abs(cls._wrap_angle(float(heading) - math.atan2(dy, dx)))
            candidate = (distance, error)
            if best is None or candidate < best:
                best = candidate
        return best

    @staticmethod
    def _endpoint_distance(feature, xy):
        points = feature.get("polyline") if isinstance(feature, Mapping) else None
        if points is None or len(points) == 0:
            return math.inf
        end = points[-1]
        return math.hypot(float(xy[0]) - float(end[0]), float(xy[1]) - float(end[1]))

    @classmethod
    def _turn_angle(cls, feature):
        points = feature.get("polyline") if isinstance(feature, Mapping) else None
        if points is None or len(points) < 3:
            return None
        span = min(5, max(1, (len(points) - 1) // 3))
        start = points[0]
        after = points[span]
        before = points[-1 - span]
        end = points[-1]
        entry = math.atan2(float(after[1]) - float(start[1]),
                           float(after[0]) - float(start[0]))
        exit_heading = math.atan2(float(end[1]) - float(before[1]),
                                  float(end[0]) - float(before[0]))
        return cls._wrap_angle(exit_heading - entry)

    def _choose_actual_connectors(self, entry_lane, planned):
        """Choose the movement from an actual entry lane closest to the planned maneuver."""
        candidates = list(self.entry_connectors.get(entry_lane, ()))
        planned_heads = {head for connector in planned
                         for head in self.connectors.get(connector, ())}
        candidates = [connector for connector in candidates
                      if planned_heads.intersection(self.connectors.get(connector, ()))]
        if not candidates:
            return (), ()

        replaced = [connector for connector in planned
                    if set(self.connectors.get(connector, ())).intersection(
                        head for candidate in candidates
                        for head in self.connectors.get(candidate, ()))]
        if not replaced:
            return (), ()

        planned_turns = [self._turn_angle(self.map_features.get(connector, {}))
                         for connector in replaced]
        planned_turns = [angle for angle in planned_turns if angle is not None]
        if len(candidates) > 1 and planned_turns:
            def turn_error(connector):
                angle = self._turn_angle(self.map_features.get(connector, {}))
                if angle is None:
                    return math.inf
                return min(abs(self._wrap_angle(angle - expected))
                           for expected in planned_turns)
            best_error = min(turn_error(connector) for connector in candidates)
            candidates = [connector for connector in candidates
                          if abs(turn_error(connector) - best_error) < 1e-9]
        return tuple(sorted(candidates)), tuple(sorted(replaced))

    def connector_scope(self, route_connectors: Iterable[str], ego_xy=None,
                        ego_heading=None, previous_entry_lane=None):
        """Narrow a planned route to the signal movement of the lane ego actually occupies.

        ``_route_lane_dict`` remains the topology gate.  It commonly contains parallel lanes in
        the same roadblock, so it is not itself a per-step lane choice.  We map-match ego only
        among controlled entry lanes present in that route, then replace the planned movement
        sharing the same signal head.  Once ego leaves the entry polyline for the intersection,
        the previous choice is held near its stop-line endpoint to avoid switching mid-turn.
        """
        route = tuple(dict.fromkeys(str(x) for x in route_connectors))
        planned = tuple(connector for connector in route if connector in self.connectors)
        diag = {
            "mode": "route",
            "planned_connectors": list(planned),
            "selected_connectors": list(planned),
        }
        if ego_xy is None or ego_heading is None or not self.map_features:
            return planned, diag

        route_set = set(route)
        matches = {}
        for entry_lane in self.entry_connectors:
            if entry_lane not in route_set:
                continue
            match = self._polyline_match(
                self.map_features.get(entry_lane, {}), ego_xy, ego_heading)
            if match is not None and match[1] <= _ENTRY_MAX_HEADING_ERROR_RAD:
                matches[entry_lane] = match

        chosen = None
        reason = None
        if matches:
            best = min(matches, key=lambda lane: matches[lane])
            if matches[best][0] <= _ENTRY_ACQUIRE_DISTANCE_M:
                chosen = best
                reason = "map_match"
                if previous_entry_lane in matches:
                    previous_distance = matches[previous_entry_lane][0]
                    if previous_distance <= matches[best][0] + _ENTRY_SWITCH_MARGIN_M:
                        chosen = previous_entry_lane
                        reason = "hysteresis"

        if chosen is None and previous_entry_lane in self.entry_connectors:
            held_distance = self._endpoint_distance(
                self.map_features.get(previous_entry_lane, {}), ego_xy)
            if held_distance <= _ENTRY_HOLD_DISTANCE_M:
                chosen = previous_entry_lane
                reason = "intersection_hold"

        if chosen is None:
            diag["reason"] = "no_entry_match"
            return planned, diag

        actual, replaced = self._choose_actual_connectors(chosen, planned)
        chosen_match = matches.get(chosen)
        diag.update({
            "mode": "driven_lane" if actual else "route",
            "reason": reason if actual else "no_shared_head_movement",
            "entry_lane": chosen,
            "entry_distance_m": None if chosen_match is None else round(chosen_match[0], 3),
            "entry_heading_error_deg": None if chosen_match is None else round(
                math.degrees(chosen_match[1]), 3),
            "replaced_connectors": list(replaced),
            "actual_lane_connectors": list(actual),
            # TLC keeps the executed/planned connector polygon as its contact gate, but must
            # read the colour from the connector governing the lane ego actually occupied.
            # Keep this explicit instead of asking the scorer to reconstruct the lane choice.
            "signal_source_overrides": (
                {connector: actual[0] for connector in replaced}
                if len(actual) == 1 else {}
            ),
        })
        if not actual:
            return planned, diag
        scope = tuple(connector for connector in planned if connector not in set(replaced)) + actual
        scope = tuple(dict.fromkeys(scope))
        diag["selected_connectors"] = list(scope)
        return scope, diag

    # ------------------------------------------------------------------ core
    def _state(self, connector: str, step: int):
        row = int(self.source_row(connector, step))
        series = self.timetable.get(connector)
        if series is None or row < 0 or row >= len(series):
            return None, row
        return normalize(series[row]), row

    def resolve(self, step: int, route_connectors: Optional[Iterable[str]] = None) -> Resolution:
        step = int(step)
        res = Resolution(step=step, in_window=0 <= step < self.num_frames)
        scope = self.connectors.keys() if route_connectors is None else \
            [c for c in (str(x) for x in route_connectors) if c in self.connectors]

        wanted: dict = {}                       # head -> {state: [(connector, identity)]}
        for c in scope:
            state, row = self._state(c, step)
            if state is None:
                continue
            for h in self.connectors[c]:
                wanted.setdefault(h, {}).setdefault(state, []).append((c, row == step))

        for h, by_state in wanted.items():
            spec = self.heads.get(h) or {}
            if not spec.get("node"):
                continue                        # no node: nothing to draw this light with
            if len(by_state) > 1:
                res.report[h] = "conflict:" + ",".join(
                    f"{s}@{'/'.join(c for c, _ in v)}" for s, v in sorted(by_state.items()))
                continue
            (state, sources), = by_state.items()
            identity = all(ident for _, ident in sources)
            if h in self.keep_natural and identity and res.in_window:
                res.report[h] = f"keep:{state}"
                continue
            reps = spec.get("representatives") or {}
            unusable = spec.get("unusable") or {}
            if state in reps and not unusable.get(state):
                res.mapping[h] = int(reps[state])
                res.report[h] = f"rep:{state}"
            else:
                res.report[h] = f"cannot:{state}"   # no observation of the needed colour

        if self.pin_uncontrolled:
            # Rule 6 (opt-in): unmapped node heads use a representative frame instead of natural
            # replay / hold_frame
            for h in self.pin_heads:
                if h in res.mapping:
                    continue
                frame, why = self._pin(h, step)
                if frame is not None:
                    res.mapping[h] = int(frame)
                    res.report[h] = f"pin:{why}"
            for h in self.pin_heads:
                if h in res.mapping:
                    self._last_pin[h] = int(res.mapping[h])

        if not res.in_window:
            # strict mode: outside the window every node head needs an observed frame
            for h in self.node_heads:
                if h not in res.mapping:
                    res.mapping[h] = int(self.heads[h]["hold_frame"])
                    res.report.setdefault(h, "hold")
                    if res.report[h].startswith("keep"):
                        res.report[h] = "hold"
        if self.allow_unobserved:
            # Label-allowed unobserved representative frames: only that head/frame goes into the
            # report and the renderer allowlist
            for h, frame in res.mapping.items():
                colour = self.allow_unobserved.get(h, {}).get(int(frame))
                if colour is not None:
                    res.report[h] = f"rep_unobserved:{colour}"
                    res.unobserved_ok[h] = int(frame)
        return res
