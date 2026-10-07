"""TLC scoring against our signal timetable (rule ``timetable``): the one TLC score for the designated set.

End to end:
  Timetable   the timetable directory (OdysseyBenchmark/data/tlc_timetable):
                designated.json   the designated set -- the signals the user named (scene, roadblock, connectors,
                                  showable colours). Only these are scored.
                <scene>.json      the scene's timetable, a tl_signal_patch/1 file (GT rows 0..999, R/G spans). The
                                  rollout applies it at the signal source (signal_patch, tl_signal_patch_path); it may
                                  also hold cross-approach spans (role "cross") -- they change what the planner, IDM
                                  and renderer see but are never scored.
              load_timetable() checks it loudly: designated.json present, every designated scene has a file, every
              designated signal is in its file with the same connectors (ValueError otherwise).
  Colour      per (step, route connector): the colour of the signal the ego obeyed at its source row.
                online          the run recorded the patch (TLC snapshot signal_patched): the recorded patched state,
                                checked cell by cell against the file (online_check; loud, the batch exits non-zero,
                                and a run recorded with another timetable file is named by the batch).
                offline         a run without the patch: the recorded DB state with the timetable applied here through
                                the recorded source rows (GT row = source row // scene_frame_stride).
  Not scored  not_designated (roadblock not designated for the scene), row_before_open (source row -1),
              row_past_log (held past the log, or GT row > 999), unknown (no known colour), not_showable:<R|G> (no
              render head of the signal can show that colour, designated.json showable).
  Contact     a RED scored cell whose zone the ego reaches counts when the ego area inside the sliver-tolerant
              union of the same roadblock's RED scored zones is >= overlap_min (1.0: the ego is wholly inside --
              full overlap). A zone is the route connector's polygon, widened lateral_margin (4 m) to either side
              in a designated roadblock. A stop line / exit edge is the polygon edge through which the connector's
              pinned baseline enters / leaves it, and the added area never lies behind any stop line of the
              roadblock nor past the widened connector's own exit edge. Full overlap therefore means the whole car
              is past the stop lines and within lateral_margin of a red connector, touching it or not (a car in the
              next lane counts), and the gaps nuPlan leaves between neighbouring connectors are sealed. When an end
              edge of any connector of a roadblock cannot be located, that roadblock is not widened and its
              connectors are listed in unwidened_connectors (driving_metrics.apply_rules refuses such a run).
  Charges     score_tlc (the events grouping) on a copy in which exactly the contact cells are RED: one event per
              intersection (roadblock) passage; render / GT waivers as score_tlc says; the near-side waiver is off.
              P_TL = 0.7 ** charged events.
  Coverage    always reported: per designated signal, route cells the ego touched and how many of them were scored
              (and why the rest were not). Low coverage means the score carries little information.

driving_metrics.apply_rules runs this for the designated scenes (rule tl_set) and multiplies P_TL into RouteDS.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.wkb import loads as load_wkb

from odyssey.manager import signal_patch
from .tlc_metric import DENSE_VERSION, VERSION as PINNED_VERSION, score_tlc

VERSION = 'tlc_timetable/2'
RULE = 'timetable'
GT_ROWS = 1000
LETTER = {'RED': 'R', 'GREEN': 'G', 'YELLOW': 'Y'}
NAME = {'R': 'RED', 'G': 'GREEN'}
GREY = '-'
OVERLAP_MIN = 1.0          # the ego box wholly inside the red union (was 0.5)
OVERLAP_TOL = 1e-6         # floating-point noise on the area ratio
UNION_CLOSE_M = 1e-3       # close the union (1 mm): seals touching edges and gaps under 2 mm
LATERAL_MARGIN_M = 4.0     # connectors widened 4 m to either side, never behind a stop line
HALF_PLANE_M = 1000.0      # size of the squares that stand in for half-planes: far beyond any connector
END_EDGE_MIN_DEG = 45.0    # a stop line / exit edge meets the baseline at least this steeply (shipped timetable: 62-90 deg)
END_EDGE_CORNER_FRACTION = 0.15  # ... and is crossed this far from either end of the edge (shipped: 0.28-0.75 along it)


def source_of(tlc, r, j):
    """(connector whose signal ego obeyed, scenario log row shown) for route column j, row r."""
    conns = tlc.get('signal_state_connector_ids')
    rows = tlc.get('signal_source_rows')
    connector = conns[r][j] if conns else tlc['connector_ids'][j]
    log_row = rows[r][j] if rows else tlc['sim_steps'][r]
    return str(connector), int(log_row)


def _closed_union(polygons):
    """Union of connector polygons without slivers: buffer(+d) then buffer(-d), d = UNION_CLOSE_M.
    Seals floating-point gaps between touching edges and any gap under 2d; the outer boundary
    stays in place (convex corners to d resolution)."""
    if len(polygons) == 1:
        return polygons[0].buffer(UNION_CLOSE_M).buffer(-UNION_CLOSE_M)
    return unary_union([p.buffer(UNION_CLOSE_M) for p in polygons]).buffer(-UNION_CLOSE_M)


def _half_plane(a, b, inner):
    """The half-plane on ``inner``'s side of the line through ``a`` and ``b``, as a square HALF_PLANE_M across.
    None when a-b has no length or ``inner`` lies on the line."""
    (ax, ay), (bx, by) = a, b
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6:
        return None
    ux, uy = (bx - ax) / length, (by - ay) / length
    nx, ny = -uy, ux
    side = (inner.x - ax) * nx + (inner.y - ay) * ny
    if abs(side) < 1e-6:
        return None
    if side < 0:
        nx, ny = -nx, -ny
    h = HALF_PLANE_M / 2
    return Polygon([(ax - ux * h, ay - uy * h), (ax + ux * h, ay + uy * h),
                    (ax + ux * h + nx * HALF_PLANE_M, ay + uy * h + ny * HALF_PLANE_M),
                    (ax - ux * h + nx * HALF_PLANE_M, ay - uy * h + ny * HALF_PLANE_M)])


def _end_sides(polygon, path):
    """(entry side, exit side) of a connector: the inner half-planes of the polygon edges through which its baseline
    ``path``, run on straight beyond both ends, first enters the polygon (the stop line) and last leaves it (the exit
    edge). A lane's baseline crosses its stop line and exit edge between the lane's two sides, square enough: None
    when the crossing lies within END_EDGE_CORNER_FRACTION of either end of the edge, when the edge meets the
    baseline at less than END_EDGE_MIN_DEG (a lane side or a corner, not a stop line), when both ends land on one
    edge, or when the polygon or path is unusable."""
    if polygon.is_empty or polygon.geom_type != 'Polygon' or path is None or len(path) < 2:
        return None
    points = [(float(xy[0]), float(xy[1])) for xy in path]
    if not all(math.isfinite(v) for xy in points for v in xy):
        return None
    points = [xy for i, xy in enumerate(points) if i == 0 or xy != points[i - 1]]
    if len(points) < 2 or LineString(points).length < 1e-3:
        return None

    def run_on(a, b):                                       # HALF_PLANE_M / 2 beyond b, straight on from a
        d, k = math.hypot(b[0] - a[0], b[1] - a[1]), HALF_PLANE_M / 2
        return b[0] + (b[0] - a[0]) / d * k, b[1] + (b[1] - a[1]) / d * k

    line = LineString([run_on(points[1], points[0])] + points + [run_on(points[-2], points[-1])])
    crossing = line.intersection(polygon.exterior)
    found = sorted(line.project(Point(xy)) for part in getattr(crossing, 'geoms', [crossing]) for xy in part.coords)
    if not found or found[-1] - found[0] < 1e-3:
        return None
    ring = [xy[:2] for xy in polygon.exterior.coords]
    edges = [LineString((a, b)) for a, b in zip(ring, ring[1:]) if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9]
    min_sin = math.sin(math.radians(END_EDGE_MIN_DEG))
    picked, sides = [], []
    for at, inside in ((found[0], found[0] + 0.5), (found[-1], found[-1] - 0.5)):
        point, ahead = line.interpolate(at), line.interpolate(inside)
        edge = min(edges, key=point.distance)
        (ax, ay), (bx, by) = edge.coords
        tx, ty = ahead.x - point.x, ahead.y - point.y
        if math.hypot(tx, ty) < 1e-6:
            return None
        squareness = abs((bx - ax) * ty - (by - ay) * tx) / (edge.length * math.hypot(tx, ty))
        where = edge.project(point) / edge.length
        if squareness < min_sin or not END_EDGE_CORNER_FRACTION <= where <= 1 - END_EDGE_CORNER_FRACTION:
            return None                                     # a lane side or a corner, not a stop line
        side = _half_plane((ax, ay), (bx, by), ahead)
        if side is None:
            return None
        picked.append(edge)
        sides.append(side)
    return None if picked[0].equals(picked[1]) else tuple(sides)


def _lateral_zones(members, polygons, paths, margin):
    """One roadblock's connectors widened ``margin`` metres to either side. -> ({connector: zone}, [not widened]).

    A connector's widening is its round buffer clipped to the inner side of its own exit edge and of every stop line
    in the roadblock, so the added area never lies behind a stop line nor past that connector's exit. When the end
    edges of any connector cannot be located, no connector of the roadblock is widened (all are reported): a stop
    line that cannot be found cannot be respected.
    """
    zones = {c: polygons[c] for c in members}
    if margin <= 0:
        return zones, []
    sides = {c: _end_sides(polygons[c], paths.get(c)) for c in members}
    if any(found is None for found in sides.values()):
        return zones, list(members)
    past_stop_lines = None
    for entry, _ in sides.values():
        past_stop_lines = entry if past_stop_lines is None else past_stop_lines.intersection(entry)
    for c, (_, exit_side) in sides.items():
        grown = polygons[c].buffer(margin).intersection(past_stop_lines).intersection(exit_side)
        zones[c] = polygons[c].union(grown.buffer(0))
    return zones, []


# ---- timetable set ---------------------------------------------------------------------------------------------------
from odyssey.utils.directory_cache import directory_cache


@directory_cache
def load_timetable(path):
    """-> {dir, designated, designated_sha256, scenes, files{scene: (doc, sha256, path)},
    designated_set_version}. Loud (ValueError / FileNotFoundError) on anything that does not fit together."""
    d = Path(path)
    raw = (d / 'designated.json').read_bytes()
    designated = json.loads(raw)
    scenes = sorted({str(x['scene']) for x in designated.get('signals') or ()})
    if not scenes:
        raise ValueError(f'timetable {d}: designated.json names no signal')
    files = {}
    for s in scenes:
        f = d / f'{s}.json'
        if not f.is_file():
            raise ValueError(f'timetable {d}: designated scene {s} has no timetable file {f.name}')
        doc, sha = signal_patch.parse(f.read_bytes())
        if doc['scene'] != s:
            raise ValueError(f'timetable {f}: file is for scene {doc["scene"]}, not {s}')
        entries = {(str(g.get('roadblock')), tuple(sorted(str(c) for c in g['connectors'])))
                   for g in doc['signals'] if g.get('role') != 'cross'}
        for x in designated['signals']:
            if str(x['scene']) != s:
                continue
            key = (str(x['roadblock']), tuple(sorted(str(c) for c in x['connectors'])))
            if key not in entries:
                raise ValueError(f'timetable {f}: designated signal {s}/{key[0]} {list(key[1])} has no entry with '
                                 f'the same connectors')
        files[s] = (doc, sha, str(f.resolve()))
    # previous_sha256 / designated_set_version are kept as keys (empty / None) for callers that read them
    return dict(dir=str(d), designated=designated, designated_sha256=hashlib.sha256(raw).hexdigest(), scenes=scenes,
                files=files, previous_sha256={s: () for s in scenes},
                designated_set_version=None)


def designated_for(designated, scene):
    """-> {roadblock: {connectors, showable: {R: bool, G: bool}}} for one scene from designated.json."""
    out = {}
    for d in (designated or {}).get('signals') or ():
        if d['scene'] != scene:
            continue
        show = d.get('showable') or {}
        out[str(d['roadblock'])] = dict(connectors=[str(c) for c in d['connectors']],
                                        showable={c: bool((show.get(c) or {}).get('showable', True)) for c in 'RG'})
    return out


def patch_spans(doc):
    """-> {connector: [(first_gt_row, last_gt_row, 'R'|'G', source)]} from a tl_signal_patch/1 doc."""
    out = {}
    for sig in (doc or {}).get('signals') or ():
        for p in sig['patches']:
            for c in sig['connectors']:
                out.setdefault(str(c), []).append((int(p[0]), int(p[1]), p[2], p[3]))
    return out


def _span_at(spans, connector, gt_row):
    for a, b, colour, source in spans.get(connector, ()):
        if a <= gt_row <= b:
            return colour, source
    return None


def _flag(tlc, key, r, j, default=False):
    rows = tlc.get(key)
    return bool(rows[r][j]) if rows is not None else default


# ---- colours ---------------------------------------------------------------------------------------------------------
def cell_colours(tlc, patch_doc, patch_sha256=None, previous_sha256=()):
    """-> (recoloured copy of tlc, mode, patched cells {(r, j)}, online_check).

    mode 'online' when the run recorded the patch (signal_patched), else 'offline' (patch applied here).
    previous_sha256: earlier byte hashes of the same timetable (only its notes changed); a run that recorded one
    of them matches."""
    table = list(tlc['signal_code_table'])
    spans = patch_spans(patch_doc)
    stride = int(tlc.get('scene_frame_stride') or 1)
    steps = [int(s) for s in tlc['sim_steps']]
    conns = [str(c) for c in tlc['connector_ids']]
    online = tlc.get('signal_patched') is not None
    out = copy.deepcopy(tlc)
    patched, mismatch, unflagged, stray = set(), [], [], []
    for r in range(len(steps)):
        for j in range(len(conns)):
            source, row = source_of(tlc, r, j)
            held = _flag(tlc, 'signal_held', r, j)
            hit = _span_at(spans, source, row // stride) if row >= 0 and not held else None
            if online:
                flagged = bool(tlc['signal_patched'][r][j])
                name = table[int(tlc['signal_states'][r][j])]
                if flagged:
                    patched.add((r, j))
                if hit and not flagged:
                    unflagged.append([steps[r], conns[j], row])
                elif flagged and not hit:
                    stray.append([steps[r], conns[j], row])
                elif hit and name != NAME[hit[0]]:
                    mismatch.append([steps[r], conns[j], row, name, hit[0]])
            elif hit:
                out['signal_states'][r][j] = table.index(NAME[hit[0]])
                patched.add((r, j))
    check = None
    if online:
        run_sha = tlc.get('signal_patch_sha256')
        check = dict(run_patch_sha256=run_sha, patch_sha256=patch_sha256,
                     sha_matches=patch_sha256 is None or run_sha == patch_sha256 or run_sha in previous_sha256,
                     cells_patched=len(patched), unflagged=unflagged[:20], n_unflagged=len(unflagged),
                     stray=stray[:20], n_stray=len(stray), colour_mismatch=mismatch[:20],
                     n_colour_mismatch=len(mismatch))
        check['ok'] = bool(check['sha_matches'] and not unflagged and not stray and not mismatch)
    return out, 'online' if online else 'offline', patched, check


def coverage(result, detail):
    """-> {signals: {rb: {touched, scored, coverage, not_scored: {reason: cells}}}, touched, scored, coverage}:
    route cells of designated signals the ego touched, and how many of them were scored."""
    desig = set(result['designated_roadblocks'])
    per = {}
    for (r, j) in sorted(detail['touch']):
        rb = detail['rb_of'][detail['conns'][j]]
        if rb not in desig:
            continue
        x = per.setdefault(rb, dict(touched=0, scored=0, not_scored={}))
        x['touched'] += 1
        why = detail['why_not'].get((r, j))
        if why is None:
            x['scored'] += 1
        else:
            x['not_scored'][why] = x['not_scored'].get(why, 0) + 1
    for x in per.values():
        x['coverage'] = round(x['scored'] / x['touched'], 4) if x['touched'] else None
    touched = sum(x['touched'] for x in per.values())
    scored = sum(x['scored'] for x in per.values())
    return dict(signals=dict(sorted(per.items())), touched=touched, scored=scored,
                coverage=round(scored / touched, 4) if touched else None)


# ---- score -----------------------------------------------------------------------------------------------------------
def evaluate(tlc, ego_polygons, dense_steps, scene, designated, patch_doc, patch_sha256=None,
             overlap_min=None, previous_sha256=(), lateral_margin=None):
    """-> (result, detail). designated = designated.json doc; patch_doc = the scene's timetable (tl_signal_patch/1;
    None: no timetable -- every cell keeps its recorded colour); lateral_margin = metres each connector is widened to
    either side for the contact rule (default LATERAL_MARGIN_M; 0: the connector polygons as they are). Prefer
    score(), which takes a loaded timetable."""
    overlap_min = OVERLAP_MIN if overlap_min is None else float(overlap_min)
    if not 0.0 < overlap_min <= 1.0:
        raise ValueError(f'overlap_min must be in (0, 1], got {overlap_min}')
    lateral_margin = LATERAL_MARGIN_M if lateral_margin is None else float(lateral_margin)
    if not 0.0 <= lateral_margin < math.inf:
        raise ValueError(f'lateral_margin must be a finite distance >= 0 m, got {lateral_margin}')
    if tlc.get('version') not in (PINNED_VERSION, DENSE_VERSION):
        raise ValueError(f'{VERSION} needs a pinned snapshot, got {tlc.get("version")}')
    if patch_doc is not None and patch_doc.get('scene') != scene:
        raise ValueError(f'{VERSION}: timetable is for {patch_doc.get("scene")}, not {scene}')
    desig = designated_for(designated, scene)
    recoloured, mode, patched, check = cell_colours(tlc, patch_doc, patch_sha256, previous_sha256)
    table = list(recoloured['signal_code_table'])
    red_code, unknown_code = table.index('RED'), table.index('UNKNOWN')
    stride = int(tlc.get('scene_frame_stride') or 1)
    steps = [int(s) for s in tlc['sim_steps']]
    by_step = {int(s): i for i, s in enumerate(dense_steps)}
    conns = [str(c) for c in tlc['connector_ids']]
    polys = {c: load_wkb(tlc['connector_wkb'][c], hex=True) for c in conns}
    groups = tlc.get('connector_roadblock') or {}
    rb_of = {c: str(groups.get(c, c)) for c in conns}
    rbs = {}
    for c in conns:
        rbs.setdefault(rb_of[c], []).append(c)
    paths = tlc.get('connector_paths') or {}
    zones, unwidened = dict(polys), []
    for rb, members in rbs.items():
        if rb in desig:
            widened, missing = _lateral_zones(members, polys, paths, lateral_margin)
            zones.update(widened)
            unwidened += missing

    # 1) grid: colour, and whether the cell is scored (and red), or why not
    colour, code, why_not, ev_red, rows_of = {}, {}, {}, set(), {}
    for r in range(len(steps)):
        for j, c in enumerate(conns):
            source, row = source_of(tlc, r, j)
            col = LETTER.get(table[int(recoloured['signal_states'][r][j])], 'U')
            colour[(r, j)] = col
            rows_of[(r, j)] = row // stride if row >= 0 else row
            d = desig.get(rb_of[c])
            if d is None:
                reason = 'not_designated'
            elif row < 0 or _flag(tlc, 'signal_inactive', r, j):
                reason = 'row_before_open'
            elif _flag(tlc, 'signal_held', r, j) or row // stride >= GT_ROWS:
                reason = 'row_past_log'
            elif col not in ('R', 'G', 'Y'):
                reason = 'unknown'
            elif col in d['showable'] and not d['showable'][col]:
                reason = 'not_showable:' + col
            else:
                reason = None
            if reason is None:
                code[(r, j)] = col
                if col == 'R':
                    ev_red.add((r, j))
            else:
                code[(r, j)] = GREY
                why_not[(r, j)] = reason

    # 2) contact rule: ego area share inside the union of the same roadblock's scored-red zones (the connectors,
    #    widened sideways in designated roadblocks); `touch` -- the connector polygons themselves -- feeds coverage only
    unions = {}

    def zone_union(members):
        key = tuple(members)
        if key not in unions:
            unions[key] = _closed_union([zones[c] for c in key])
        return unions[key]

    red_ratio, all_ratio, touch, reach = {}, {}, set(), set()
    for r, step in enumerate(steps):
        ego = ego_polygons[by_step[step]]
        area = ego.area
        for rb, members in rbs.items():
            near = [c for c in members if ego.intersects(zones[c])]
            if not near or area <= 0:
                continue
            for c in near:
                reach.add((r, conns.index(c)))
                if ego.intersects(polys[c]):
                    touch.add((r, conns.index(c)))
            all_ratio[(r, rb)] = ego.intersection(zone_union(members)).area / area
            red = [c for c in members if (r, conns.index(c)) in ev_red]
            if red:
                red_ratio[(r, rb)] = ego.intersection(zone_union(red)).area / area
    contact = {(r, j) for (r, j) in ev_red if (r, j) in reach
               and red_ratio.get((r, rb_of[conns[j]]), 0.0) >= overlap_min - OVERLAP_TOL}

    # 3) events: score_tlc on a copy where exactly the contact cells are RED; near-side waiver off. score_tlc finds
    #    the contacts again by intersection, so it gets the zones the contact rule used.
    masked = copy.deepcopy(recoloured)
    masked['apply_nearside_turn_filter'] = False
    masked['connector_wkb'] = {c: wkb if zones.get(c, polys.get(c)) is polys.get(c) else zones[c].wkb_hex
                               for c, wkb in masked['connector_wkb'].items()}
    for r in range(len(steps)):
        for j in range(len(conns)):
            if int(masked['signal_states'][r][j]) == red_code and (r, j) not in contact:
                masked['signal_states'][r][j] = unknown_code
    capture = {}
    scored = score_tlc(masked, ego_polygons, dense_steps, capture=capture)
    row_of = {s: r for r, s in enumerate(steps)}
    gt = scored['tlc_filter_reason'] == 'gt'
    events = []
    for event in json.loads(scored['tlc_events_json']):
        members = [str(c) for c in event.get('connectors', [event['connector']])]
        mine = [(row_of[int(s)], conns.index(str(c)))
                for s, c in zip(capture['tlc_contact_sim_steps'].tolist(), capture['tlc_contact_connector_ids'].tolist())
                if str(c) in members and event['first_step'] <= int(s) <= event['last_step']]
        n_patched = sum((r, j) in patched for r, j in mine)
        gt_rows = [rows_of[(r, j)] for r, j in mine]
        render_w = bool(event.get('render_waived'))
        events.append(dict(
            roadblock=event.get('roadblock'), connectors=sorted(members),
            first_step=int(event['first_step']), last_step=int(event['last_step']),
            first_row=int(event['first_frame']), last_row=int(event['last_frame']),
            **{'class': 'designated'}, charged=not render_w and not gt, render_waived=render_w,
            nearside_waived=False, contacts=len(mine), patched_contacts=n_patched,
            colour_source='camera' if mine and n_patched == len(mine) else 'db' if not n_patched else 'mixed',
            max_overlap=round(max(red_ratio.get((r, rb_of[conns[j]]), 0.0) for r, j in mine), 4) if mine else None,
            gt_rows=[min(gt_rows), max(gt_rows)] if gt_rows else None))
    charged = sum(e['charged'] for e in events)
    reasons = {}
    touched_reasons = {}
    for k, v in why_not.items():
        reasons[v] = reasons.get(v, 0) + 1
        if k in touch:
            touched_reasons[v] = touched_reasons.get(v, 0) + 1
    on_route = sorted({rb_of[c] for c in conns if rb_of[c] in desig})
    counts = dict(events=len(events), charged=charged, contacts=len(contact),
                  touches_below_overlap=len({k for k in ev_red if k in touch and k not in contact}),
                  scored_cells=len(code) - len(why_not), not_scored_cells=reasons,
                  not_scored_touched_cells=touched_reasons,
                  patched_cells=len(patched), patched_scored_cells=len([k for k in patched if k not in why_not]),
                  patched_touched_cells=len(patched & touch))
    reason = 'no_violation' if not events else 'gt' if gt else 'unwaived_violation' if charged else 'rendering_sanity'
    detail = dict(steps=steps, conns=conns, rb_of=rb_of, colour=colour, code=code, why_not=why_not,
                  patched=patched, red_ratio=red_ratio, all_ratio=all_ratio, touch=touch, contact=contact,
                  gt_row=rows_of, recoloured=recoloured)
    result = dict(version=VERSION, rule=RULE, source_tlc_version=tlc['version'], scene=scene,
                  patch_applied=mode, patch_sha256=patch_sha256, online_check=check,
                  designated_roadblocks=sorted(desig), designated_on_route=on_route,
                  tlc_filter_reason=reason, p_tl=float(0.7 ** charged), charged_count=charged,
                  counts=counts, events=events, overlap_min=overlap_min, lateral_margin_m=lateral_margin,
                  unwidened_connectors=sorted(unwidened),
                  showable={rb: d['showable'] for rb, d in sorted(desig.items())})
    result['coverage'] = coverage(result, detail)
    return result, detail


def score(tlc, ego_polygons, dense_steps, scene, timetable, overlap_min=None, lateral_margin=None):
    """The entry point: score one run's TLC snapshot against a loaded timetable (load_timetable). -> (result, detail).

    ValueError when the scene is not designated (nothing to score)."""
    if scene not in timetable['files']:
        raise ValueError(f'{VERSION}: scene {scene} is not in the designated set of {timetable["dir"]}')
    doc, sha, path = timetable['files'][scene]
    result, detail = evaluate(tlc, ego_polygons, dense_steps, scene, timetable['designated'], doc, sha, overlap_min,
                              previous_sha256=timetable.get('previous_sha256', {}).get(scene, ()),
                              lateral_margin=lateral_margin)
    result['timetable'] = dict(dir=timetable['dir'],
                               designated_sha256=timetable['designated_sha256'], file=path, sha256=sha)
    return result, detail
