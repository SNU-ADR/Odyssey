"""Camera-verified signal patch at the signal source (opt-in ``tl_signal_patch_path``, default null).

For the designated signals the
DB timetable (scenario ``dynamic_map_states``) is patched once, where every consumer reads it, so the ego planner
(PDM), reactive IDM, the dense reward, the online metric, the TLC signal snapshot and the renderer (tl_control) all
see the same colour for the same (connector, source row).

Where the per-step status is produced (paths under OdysseyTrafficAgent/odyssey):
  * ``OdysseyToNuPlanConverter.convert_to_traffic_lights`` (components/agents/policy/pdm_planner/utils/odyssey_to_pdm_utils.py)
    reads ``scene['dynamic_map_states'][..]['state']['traffic_light_state'][source_row]`` with
    ``source_row = agent_manager.traffic_light_source_row``. Callers: PDM ego planner (pdm_policy.py), reactive IDM
    (nuplan_idm_policy.get_traffic_light_status_at_iteration), the online metric (metric_manager.py) and the TLC
    snapshot (tlc_metric.capture_tlc_signal_snapshot).
  * ``TrafficLightController.from_scenario`` (OdysseyRenderer/odyssey_renderer/omnire/tl_control.py) copies the same arrays as its timetable,
    with the same ``traffic_light_source_row`` clock.
So the single point is the scene's ``traffic_light_state`` arrays: ScenarioManager applies the patch there right
after loading/resampling, before any manager, policy or converter holds the scene.

Patch file (``<set>/<scene>.json``, e.g. OdysseyBenchmark/data/tlc_timetable/odyssey_scene060.json, or a directory holding those)::

  {"schema": "tl_signal_patch/1", "scene": "odyssey_scene060",
   "signals": [{"roadblock": "50634", "connectors": ["52588", "53009", "53137"],
                "patches": [[0, 93, "R", "user+camera", {...evidence}], [94, 134, "G", "camera", {...}]]}],
   "provenance": {...}}

Rows are GT scene rows 0..999 (10 Hz, both ends inclusive). If the scene is upsampled by S (cadence.upsample_n) a
GT row r covers scene rows r*S .. r*S+S-1 (the resampler holds signals with a nearest-floor sample). Colours: R, G.
Every named connector must have a DB signal in the scene; a signal may not name a connector twice; spans may not
overlap per connector. A span that lies wholly past a connector's log patches nothing and one that runs past it is
clipped: both are logged as warnings and counted in the record (spans_past_log, spans_clipped).

IDM: nuplan_idm_policy treats an intersection whose signals are ALL UNKNOWN at a step as unsignalized (UNKNOWN ->
GREEN for route planning, FIFO/corridor rules for admission). Filling UNKNOWN rows turns that fallback off on those
rows -- intended (the camera shows a signal there), but it changes IDM behaviour. The record counts, per signal, the
scene rows where every connector of the signal was UNKNOWN before the patch and is known after
(idm_all_unknown_rows_filled, and the connector cells, idm_all_unknown_cells_filled), and a warning line says so. The
IDM groups connectors by the map's intersection, which can hold more than one designated signal, so this is the upper
bound of rows where the fallback stops.

The scene carries the result as ``scene['tl_signal_patch']`` (sha256, path, spans, rows per connector); consumers
use :func:`is_patched` for the per-(connector, row) flag recorded in the npz. Rendering-only edge rules (row -1 ->
the sector's spawn-row colour, row past the log -> hold the last row) live in render_manager; the scorer never
uses them. The spawn-row colour exists only with the sector-replay clock (agent_manager traffic_light_spawn_row
reads the sector clock's spawn rows); with the identity clock (GT replay, absolute) no row is -1 and the rule never
fires -- a sector that has not opened has no spawn row to draw otherwise, and the renderer keeps its usual -1 handling.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from odyssey.manager import tlc_timetable_set

logger = logging.getLogger(__name__)

SCHEMA = 'tl_signal_patch/1'
GT_ROWS = 1000
RAW = {'R': 'TRAFFIC_LIGHT_RED', 'G': 'TRAFFIC_LIGHT_GREEN'}
KEY = 'tl_signal_patch'


def scene_code(scene_id):
    m = re.search(r'odyssey_scene\d{3}', str(scene_id))
    return m.group(0) if m else None


def parse(raw):
    """Strictly check a patch file. -> (doc, sha256). ValueError on anything malformed."""
    raw = raw.encode('utf-8') if isinstance(raw, str) else bytes(raw)
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise ValueError(f'tl_signal_patch: not JSON ({e})') from None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        raise ValueError(f'tl_signal_patch: schema must be {SCHEMA!r}')
    if not re.fullmatch(r'odyssey_scene\d{3}', str(doc.get('scene', ''))):
        raise ValueError('tl_signal_patch: scene must be a published scene name (odyssey_sceneNNN)')
    signals = doc.get('signals')
    if not isinstance(signals, list):
        raise ValueError('tl_signal_patch: signals must be a list')
    seen = {}
    for i, sig in enumerate(signals):
        where = f'tl_signal_patch: signals[{i}]'
        conns = sig.get('connectors') if isinstance(sig, dict) else None
        if not isinstance(conns, list) or not conns or not all(isinstance(c, str) and c for c in conns):
            raise ValueError(f'{where}.connectors must be a non-empty list of connector id strings')
        dup = sorted({c for c in conns if conns.count(c) > 1})
        if dup:
            raise ValueError(f'{where}.connectors names a connector more than once: {dup}')
        patches = sig.get('patches')
        if not isinstance(patches, list):
            raise ValueError(f'{where}.patches must be a list')
        for j, p in enumerate(patches):
            if (not isinstance(p, list) or len(p) < 3 or not all(type(x) is int for x in p[:2])
                    or not 0 <= p[0] <= p[1] < GT_ROWS or p[2] not in RAW):
                raise ValueError(f'{where}.patches[{j}] must be [first, last, "R"|"G", ...] with '
                                 f'0 <= first <= last < {GT_ROWS}, got {p!r}')
            for c in conns:
                for (a, b, k) in seen.get(c, ()):
                    if p[0] <= b and a <= p[1]:
                        raise ValueError(f'{where}.patches[{j}] overlaps {k} on connector {c}')
                seen.setdefault(c, []).append((p[0], p[1], f'signals[{i}].patches[{j}]'))
    return doc, hashlib.sha256(raw).hexdigest()


def load_for_scene(path, scene_id):
    """-> (doc, sha256, file) for this scene, or None if a directory holds no file for it.

    ``path`` is a patch file (its scene must be this scene, else ValueError) or a directory of <scene>.json."""
    code = scene_code(scene_id)
    p = Path(str(path)).expanduser()
    if p.is_dir():
        f = p / f'{code}.json'
        if not f.is_file():
            return None
    else:
        f = p
    doc, sha = parse(f.read_bytes())
    if doc['scene'] != code:
        raise ValueError(f'tl_signal_patch: {f} is for scene {doc["scene"]}, not {code} ({scene_id})')
    return doc, sha, str(f.resolve())


def apply_to_scene(scene, doc, sha256, path, row_scale=1):
    """Write the patch into ``scene['dynamic_map_states']`` in place and record it in ``scene[KEY]``.

    -> the record. ``row_scale`` = cadence.upsample_n (GT row r -> scene rows r*S .. r*S+S-1)."""
    s = int(row_scale)
    if s < 1:
        raise ValueError('row_scale must be >= 1')
    lights = {}
    for item in (scene.get('dynamic_map_states') or {}).values():
        if item.get('type') == 'TRAFFIC_LIGHT' and item.get('traffic_light_lane') is not None:
            lights.setdefault(str(item['traffic_light_lane']), []).append(item)
    rows, changed, known_overwritten, spans = {}, 0, 0, 0
    past_log, clipped, idm = [], [], {}
    for sig in doc['signals']:
        absent = sorted(c for c in sig['connectors'] if c not in lights)
        if absent:
            raise ValueError(f'tl_signal_patch: connectors with no DB signal in scene {doc["scene"]}: {absent}')
        before = {c: [list(item['state']['traffic_light_state']) for item in lights[c]] for c in sig['connectors']}
        for p in sig['patches']:
            spans += 1
            first, last, colour = int(p[0]) * s, int(p[1]) * s + s - 1, RAW[p[2]]
            for c in sig['connectors']:
                n = max(len(item['state']['traffic_light_state']) for item in lights[c])
                if first > n - 1:
                    past_log.append([c, int(p[0]), int(p[1])])
                elif last > n - 1:
                    clipped.append([c, int(p[0]), int(p[1]), n])
                n = 0
                for item in lights[c]:
                    state = item['state']
                    series = list(state['traffic_light_state'])
                    n = max(n, len(series))
                    for row in range(first, min(last, len(series) - 1) + 1):
                        if series[row] != colour:
                            changed += 1
                            known_overwritten += str(series[row]) in ('TRAFFIC_LIGHT_RED', 'TRAFFIC_LIGHT_GREEN')
                        series[row] = colour
                    state['traffic_light_state'] = series
                if first <= min(last, n - 1):          # rows past the log are not in the DB: nothing to patch
                    rows.setdefault(c, []).append([first, min(last, n - 1)])
        filled = _all_unknown_filled(before, {c: [item['state']['traffic_light_state'] for item in lights[c]]
                                              for c in sig['connectors']})
        if filled[0]:
            idm[str(sig.get('roadblock'))] = dict(rows=filled[0], cells=filled[1])
    for c, a, b in past_log:
        logger.warning('[tl-patch] scene=%s connector=%s span %d-%d lies wholly past the log: nothing patched',
                       doc['scene'], c, a, b)
    for c, a, b, n in clipped:
        logger.warning('[tl-patch] scene=%s connector=%s span %d-%d runs past the log (%d scene rows): clipped',
                       doc['scene'], c, a, b, n)
    if idm:
        logger.warning('[tl-patch] scene=%s IDM all-UNKNOWN fallback now off on %d rows (%d connector cells) where '
                       'the patch filled a wholly UNKNOWN signal: %s', doc['scene'],
                       sum(v['rows'] for v in idm.values()), sum(v['cells'] for v in idm.values()),
                       ', '.join(f'rb {k}: {v["rows"]} rows' for k, v in sorted(idm.items())))
    record = dict(schema=SCHEMA, scene=doc['scene'], sha256=sha256, path=str(path), spans=spans,
                  row_scale=s, rows={c: sorted(v) for c, v in sorted(rows.items())},
                  cells_changed=changed, known_cells_overwritten=known_overwritten,
                  spans_past_log=past_log, spans_clipped=clipped,
                  idm_all_unknown_rows_filled=sum(v['rows'] for v in idm.values()),
                  idm_all_unknown_cells_filled=sum(v['cells'] for v in idm.values()),
                  idm_all_unknown_by_signal=idm)
    scene[KEY] = record
    return record


def _known(value):
    return str(value) in ('TRAFFIC_LIGHT_RED', 'TRAFFIC_LIGHT_GREEN', 'TRAFFIC_LIGHT_YELLOW')


def _all_unknown_filled(before, after):
    """-> (rows, connector cells) where every connector of a signal was UNKNOWN before and at least one is known after.

    before/after = {connector: [series per DB track]}; a connector is known at a row when any of its tracks is."""
    n = max((len(t) for ts in after.values() for t in ts), default=0)
    rows = cells = 0
    for r in range(n):
        was = [any(r < len(t) and _known(t[r]) for t in ts) for ts in before.values()]
        now = [any(r < len(t) and _known(t[r]) for t in ts) for ts in after.values()]
        if not any(was) and any(now):
            rows += 1
            cells += sum(now)
    return rows, cells


def is_patched(scene, connector, row):
    """True when ``row`` of ``connector`` is inside an applied patch span of this scene (False without a patch)."""
    rec = scene.get(KEY) if hasattr(scene, 'get') else None
    if not rec:
        return False
    for a, b in rec['rows'].get(str(connector), ()):
        if a <= int(row) <= b:
            return True
    return False


def apply_configured(global_config, scene_id, scene, row_scale=1):
    """ScenarioManager hook. No-op (returns None, touches nothing) unless ``tl_signal_patch_path`` is set, directly or
    through the ``tlc_timetable_set`` preset for a scene in the set (explicit key wins; see tlc_timetable_set)."""
    preset = tlc_timetable_set.resolve(global_config, scene_id)
    path = tlc_timetable_set.value_for(global_config, 'tl_signal_patch_path', preset)
    if path in (None, ''):
        return None
    found = load_for_scene(path, scene_id)
    if found is None:
        logger.warning('[tl-patch] scene=%s: no patch file under %s; DB timetable used as is', scene_id, path)
        return None
    doc, sha, f = found
    rec = apply_to_scene(scene, doc, sha, f, row_scale)
    logger.info('[tl-patch] scene=%s sha=%s spans=%d cells_changed=%d known_cells_overwritten=%d path=%s',
                doc['scene'], sha, rec['spans'], rec['cells_changed'], rec['known_cells_overwritten'], f)
    return rec
