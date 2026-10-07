"""One-key preset for the traffic-light timetable mode (opt-in ``tlc_timetable_set``, default null).

``tlc_timetable_set`` names a timetable set: a directory name under ``tlc_timetable_sets_dir`` (the benchmark
launcher passes ``tlc_timetable`` and OdysseyBenchmark/data; unset, names are looked up in SETS_DIR) or a
path to a set directory. A set directory holds::

  tlc_timetable_set.json       {"schema": "tlc_timetable_set/1",
                                "scenes": {"odyssey_scene011": {"allow_excluded": true, "allow_unobserved": false}, ...}}
                               (an optional "name"; the default is the directory's name)
  <scene>.json                 the scene's timetable (tl_signal_patch/1, see signal_patch)
  designated.json              the scored signals, read by the offline scorer only (tlc_timetable.load_timetable)

For a scene listed in the set, the preset supplies:
  tl_signal_patch_path          <set>/<scene>.json
  tl_control_pin_uncontrolled   true
  tl_control_allow_excluded     [scene] when the set says allow_excluded (the scene is on the tl_control denylist)
The scene's representative-frame label (tl_control/1) is not part of the set: it is the scene's own published
tl_control.json, given as tl_control_path (the benchmark launcher passes it). An in-set scene without
tl_control_path raises ValueError, so the timetable is never drawn without its label.
A scene not in the set: nothing is supplied (the run is the same as without the preset) and one log line says so.

Precedence: an explicitly set key wins over the preset. tl_signal_patch_path: a non-null value replaces the preset's
file. tl_control_pin_uncontrolled: the preset turns it on; false is the default and cannot be told apart from
"not set", so to run without pinning use the individual keys instead of the preset.
tl_control_allow_excluded: the explicit list and the preset's entry are combined.

Default null: :func:`resolve` returns None and every caller reads its key exactly as before (byte-identical).
Anything wrong with a configured set -- unknown name, missing directory or file, malformed manifest, a label whose
allow_unobserved does not match the manifest -- raises ValueError at scene load.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

KEY = 'tlc_timetable_set'
SCHEMA = 'tlc_timetable_set/1'
MANIFEST = 'tlc_timetable_set.json'
#: Where set names are looked up when ``tlc_timetable_sets_dir`` is unset. The simulator ships no set of its own.
SETS_DIR = Path(__file__).resolve().parents[1] / 'data' / 'tlc_timetable'
DIR_KEY = 'tlc_timetable_sets_dir'
_SCENE_FIELDS = {'allow_excluded', 'allow_unobserved'}


def scene_code(scene_id):
    m = re.search(r'odyssey_scene\d{3}', str(scene_id))
    return m.group(0) if m else None


def configured(global_config):
    """The raw ``tlc_timetable_set`` value, or None when unset (null / empty)."""
    value = global_config.get(KEY) if hasattr(global_config, 'get') else None
    return None if value in (None, '') else str(value)


def sets_dir(global_config):
    """The directory set names are looked up in: ``tlc_timetable_sets_dir`` when set, else SETS_DIR."""
    value = global_config.get(DIR_KEY) if hasattr(global_config, 'get') else None
    return SETS_DIR if value in (None, '') else Path(str(value))


def set_directory(value, base=None):
    """Set name (a directory under ``base``, default SETS_DIR; names are looked up there first) or path -> the set
    directory.

    ValueError when neither exists."""
    base = SETS_DIR if base is None else Path(base)
    text = str(value)
    if re.fullmatch(r'[A-Za-z0-9_.-]+', text) and (base / text).is_dir():
        return (base / text).resolve()
    d = Path(text).expanduser()
    if not d.is_dir():
        known = sorted(p.name for p in base.glob('*') if (p / MANIFEST).is_file()) if base.is_dir() else []
        raise ValueError(f'{KEY}: {text!r} is neither a set name under {base} (known: {known}) '
                         f'nor a set directory')
    return d.resolve()


def load_set(value, base=None):
    """-> {dir, name, sha256, scenes{code: {allow_excluded, allow_unobserved}}}; ValueError if malformed.

    ``base``: where a set name is looked up (default SETS_DIR)."""
    d = set_directory(value, base)
    f = d / MANIFEST
    if not f.is_file():
        raise ValueError(f'{KEY}: {d} has no {MANIFEST}')
    raw = f.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise ValueError(f'{KEY}: {f} is not JSON ({e})') from None
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        raise ValueError(f'{KEY}: {f}: schema must be {SCHEMA!r}')
    scenes = doc.get('scenes')
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError(f'{KEY}: {f}: scenes must be a non-empty {{odyssey_sceneNNN: {{...}}}} map')
    out = {}
    for code, spec in scenes.items():
        if not re.fullmatch(r'odyssey_scene\d{3}', str(code)):
            raise ValueError(f'{KEY}: {f}: scene {code!r} is not a published scene name (odyssey_sceneNNN)')
        if not isinstance(spec, dict) or set(spec) - _SCENE_FIELDS or not all(
                isinstance(spec.get(k, False), bool) for k in _SCENE_FIELDS):
            raise ValueError(f'{KEY}: {f}: scene {code}: only boolean {sorted(_SCENE_FIELDS)} are allowed, got {spec!r}')
        if not (d / f'{code}.json').is_file():
            raise ValueError(f'{KEY}: {f}: scene {code} has no {code}.json')
        out[str(code)] = {k: bool(spec.get(k, False)) for k in sorted(_SCENE_FIELDS)}
    return dict(dir=str(d), name=str(doc.get('name') or d.name),
                sha256=hashlib.sha256(raw).hexdigest(), scenes=out)


def resolve(global_config, scene_id):
    """None when ``tlc_timetable_set`` is unset. Otherwise the set's record for this scene:
    {set, dir, name, sha256, scene, in_set} plus, when in_set, the supplied values and the label
    path (tl_control_path, taken from the config)."""
    value = configured(global_config)
    if value is None:
        return None
    s = load_set(value, sets_dir(global_config))
    code = scene_code(scene_id)
    rec = dict(set=value, dir=s['dir'], name=s['name'], sha256=s['sha256'],
               scene=code, in_set=code in s['scenes'])
    if not rec['in_set']:
        return rec
    spec = s['scenes'][code]
    d = Path(s['dir'])
    label = global_config.get('tl_control_path') if hasattr(global_config, 'get') else None
    if label in (None, ''):
        raise ValueError(f'{KEY}: scene {code} is in set {s["name"]} but tl_control_path is not set: the set '
                         "needs the scene's own tl_control.json label")
    label = str(label)
    heads = (json.loads(Path(label).expanduser().read_text(encoding='utf-8')).get('heads') or {}).values()
    has_unobserved = any(isinstance(h, dict) and h.get('allow_unobserved') is not None for h in heads)
    if has_unobserved != spec['allow_unobserved']:
        raise ValueError(f'{KEY}: {label} {"has" if has_unobserved else "has no"} allow_unobserved heads but the set '
                         f'says allow_unobserved={spec["allow_unobserved"]} for {code}')
    rec.update(tl_signal_patch_path=str(d / f'{code}.json'), tl_control_path=label,
               tl_control_pin_uncontrolled=True,
               tl_control_allow_excluded=[code] if spec['allow_excluded'] else [],
               allow_unobserved=spec['allow_unobserved'])
    return rec


def value_for(global_config, key, preset, default=None):
    """The effective value of one supplied key: the explicit config value, else the preset's (see precedence)."""
    explicit = global_config.get(key, default) if hasattr(global_config, 'get') else default
    if not preset or not preset.get('in_set'):
        return explicit
    if key == 'tl_control_allow_excluded':
        items = [explicit] if isinstance(explicit, str) else list(explicit or ())
        return items + [c for c in preset[key] if c not in items]
    if key == 'tl_control_pin_uncontrolled':
        return bool(explicit) or preset[key]
    return preset[key] if explicit in (None, '') else explicit


def overridden(global_config, preset):
    """Keys whose explicit value replaces the preset's (for the startup log)."""
    if not preset or not preset.get('in_set'):
        return []
    return [k for k in ('tl_signal_patch_path',)
            if global_config.get(k) not in (None, '') and str(global_config.get(k)) != preset[k]]


def log_scene(global_config, scene_id):
    """ScenarioManager hook: one startup line per scene when the preset is set; nothing when it is null."""
    preset = resolve(global_config, scene_id)
    if preset is None:
        return None
    if not preset['in_set']:
        logger.warning('[tlc-set] scene=%s is not in set %s (%s): no-op, DB timetable and natural traffic-light '
                       'replay unless other keys say otherwise', preset['scene'], preset['name'], preset['dir'])
        return preset
    over = overridden(global_config, preset)
    # an explicit key replacing the set's file mixes two sources (e.g. another label with this timetable): warn
    logger.log(logging.WARNING if over else logging.INFO,
               '[tlc-set] scene=%s set=%s sha=%s timetable=%s tl_control=%s pin_uncontrolled=on '
               'allow_excluded=%s allow_unobserved=%s explicit_overrides=%s',
               preset['scene'], preset['name'], preset['sha256'],
               preset['tl_signal_patch_path'], preset['tl_control_path'],
               bool(preset['tl_control_allow_excluded']), preset['allow_unobserved'], over or 'none')
    return preset
