"""Per-scene simulator inputs from the published scene release.

ODYSSEY_SCENES_ROOT is the ``ADRLAB/odyssey-scenes`` download as it is: ``scenes.csv`` plus one
``odyssey_sceneNNN/`` folder per scene. A scene is named as published (``odyssey_scene041``,
``scene041``, ``041`` or ``41``). Every file is passed to the simulator by its own path, so the
download is only read, never written.
"""
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import re

RELEASE_COLUMNS = ('scene', 'token')
RELEASE_FILES = ('scene.ckpt', 'scenario.pkl', 'road_surface.npz', 'road_surface.json', 'route.npz',
                 'tl_control.json', 'checkpoint_inventory.json')


@dataclass(frozen=True)
class SceneInputs:
    name: str            # the published name, odyssey_sceneNNN
    folder: Path         # <ODYSSEY_SCENES_ROOT>/<name>
    token: str           # the scene's first nuPlan lidar token (scenes.csv and scenario.pkl agree)
    scenario_key: str    # the key the scenario is stored under in scenario.pkl (= name)
    metadata: dict       # the scenario's metadata block

    @property
    def scenario(self):
        return self.folder / 'scenario.pkl'

    @property
    def checkpoint(self):            # the scene's Gaussian reconstruction
        return self.folder / 'scene.ckpt'

    @property
    def road_surface(self):          # npz; the renderer also reads its .json sibling (drop_static)
        return self.folder / 'road_surface.npz'

    @property
    def route(self):                 # the SD route the planner reads and the scorer matches against
        return self.folder / 'route.npz'

    @property
    def tl_control(self):            # traffic-light labels
        return self.folder / 'tl_control.json'

    @property
    def inventory(self):             # the reconstruction's node inventory (traffic-light nodes)
        return self.folder / 'checkpoint_inventory.json'


def _load_scenario(path):
    with Path(path).open('rb') as handle:
        scenarios = pickle.load(handle)
    if len(scenarios) != 1:
        raise ValueError(f'{path}: one scene per exported scenario file is required')
    key, scenario = next(iter(scenarios.items()))
    metadata = scenario['metadata']
    token = metadata['nuplan_lidar_pc_tokens'][0]
    if not isinstance(token, str):
        raise ValueError(f'{path}: scenario token must be a string')
    return str(key), metadata, token


def release_table(scenes_root):
    path = Path(scenes_root) / 'scenes.csv'
    if not path.is_file():
        raise FileNotFoundError(f'{path}: not an odyssey-scenes download (scenes.csv missing)')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    missing = [c for c in RELEASE_COLUMNS if not rows or c not in rows[0]]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}; download the current release')
    return rows


def published_name(scene):
    """odyssey_scene041 / scene041 / 041 / 41 -> 'odyssey_scene041'; None for anything else."""
    number = re.fullmatch(r'(?:odyssey_)?(?:scene)?0*(\d{1,3})', str(scene).strip())
    return f'odyssey_scene{int(number.group(1)):03d}' if number else None


def find_release_row(rows, scene):
    """The scenes.csv row of a published scene name."""
    wanted = published_name(scene)
    hits = [r for r in rows if wanted and r['scene'] == wanted]
    if len(hits) != 1:
        raise ValueError(f'unknown scene {scene!r}: use a published name listed in scenes.csv '
                         '(e.g. odyssey_scene001, scene001 or 001)')
    return hits[0]


def from_release(scenes_root, scene):
    root = Path(os.path.abspath(scenes_root))
    row = find_release_row(release_table(root), scene)
    folder = root / row['scene']
    missing = [f for f in RELEASE_FILES if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError(f'{folder}: missing {missing}; download this scene completely '
                                '(the files are listed in MD5SUMS)')
    key, metadata, token = _load_scenario(folder / 'scenario.pkl')
    if key != row['scene']:
        raise ValueError(f'{folder}: scenario.pkl holds scene {key!r}, not {row["scene"]!r}; download the '
                         'current ADRLAB/odyssey-scenes release')
    if token != row['token']:
        raise ValueError(f'{folder}: scenario.pkl is scene token {token}, but scenes.csv lists {row["token"]}')
    return SceneInputs(name=row['scene'], folder=folder, token=token, scenario_key=key, metadata=metadata)


def resolve(scene, environ=None):
    scenes_root = (os.environ if environ is None else environ).get('ODYSSEY_SCENES_ROOT')
    if not scenes_root:
        raise ValueError('ODYSSEY_SCENES_ROOT is not set: point it at the ADRLAB/odyssey-scenes download')
    return from_release(scenes_root, scene)
