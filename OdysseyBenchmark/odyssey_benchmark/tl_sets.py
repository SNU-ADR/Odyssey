"""The benchmark's traffic-light timetable: OdysseyBenchmark/data/tlc_timetable/.

The launcher drives the scenes it lists with its timetable (``tlc_timetable_set=tlc_timetable``, looked
up under ``tlc_timetable_sets_dir`` = DATA_DIR) and pins NAME as the run's ``tl_set`` rule; the scorer
resolves a pinned name here.
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
TIMETABLE_DIR = DATA_DIR / 'tlc_timetable'
#: The name pinned with every run: the directory's name (the manifest carries no name of its own).
NAME = TIMETABLE_DIR.name


def timetable_dir(name):
    """A ``tl_set`` name -> the timetable directory. ValueError for any other name."""
    if name != NAME:
        raise ValueError(f'unknown traffic-light timetable {name!r}: the benchmark has one, {NAME!r}')
    return TIMETABLE_DIR
