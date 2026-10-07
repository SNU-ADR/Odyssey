"""drop_static_actors: static classes are removed from the SCORED actor frames (the pinned
driving_inputs_json['actors'] stay as observed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from odyssey_benchmark.driving_metrics import STATIC_CLASSES, STATIC_MODE, drop_static_actors  # noqa: E402

NF = 10


def _row(token, kind, x=1.0, y=2.0, h=0.25, ln=0.5, wd=0.5, ht=0.8, vx=0.0, vy=0.0):
    return [token, kind, x, y, h, ln, wd, ht, vx, vy]


def _frames(seen, kind="TRAFFIC_CONE", **pose):
    """One static object is visible in the `seen` frames only; one car moves every frame."""
    out = []
    for i in range(NF):
        actors = [_row("veh", "VEHICLE", x=float(i), y=0.0, h=0.0, ln=5.0, wd=2.0, vx=1.0)]
        if i in seen:
            actors.append(_row("cone", kind, **pose))
        out.append(actors)
    return out


def test_static_actors_are_removed_from_every_frame():
    frames = _frames({3, 4})
    out, info = drop_static_actors(frames)
    assert info == {"mode": STATIC_MODE, "dropped": 1, "by_class": {"TRAFFIC_CONE": 1}}
    assert not any(a[1] == "TRAFFIC_CONE" for actors in out for a in actors)
    # non-static rows are all kept
    assert (sum(len(x) for x in out)
            == sum(1 for actors in frames for a in actors if a[1] != "TRAFFIC_CONE"))
    # without a static actor the same object comes back (idempotent)
    again, info2 = drop_static_actors(out)
    assert again is out and info2["dropped"] == 0


def test_every_static_class_is_dropped_and_moving_classes_are_not():
    frames = [[_row("cone", "TRAFFIC_CONE"), _row("junk", "GENERIC_OBJECT", x=4.0),
               _row("bar", "BARRIER"), _row("sign", "CZONE_SIGN"), _row("ped", "PEDESTRIAN")]]
    out, info = drop_static_actors(frames)
    assert set(info["by_class"]) == set(STATIC_CLASSES) and info["dropped"] == 4
    assert [a[0] for a in out[0]] == ["ped"]
