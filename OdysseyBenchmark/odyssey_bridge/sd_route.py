"""Hand-drawn SD edges that the scorer's SD graph overlays (OdysseyBenchmark/data/sd_roadblock_map).

The OSM road graph misses a few road shapes a logged drive takes (a turnaround loop, a detour, a
driveway). Each is drawn once, between real SD graph nodes, as an edge with a fixed negative id that
a route's sd_edges can name; `replaces` lists the SD edges those pieces cover.
odyssey_benchmark.sdroute_sdf.graph(map_location, include_manual=True) applies both.
"""
from __future__ import annotations

import json
import os

import numpy as np

#: The hand-drawn edges travel with the code.
SDMAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sd_roadblock_map")


def manual_edges(map_location, sdmap_dir=SDMAP):
    """Hand-drawn edges. id (negative) -> {u, v, xy, name}. Empty dict for maps without a file.

    Represents a road shape missing from the SD graph (e.g. a turnaround loop) as one edge. Nodes
    u/v are real SD graph nodes, so it connects directly to the neighbouring SD edges. The id is
    taken from the file and never reassigned -- that integer is stored in the sidecar's sd_edges.
    """
    p = os.path.join(sdmap_dir, f"sd_manual_edges_{map_location}.json")
    if not os.path.isfile(p):
        return {}
    out = {}
    for e in json.load(open(p))["edges"]:
        i = int(e["id"])
        if i >= 0 or i in out:
            raise ValueError(f"{p}: manual edge ids must be unique negative integers (got {i})")
        out[i] = dict(u=int(e["u"]), v=int(e["v"]), name=e["name"],
                      xy=np.asarray(e["xy"], dtype=np.float64))
    return out


def replaced_sd_edges(map_location, sdmap_dir=SDMAP):
    """SD edge ids fully split into manual pieces between nodes; the graph uses the pieces instead.

    If an original edge and its pieces overlap on the same line, nodes shift depending on which
    one gets picked. The list in the file is used as-is -- coverage is not recomputed here.
    """
    p = os.path.join(sdmap_dir, f"sd_manual_edges_{map_location}.json")
    if not os.path.isfile(p):
        return set()
    return {int(i) for i in json.load(open(p))["replaces"]}
