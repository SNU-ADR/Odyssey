"""SD-route builder shared by every model under models/.

Scene -> ego-local (P,5) [x, y, dx, dy, heading] polyline + (P,) mask: the HD routed centerline
(PDM helpers, walked along the GT ego trajectory) re-matched onto the OSM graph in `assets/`, then
resampled at 1 m.

This is the only source of the route tensor. Each repo's navsim/agents/sdroute/route_target.py
loads this directory by file path (every repo ships its own top-level `navsim`, so it cannot be a
normal import), and the PDM helpers it needs are imported from that host repo's navsim.
"""
from .route_centerline import build_route_centerline_target

__all__ = ["build_route_centerline_target"]
