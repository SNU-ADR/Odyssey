"""SD-route conditioning, copied verbatim into each repo under models/ that uses it.

Only torch is imported here; route_target.py, which needs nuPlan, is imported explicitly by the
feature/target builders. Keep all copies byte-identical.
"""
from navsim.agents.sdroute.route_attention import (
    SDRouteDecoder,
    SDRouteDecoderLayer,
    assert_matches_stock,
    SDRouteSegmentEncoder,
    SDRouteCrossAttention,
    ROUTE_NUM_POINTS,
    ROUTE_SEG_LEN,
    ROUTE_NUM_SEGMENTS,
)

__all__ = [
    "SDRouteDecoder",
    "SDRouteDecoderLayer",
    "assert_matches_stock",
    "SDRouteSegmentEncoder",
    "SDRouteCrossAttention",
    "ROUTE_NUM_POINTS",
    "ROUTE_SEG_LEN",
    "ROUTE_NUM_SEGMENTS",
]
