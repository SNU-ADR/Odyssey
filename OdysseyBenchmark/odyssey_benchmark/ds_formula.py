"""RouteDS, the benchmark's episode score (paper Appendix C.3, eq. 2). The one definition.

    RouteDS = 100 * RC * P_SD * P_col * P_off * P_TL * P_PLC

RC is the SD-route completion in [0, 1]; the five P_* factors are penalties in [0, 1]. The
scorer (``odyssey_benchmark.driving_metrics``) fills the terms; this module only multiplies them,
so every consumer scores a row the same way. Standard library only.
"""
from __future__ import annotations
import math

#: Row keys, in the order the formula multiplies them.
TERMS = ('RC', 'P_SD', 'P_col', 'P_off', 'P_TL', 'P_PLC')
#: Rule-owned factors: absent means "that rule was not applied", scored as 1.0. A present but
#: invalid value is never silently accepted.
RULE_TERMS = ('P_TL', 'P_PLC')


def route_ds(row):
    """RouteDS from a scored row, or None when a term is missing or not a valid factor.

    An unscored row stays unscored; nothing here fabricates a pass.
    """
    values = []
    for key in TERMS:
        raw = row.get(key)
        if raw in (None, '') and key in RULE_TERMS:
            values.append(1.0)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    if values[1] not in (0., 1.):                    # P_SD is a verdict, never a fraction
        return None
    if any(not 0. <= v <= 1. for v in values[2:]):
        return None
    return compose(*values)


def compose(rc, p_sd, p_col, p_off, p_tl, p_plc) -> float:
    """RouteDS = 100 * RC * P_SD * P_col * P_off * P_TL * P_PLC, floored at 0."""
    return max(float(rc) * float(p_sd) * float(p_col) * float(p_off)
               * float(p_tl) * float(p_plc), 0.0) * 100.0
