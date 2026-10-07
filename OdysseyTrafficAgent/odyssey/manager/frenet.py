"""Frenet projection onto a reference curve -- separates "how far it got" from "how much it moved".

Why it exists
    v1 route_completion is  driven_arclen / gt_arclen , the ratio of the lengths of two
    independent curves, so it has no notion of direction. A run that went 100 m sideways scores
    the same as one that went 100 m along the route, and circling in place still grows the
    numerator.

    Projecting the ego onto a reference curve as (s, d) separates the two.

        s   progress along the reference curve      (longitudinal, arc length)
        d   deviation from the reference curve     (lateral, signed)

    Progress is s_end - s_start; sideways motion goes into d and does not mix in.

    Kept as pure functions so they can be called during scoring. No dependency beyond numpy --
    no simulator objects, no file I/O.

What is the reference curve
    The caller decides. metric_manager passes the GT (expert) trajectory, so d is "deviation
    from the expert line", not "deviation from the lane centre". That is the right reference
    for "did it make progress", not for "did it keep its lane" -- for the latter, pass the lane
    centreline.
"""
from __future__ import annotations


import numpy as np
import numpy.typing as npt

WIN_FWD_M = 60.0     # [m] look-ahead range from the previous match
MIN_REF_LENGTH_M = 0.5   # a reference curve shorter than this cannot be a denominator
STATIONARY_M = 0.05  # [m] segments shorter than this count as 'not moving' -- arclen tail cut


def arclen(xy: npt.NDArray[np.float64]):
    """Cumulative arc length of the polyline as given -> (M,2) coordinates, (M,) arc length.

    Same preprocessing as densify (trim the stationary tail at the end) but **no resampling.**
    In point-to-segment projection (_segment_project) s is continuous along each segment, so a
    finer grid adds no accuracy. Measured on 65 runs: without densify the reference length
    differed by 0.000000 m, the trigger frame was identical in 64/65 runs, and the point count
    dropped 6.5x.

    The return shape matches densify so callers can use it unchanged.
    """
    xy = np.asarray(xy, dtype=np.float64)

    # Trim the stationary tail at the end -- see the matching block comment in densify.
    if len(xy) > 2:
        _seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        _moved = np.nonzero(_seg > STATIONARY_M)[0]
        if len(_moved):
            xy = xy[:_moved[-1] + 2]

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return xy, np.concatenate([[0.0], np.cumsum(seg)])


def _segment_project(p, ref, ref_s, lo, hi):
    """Project point p onto the **segments** of ref[lo:hi] -> (s, d, k).

    Point-to-segment, not nearest point. The difference shows between grid points:

      * Point-to-point can only return grid values of s, so on a short reference the last cell
        exceeds the threshold. With DENSIFY_M=0.5 and a 19.5 m GT, one cell is 2.56%, so unless
        the ego landed exactly on the last grid point, progress was stuck at 0.9744 and never
        crossed 0.99 (8 of 219 scenes measured).
      * Point-to-point has no notion of "passing". If the ego brushes the end point and turns
        away, an earlier grid point becomes closer, the cursor locks there, and progress stayed
        flat while the ego drove another 122 m.

    Using the segment parameter t makes s continuous, which removes both. t saturating at 1 on
    the last segment means "passed the end", and s naturally becomes ref_s[-1].

    :return: (arc length, signed lateral offset -- positive to the left of the tangent,
             grid index of the segment start)
    """
    a = ref[lo:hi - 1]
    b = ref[lo + 1:hi]
    ab = b - a
    den = np.einsum('ij,ij->i', ab, ab)
    den[den == 0.0] = 1.0
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    dist = np.linalg.norm(proj - p, axis=1)
    # On ties pick the **furthest-along** segment. np.argmin picks the first, which can trap the
    # cursor for good where the end of the reference is compressed by braking: the segments just
    # before GT stops shrink to centimetres (measured 0.402 / 0.026 / 0.099 m), so from a distant
    # ego they are effectively one point and distances differ by only 0.01 m. If the earlier
    # segment wins there, t saturates at 1 and s freezes; the ego drove 125 m past the end while
    # progress stayed at 0.9934. Picking the later one only advances s at equal distance, so
    # monotonicity is preserved.
    j = int(len(dist) - 1 - np.argmin(dist[::-1]))
    k = lo + j
    seg_len = float(np.linalg.norm(ab[j]))
    s = float(ref_s[k] + t[j] * seg_len)
    tan = ab[j] / max(seg_len, 1e-9)
    v = p - proj[j]
    d = float(-tan[1] * v[0] + tan[0] * v[1])
    return s, d, k
