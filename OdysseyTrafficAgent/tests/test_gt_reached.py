"""Regression tests for the destination_arrival check and its reference line (frenet).

This logic was silently wrong four times in a row -- densify overshooting its grid, the
stationary tail at the end of the GT, point-to-point projection, and a frame mismatch
(center vs rear_axle). Each changed scores without raising, so each is pinned here.
"""
import numpy as np
import pytest

from odyssey.manager import frenet


def test_arclen_keeps_polyline_and_length():
    """No resampling -- point count and length match the input."""
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    ref, ref_s = frenet.arclen(xy)
    assert len(ref) == 3
    assert ref_s[-1] == pytest.approx(10.0)


def test_arclen_trims_stationary_tail():
    """Millimetre jitter after the ego stops is not part of the reference line (48 of 219 scenes)."""
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0],
                   [10.001, 0.0], [10.002, 0.001]])
    ref, ref_s = frenet.arclen(xy)
    assert len(ref) == 3
    assert ref_s[-1] == pytest.approx(10.0)


def test_arclen_keeps_leading_stationary():
    """Waiting before departure is the real start of the reference line, so it is kept."""
    xy = np.array([[0.0, 0.0], [0.001, 0.0], [5.0, 0.0], [10.0, 0.0]])
    ref, ref_s = frenet.arclen(xy)
    assert len(ref) == 4
    assert ref_s[-1] == pytest.approx(10.0)


def test_segment_project_is_continuous_between_vertices():
    """Point-to-segment, so s is not stuck at vertex values (the point-to-point projection bug)."""
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    ref_s = np.array([0.0, 10.0])
    s, d, _ = frenet._segment_project(np.array([3.7, 2.0]), ref, ref_s, 0, 2)
    assert s == pytest.approx(3.7)
    assert abs(d) == pytest.approx(2.0)
