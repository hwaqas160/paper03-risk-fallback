"""Geometry + rule checks for src/nuplan_metrics.py (no simulator)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nuplan_metrics as nm  # noqa: E402


def ego(x=0.0, y=0.0, h=0.0, v=10.0):
    return dict(x=x, y=y, h=h, v=v, L=4.5, W=1.9)


def agent(x, y, h=0.0, v=0.0, L=4.5, W=1.9):
    return np.array([[x, y, h, v, L, W]], float)


def test_overlap_basic():
    a = nm.box_corners(0, 0, 0, 4, 2)
    assert nm.boxes_overlap(a, nm.box_corners(3, 0, 0, 4, 2))
    assert not nm.boxes_overlap(a, nm.box_corners(5, 0, 0, 4, 2))
    assert nm.boxes_overlap(a, nm.box_corners(0, 1.5, np.pi / 4, 4, 2))


def test_ttc_rear_end_on_stopped_car():
    # gap 20 m - 4.5 m of bumpers = 15.5 m at 10 m/s -> overlap first at 1.6 s
    t = nm.ttc_nuplan(ego(v=10), agent(20, 0))
    assert abs(t - 1.6) < 1e-9, t


def test_ttc_ignores_track_behind_and_stopped_ego():
    assert nm.ttc_nuplan(ego(v=10), agent(-8, 0, v=20)) == float("inf")
    assert nm.ttc_nuplan(ego(v=0.0), agent(6, 0)) == float("inf")


def test_ttc_adjacent_lane_passes():
    assert nm.ttc_nuplan(ego(v=10), agent(10, 3.5, v=0)) == float("inf")


def test_ttc_beyond_horizon_is_inf():
    assert nm.ttc_nuplan(ego(v=10), agent(50, 0)) == float("inf")


def test_collision_rules():
    # moving ego hits stopped car ahead -> stopped_track, at fault
    assert nm.classify_collision(ego(v=5), agent(4.4, 0), 0.0)[:2] == (nm.STOPPED_TRACK, True)
    # moving ego rear-ends moving car -> active_front, at fault
    assert nm.classify_collision(ego(v=8), agent(4.4, 0, v=3), 0.0)[:2] == (nm.ACTIVE_FRONT, True)
    # moving ego (slower) gets rear-ended -> active_rear, NOT at fault
    assert nm.classify_collision(ego(v=2), agent(-4.4, 0, v=8), 0.0)[:2] == (nm.ACTIVE_REAR, False)
    # stopped ego is hit (the freezing cost) -> stopped_ego, NOT at fault
    assert nm.classify_collision(ego(v=0.0), agent(-4.4, 0, v=8), 0.0)[:2] == (nm.STOPPED_EGO, False)
    # side swipe: at fault only if ego is out of its lane
    assert nm.classify_collision(ego(v=5), agent(0, 1.9, v=5), 0.0)[:2] == (nm.ACTIVE_LATERAL, False)
    assert nm.classify_collision(ego(v=5), agent(0, 1.9, v=5), 1.5)[:2] == (nm.ACTIVE_LATERAL, True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
