"""
Safety metrics following the nuPlan closed-loop benchmark (the standard in recent
closed-loop planning work), re-implemented for MetaDrive state.

Source of every constant: nuplan-devkit metric docs and code, checked 2026-09-22
  https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html
  nuplan/planning/metrics/evaluation_metrics/common/time_to_collision_within_bound.py
  Karnchanachari et al., "Towards learning-based planning: the nuPlan benchmark for
  real-world autonomous driving", ICRA 2024 (arXiv:2403.04133).

  time_to_collision_within_bound : least_min_ttc = 0.95 s, time_horizon = 3.0 s,
                                   time_step_size = 0.1 s; not computed while ego is
                                   stopped (speed <= 5e-3 m/s); tracks behind ego ignored.
  no_ego_at_fault_collisions     : at-fault = STOPPED_TRACK, ACTIVE_FRONT, and
                                   ACTIVE_LATERAL when ego is not fully in one lane.
                                   Not at-fault = STOPPED_EGO, ACTIVE_REAR.

Where this departs from nuPlan (documented, not hidden):
  * nuPlan keeps tracks behind ego inside intersections / between lanes; MetaDrive gives
    no cheap intersection flag, so behind-ego tracks are always dropped.
  * "ego not fully in one lane" is approximated by |lateral offset from the ego's route|
    > (LANE_WIDTH - ego width) / 2 with a nominal LANE_WIDTH.
  * MetaDrive reports THAT a contact happened, not WITH WHOM; the partner is taken to be
    the agent whose box is closest to ego's box at the contact step.
"""
from __future__ import annotations

import numpy as np

LEAST_MIN_TTC = 0.95
TTC_HORIZON = 3.0
TTC_STEP = 0.1
STOPPED_SPEED = 5e-3
LANE_WIDTH = 3.5

# collision types (nuPlan names)
STOPPED_EGO = "stopped_ego"
STOPPED_TRACK = "stopped_track"
ACTIVE_FRONT = "active_front"
ACTIVE_REAR = "active_rear"
ACTIVE_LATERAL = "active_lateral"


# ------------------------------------------------------------------------ geometry
def box_corners(x, y, heading, length, width):
    """Corners of oriented boxes. Inputs broadcast; returns (..., 4, 2)."""
    x, y, h, L, W = np.broadcast_arrays(*[np.asarray(a, float) for a in (x, y, heading, length, width)])
    c, s = np.cos(h), np.sin(h)
    dl = np.stack([L, L, -L, -L], -1) / 2
    dw = np.stack([W, -W, -W, W], -1) / 2
    cx = x[..., None] + dl * c[..., None] - dw * s[..., None]
    cy = y[..., None] + dl * s[..., None] + dw * c[..., None]
    return np.stack([cx, cy], -1)


def boxes_overlap(a, b):
    """Separating-axis test between corner sets a, b of shape (..., 4, 2). Returns bool (...)."""
    def axes(p):
        e = p[..., [1, 2], :] - p[..., [0, 1], :]              # two edge directions
        return np.stack([-e[..., 1], e[..., 0]], -1)            # their normals, (...,2,2)

    ax = np.concatenate([axes(a), axes(b)], -2)                 # (...,4,2)
    pa = np.einsum("...ck,...ak->...ac", a, ax)                 # (...,4 axes,4 corners)
    pb = np.einsum("...ck,...ak->...ac", b, ax)
    separated = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~separated.any(-1)


def box_gap(ego, agents):
    """Approximate clearance between ego box and each agent box (centre distance minus
    half-extents along the joining line). ego: dict; agents: (A, 6) [x,y,h,v,L,W]."""
    d = agents[:, :2] - np.array([ego["x"], ego["y"]])
    dist = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dist[:, None], 1e-9)

    def half_extent(h, L, W, u):
        c, s = np.cos(h), np.sin(h)
        return 0.5 * (L * np.abs(u[:, 0] * c + u[:, 1] * s) + W * np.abs(-u[:, 0] * s + u[:, 1] * c))

    return dist - half_extent(ego["h"], ego["L"], ego["W"], u) - half_extent(agents[:, 2], agents[:, 4], agents[:, 5], u)


def ego_frame(ego, agents):
    """Agent centres in ego frame: (dx along heading, dy to the left)."""
    d = agents[:, :2] - np.array([ego["x"], ego["y"]])
    c, s = np.cos(ego["h"]), np.sin(ego["h"])
    return d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c


# ------------------------------------------------------------------------- metrics
def ttc_nuplan(ego, agents) -> float:
    """
    nuPlan time_to_collision_within_bound, one timestamp. Project ego and every track ahead
    of ego forward at constant speed and heading; return the first t in (0, 3.0] s at
    which the boxes overlap, or inf. Ego stopped -> inf (nuPlan returns None: no check).
    """
    if ego["v"] <= STOPPED_SPEED or len(agents) == 0:
        return float("inf")
    dx, _ = ego_frame(ego, agents)
    ahead = agents[dx > 0]
    if len(ahead) == 0:
        return float("inf")
    ts = np.arange(1, int(round(TTC_HORIZON / TTC_STEP)) + 1) * TTC_STEP          # (T,)
    ex = ego["x"] + ego["v"] * np.cos(ego["h"]) * ts
    ey = ego["y"] + ego["v"] * np.sin(ego["h"]) * ts
    ec = box_corners(ex, ey, ego["h"], ego["L"], ego["W"])                          # (T,4,2)
    ax = ahead[:, 0, None] + ahead[:, 3, None] * np.cos(ahead[:, 2, None]) * ts     # (A,T)
    ay = ahead[:, 1, None] + ahead[:, 3, None] * np.sin(ahead[:, 2, None]) * ts
    ac = box_corners(ax, ay, ahead[:, 2, None], ahead[:, 4, None], ahead[:, 5, None])  # (A,T,4,2)
    hit = boxes_overlap(np.broadcast_to(ec, ac.shape), ac)                          # (A,T)
    any_t = hit.any(0)
    return float(ts[np.argmax(any_t)]) if any_t.any() else float("inf")


def classify_collision(ego, agents, ego_lateral: float) -> tuple[str, bool, int]:
    """
    Type and at-fault flag of a contact, nuPlan rules. The partner is the agent with the
    smallest box gap. Returns (type, at_fault, partner_index); partner -1 if no agents
    (contact with a static object: at-fault iff ego is moving).
    """
    if len(agents) == 0:
        moving = ego["v"] > STOPPED_SPEED
        return (ACTIVE_FRONT if moving else STOPPED_EGO), moving, -1
    k = int(np.argmin(box_gap(ego, agents)))
    if ego["v"] <= STOPPED_SPEED:
        return STOPPED_EGO, False, k
    if agents[k, 3] <= STOPPED_SPEED:
        return STOPPED_TRACK, True, k
    dx, dy = ego_frame(ego, agents[k:k + 1])
    half_w = 0.5 * (ego["W"] + agents[k, 5])
    if abs(dy[0]) < half_w:
        return (ACTIVE_FRONT, True, k) if dx[0] > 0 else (ACTIVE_REAR, False, k)
    out_of_lane = abs(ego_lateral) > 0.5 * (LANE_WIDTH - ego["W"])
    return ACTIVE_LATERAL, bool(out_of_lane), k
