"""
Closed-loop rollout harness: run one scenario under a trigger threshold lam and record the
outcome metric set, plus the lam = inf counterfactual that labels "unnecessary stop".

Everything is predictor-agnostic: a Predictor returns K predicted modes per nearby agent.
The stub here is a constant-velocity predictor with configurable spread, so the whole
pipeline (triggers -> rollouts -> losses -> src/risk_control.py) can be built and tested
before Paper 01's AutoBot checkpoint exists. Swapping in AutoBot means replacing the
Predictor only.

    python src/rollout.py --db av2_test --seeds 0 5 --lams 0.0 1.0 2.0 inf
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict

import numpy as np

import simenv

# ---------------------------------------------------------------------------- constants
DT = 0.1                    # sim step, s (ScenarioEnv runs at 10 Hz)
PRED_HORIZON = 30           # prediction horizon, steps (3.0 s)
SAFETY_MARGIN = 1.0         # m, added to the vehicle half-extents when testing intrusion
STOPPED_SPEED = 0.5         # m/s, 'stopped' for the stopped-duration metric (nuPlan's 5e-3 is used for TTC/fault)


# ------------------------------------------------------------------------------ policies
def make_fallback_policy_cls(mrm_decel: float = 4.0):
    """
    TrajectoryIDMPolicy + a minimum-risk manoeuvre. Built lazily so the module can be
    imported before MetaDrive (block_torch must run first in sim workers).

    The manoeuvre is the regulatory minimum-risk manoeuvre of UN R157 (ALKS) §5.5.1: slow
    down INSIDE the lane with a deceleration demand not greater than 4.0 m/s². Lane-keeping
    PID steering is kept; longitudinally, a P-controller tracks the speed profile
    v(t) = max(v0 - MRM_DECEL * t, 0) and holds the brake once stopped.

    History: the first version commanded a fixed brake action of 0.5, which MetaDrive turns
    into ~10 m/s² (≈1 g) — an emergency stop — and produced rear-end collisions that were an
    artifact of that choice, not of stopping per se (results/sweep_dev_nuplan.*, 2026-09-22).
    """
    from metadrive.policy.idm_policy import TrajectoryIDMPolicy

    class FallbackIDMPolicy(TrajectoryIDMPolicy):
        MRM_DECEL = mrm_decel   # m/s²; 4.0 = UN R157 §5.5.1 upper bound (headline), 2.0 = comfort variant
        KP = 0.4            # action per (m/s) of speed error
        HOLD = -0.3         # brake action once stopped
        DT = 0.1

        def act(self, *args, **kwargs):
            if not getattr(self, "fallback_engaged", False):
                self._mrm_t = None
                return super().act(*args, **kwargs)
            v = self.control_object.speed
            if getattr(self, "_mrm_t", None) is None:
                self._mrm_t, self._mrm_v0 = 0.0, v
            self._mrm_t += self.DT
            v_des = max(self._mrm_v0 - self.MRM_DECEL * self._mrm_t, 0.0)
            acc = self.HOLD if v_des == 0.0 and v < 0.5 else float(np.clip(self.KP * (v_des - v), -1.0, 0.0))
            steering = self.steering_control(self.routing_target_lane)
            action = [steering, acc]
            self.last_action = action
            self.action_info["action"] = action
            return action

    return FallbackIDMPolicy


# ---------------------------------------------------------------------------- predictors
class ConstantVelocityPredictor:
    """
    Stub predictor: K modes per agent, all constant-velocity, fanned by `spread` (m at the
    horizon) and jittered by `noise`. `bias` shrinks the fan without changing the mean --
    a crude 'overconfident predictor' knob for rehearsing H3 before real checkpoints land.
    """

    def __init__(self, k: int = 6, spread: float = 3.0, noise: float = 0.3,
                 bias: float = 1.0, seed: int = 0):
        self.k, self.spread, self.noise, self.bias = k, spread, noise, bias
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, scenario_seed: int):
        """
        Common random numbers: every rollout of the same scenario sees the same noise
        stream, whatever lam it runs under. Without this, rollouts at different lam draw
        different noise and the trigger stops being monotone in lam (seen: autonomy fell
        from 0.714 to 0.571 as lam rose).
        """
        self.rng = np.random.default_rng((self.base_seed, scenario_seed))

    def predict(self, states: np.ndarray) -> np.ndarray:
        """states (A, 4) = [x, y, vx, vy]  ->  predictions (A, K, T, 2)."""
        a = len(states)
        if a == 0:
            return np.zeros((0, self.k, PRED_HORIZON, 2))
        t = np.arange(1, PRED_HORIZON + 1) * DT                      # (T,)
        base = states[:, None, :2] + states[:, None, 2:4] * t[None, :, None]   # (A,T,2)
        head = np.arctan2(states[:, 3], states[:, 2])                # (A,)
        lat = np.stack([-np.sin(head), np.cos(head)], -1)            # (A,2) unit lateral
        fan = np.linspace(-1, 1, self.k) * self.spread * self.bias   # (K,)
        ramp = (t / t[-1]) ** 2                                      # uncertainty grows
        out = (base[:, None] +
               fan[None, :, None, None] * ramp[None, None, :, None] * lat[:, None, None, :])
        return out + self.rng.normal(0, self.noise, out.shape)


# ------------------------------------------------------------------------------- scoring
LON_MARGIN = 1.0   # m, added to the length-based longitudinal half-extent (following distance)
LAT_MARGIN = 0.5   # m, added to the width-based lateral half-extent (passing clearance)


def ego_plan_route(env, steps: int = PRED_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Where the route says the ego will be over the horizon (follows the reference path's
    curvature at current speed). NOT used by trigger_score as of 2026-09-22 — see ego_plan_kinematic."""
    traj = env.agent.navigation.reference_trajectory
    long, _ = traj.local_coordinates(env.agent.position)
    v = max(env.agent.speed, 1.0)
    longs = [min(long + v * DT * (i + 1), traj.length) for i in range(steps)]
    pos = np.array([traj.position(s, 0) for s in longs])
    head = np.array([traj.heading_theta_at(s) for s in longs])
    return pos, head


def ego_plan_kinematic(env, steps: int = PRED_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the ego will be if it keeps its CURRENT heading and speed — no route curvature.
    Matches nuplan_metrics.ttc_nuplan's ego extrapolation exactly (both nuPlan's own
    definition and this codebase's harm label use this model, not route-following).

    This is the corridor trigger_score uses (since 2026-09-22). Using ego_plan_route instead
    made the trigger MISS real near-misses: on a curve, the route corridor diverged from this
    kinematic line by up to ~8 m at a 1 s horizon and ~30 m at 3 s (seeds 60/74, dev split) —
    the trigger was effectively "trusting" that steering would avoid a conflict that the
    kinematic TTC metric (deliberately route-agnostic, as a raw safety check) flags as risky.
    Scoring against the corridor the harm metric can't see into defeats the trigger's purpose.
    """
    p0 = np.asarray(env.agent.position, float)
    v, h = env.agent.speed, env.agent.heading_theta
    ts = np.arange(1, steps + 1) * DT
    pos = p0[None, :] + v * np.array([np.cos(h), np.sin(h)])[None, :] * ts[:, None]
    return pos, np.full(steps, h)


ego_plan = ego_plan_kinematic


def predict_agents(env, predictor, other_states: np.ndarray, extents: np.ndarray):
    """
    Run the predictor once per decision tick and return (preds, probs, extents) in one
    common shape — (A,K,T,2), (A,K), (A,2) — regardless of predictor type, so every trigger
    score is computed from a SINGLE predictor call. Splitting this out of trigger scoring
    (2026-09-23) is what makes T1 (confidence) and T3/T4 (geometric) cheap to compute
    together: before, each scorer called the predictor itself, and AutoBot inference is the
    dominant cost of a rollout (Section 9, notes/design.md).

    The constant-velocity stub has no real notion of confidence: probs is uniform over its
    K modes. T1 (score_confidence) is therefore only meaningful with a trained predictor —
    this is disclosed, not hidden, since it directly limits what T1's baseline can show
    when run against the stub.
    """
    if hasattr(predictor, "predict_env"):                    # AutoBot: needs sim history
        preds, probs, extents = predictor.predict_env(env)
        return preds[:, :, :PRED_HORIZON], probs, extents
    if len(other_states) == 0:
        return np.zeros((0, 1, PRED_HORIZON, 2)), np.zeros((0, 1)), np.zeros((0, 2))
    preds = predictor.predict(other_states)                  # (A,K,T,2)
    probs = np.full(preds.shape[:2], 1.0 / preds.shape[1])   # uniform: stub has no confidence
    return preds, probs, extents


def _intrusion(env, preds: np.ndarray, extents: np.ndarray, plan: np.ndarray, head: np.ndarray) -> float:
    """Path-relative box intrusion of predicted agents into an ego corridor (see score_geometric)."""
    if len(preds) == 0:
        return 0.0
    tangent = np.stack([np.cos(head), np.sin(head)], -1)
    normal = np.stack([-np.sin(head), np.cos(head)], -1)
    d = preds - plan[None, None, :, :]
    along = np.abs(np.einsum("aktd,td->akt", d, tangent))
    lat = np.abs(np.einsum("aktd,td->akt", d, normal))
    lon_margin = 0.5 * (env.agent.LENGTH + extents[:, 0]) + LON_MARGIN
    lat_margin = 0.5 * (env.agent.WIDTH + extents[:, 1]) + LAT_MARGIN
    r = np.sqrt((along / lon_margin[:, None, None]) ** 2 + (lat / lat_margin[:, None, None]) ** 2)
    return float(np.max(np.clip(1.0 - r, 0.0, 1.0) * lat_margin[:, None, None]))


def score_geometric_isotropic(env, preds: np.ndarray, extents: np.ndarray) -> float:
    """
    ABLATION ONLY ("geom_iso"): the pre-2026-09-22 scorer — a single isotropic clearance
    radius sized from vehicle WIDTH alone (length ignored), against the kinematic corridor.
    Kept to measure how much the length-aware box margin of score_geometric actually buys.
    """
    if len(preds) == 0:
        return 0.0
    plan, _ = ego_plan_kinematic(env)
    d = np.linalg.norm(preds - plan[None, None, :, :], axis=-1)
    clearance = d - (0.5 * env.agent.WIDTH + 0.5 * extents[:, 1][:, None, None] + SAFETY_MARGIN)
    return float(max(0.0, -np.min(clearance)))


def score_geometric_route(env, preds: np.ndarray, extents: np.ndarray) -> float:
    """
    ABLATION ONLY ("geom_route"): score_geometric's box margin, but against the ROUTE-following
    corridor (ego_plan_route) instead of the route-agnostic kinematic one. Kept to measure
    how much aligning the trigger's ego-motion model with the harm metric's actually buys.
    """
    plan, head = ego_plan_route(env)
    return _intrusion(env, preds, extents, plan, head)


def score_geometric(env, preds: np.ndarray, extents: np.ndarray) -> float:
    """
    T3/T4 score family (notes/design.md Sec. 2): how deeply the predicted region of any
    agent intrudes into the ego's planned corridor over the horizon. 0 = no predicted
    conflict. T3 and T4 share this exact score; they differ only in how lam is CHOSEN
    (T3: a plain split-conformal quantile, swept; T4: Learn-then-Test / Conformal Risk
    Control on the closed-loop loss, Section 3).

    Intrusion is measured in the ego's path-relative frame at each horizon step: decompose
    the offset to a predicted agent position into LONGITUDINAL (along the planned heading)
    and LATERAL (perpendicular) components, and apply separate margins — a length-based
    longitudinal margin (following distance) and a width-based lateral one (lane clearance).
    A single isotropic radius cannot represent both without either missing straight-ahead
    lead vehicles (radius sized by width alone; length ignored — MISSED 11/12 dev misses at
    lam=0 on 2026-09-22, notes/design.md §11) or over-triggering on adjacent-lane traffic
    (radius sized by the half-diagonal; SUPERSEDED, notes/design.md §1).
    """
    if len(preds) == 0:
        return 0.0
    plan, head = ego_plan(env)                                # (T,2), (T,)
    tangent = np.stack([np.cos(head), np.sin(head)], -1)      # (T,2)
    normal = np.stack([-np.sin(head), np.cos(head)], -1)      # (T,2)

    d = preds - plan[None, None, :, :]                        # (A,K,T,2)
    along = np.abs(np.einsum("aktd,td->akt", d, tangent))
    lat = np.abs(np.einsum("aktd,td->akt", d, normal))

    lon_margin = 0.5 * (env.agent.LENGTH + extents[:, 0]) + LON_MARGIN   # (A,)
    lat_margin = 0.5 * (env.agent.WIDTH + extents[:, 1]) + LAT_MARGIN    # (A,)
    # elliptical penetration depth in [0, lat_margin]: 0 outside the margin box, lat_margin
    # at the ego's own position. Smoother than a hard box indicator (keeps lam sweepable).
    r = np.sqrt((along / lon_margin[:, None, None]) ** 2 + (lat / lat_margin[:, None, None]) ** 2)
    intrusion = np.clip(1.0 - r, 0.0, 1.0) * lat_margin[:, None, None]
    # worst over time, then the most pessimistic mode, then the most threatening agent
    return float(np.max(intrusion))


def agent_table(env, preds: np.ndarray, probs: np.ndarray, extents: np.ndarray, is_net=None) -> list:
    """
    Per road user, one row [conf, gap, dist, ahead, net]:
      conf  = 1 - max_k P(mode k)                      (the predictor's stated uncertainty)
      gap   = min over (mode, t) of the Chebyshev box gap to the ego's kinematic corridor,
              metres, <= 0 = predicted overlap         (same geometry as score_gap)
      dist  = distance from ego to the agent's first predicted position, metres
      ahead = 1 if that position lies ahead of the ego along its heading
      net   = 1 if the NEURAL NETWORK predicted this agent. 0 = constant-velocity fallback
              with fabricated uniform probabilities: its `conf` is NOT a model confidence
              and must be excluded from any confidence-based trigger.
    Stored per decision tick in the reference run, so ANY agent-selection rule for a
    confidence baseline (most conflicting, nearest ahead, least confident) is computed
    offline with no re-simulation.
    """
    if len(preds) == 0:
        return []
    plan, head = ego_plan_kinematic(env)
    tangent = np.stack([np.cos(head), np.sin(head)], -1)
    normal = np.stack([-np.sin(head), np.cos(head)], -1)
    d = preds - plan[None, None, :, :]
    along = np.abs(np.einsum("aktd,td->akt", d, tangent))
    lat = np.abs(np.einsum("aktd,td->akt", d, normal))
    lon_m = 0.5 * (env.agent.LENGTH + extents[:, 0])[:, None, None]
    lat_m = 0.5 * (env.agent.WIDTH + extents[:, 1])[:, None, None]
    gap = np.maximum(along - lon_m, lat - lat_m).reshape(len(preds), -1).min(1)
    p0 = np.asarray(env.agent.position, float)
    first = preds[:, :, 0, :].mean(1) - p0                         # (A,2)
    dist = np.linalg.norm(first, axis=1)
    ahead = (first @ np.array([np.cos(env.agent.heading_theta), np.sin(env.agent.heading_theta)]) > 0)
    conf = 1.0 - probs.max(axis=1)
    net = np.ones(len(preds), bool) if is_net is None else np.asarray(is_net, bool)
    return [[round(float(c), 4), round(float(min(g, 100.0)), 3), round(float(di), 2), int(a), int(n)]
            for c, g, di, a, n in zip(conf, gap, dist, ahead, net)]


def score_confidence(env, preds: np.ndarray, probs: np.ndarray) -> float:
    """
    Confidence of the LEAST confident predicted vehicle ("conf_any"): u = max over up to 16
    nearby vehicles of 1 - max_k P(mode k).

    THIS IS NOT THE T1 BASELINE AND IS A KNOWN-BROKEN QUANTITY. On 2 000 calibration
    scenarios its scenario-peak took 8 distinct values (median = max = 0.833 = 1 - 1/K).
    Cause (found 2026-09-24): the adapter gives every agent the network did NOT predict
    (pedestrians, cyclists, vehicles beyond 60 m) a constant-velocity forecast with
    FABRICATED uniform mode probabilities, so any scene containing one pins this score at its
    ceiling. It is not evidence about AutoBot's confidence -- Paper 01's independent inference
    path shows real variation (median max-prob 0.25, 462 distinct values of 1-max, AUROC 0.63
    for high error). T1 is `conf_conflict` (evaluate.py): the confidence of the NETWORK-
    predicted agent whose modes come closest to the ego -- notes/design.md Sec. 2's
    "most conflicting agent". This function is kept only so stored traces stay interpretable.
    """
    if len(probs) == 0:
        return 0.0
    return float(np.max(1.0 - probs.max(axis=1)))


def score_gap(env, preds: np.ndarray, extents: np.ndarray) -> float:
    """
    Negative box gap in metres: -min over (agent, mode, t) of the Chebyshev gap between a
    predicted agent position and the ego's kinematic corridor, after the same length/width
    margins as score_geometric. <= 0 inside = overlap. Firing when the gap drops below q
    is exactly "inflate every predicted mode by radius q and fire if it touches the ego" --
    so the classic OPEN-LOOP conformal trigger (T3: q = the split-conformal radius of the
    predictor's error, a guarantee on the prediction, not the decision) is the threshold
    lam = -q on this one stored trace, for any alpha, with no extra simulation.
    """
    if len(preds) == 0:
        return -100.0
    plan, head = ego_plan_kinematic(env)
    tangent = np.stack([np.cos(head), np.sin(head)], -1)
    normal = np.stack([-np.sin(head), np.cos(head)], -1)
    d = preds - plan[None, None, :, :]
    along = np.abs(np.einsum("aktd,td->akt", d, tangent))
    lat = np.abs(np.einsum("aktd,td->akt", d, normal))
    lon_m = 0.5 * (env.agent.LENGTH + extents[:, 0])[:, None, None]
    lat_m = 0.5 * (env.agent.WIDTH + extents[:, 1])[:, None, None]
    gap = np.maximum(along - lon_m, lat - lat_m)
    return float(-min(np.min(gap), 100.0))


SCORE_FNS = {
    "gap": score_gap,                          # T3 (open-loop conformal) + tuned-gap baselines
    "geom": score_geometric,                   # T3/T4 (ours): kinematic corridor, box margin
    "conf": score_confidence,                  # T1: 1 - max mode probability
    "geom_route": score_geometric_route,       # ablation: route-following corridor
    "geom_iso": score_geometric_isotropic,     # ablation: width-only isotropic radius
}


def all_scores(env, preds: np.ndarray, probs: np.ndarray, extents: np.ndarray) -> dict:
    """Every trigger score from ONE predictor call, so one reference rollout serves all triggers."""
    return {
        "gap": score_gap(env, preds, extents),
        "geom": score_geometric(env, preds, extents),
        "conf": score_confidence(env, preds, probs),
        "geom_route": score_geometric_route(env, preds, extents),
        "geom_iso": score_geometric_isotropic(env, preds, extents),
    }


def trigger_score(env, predictor, other_states: np.ndarray, extents: np.ndarray,
                  score_key: str = "geom") -> float:
    """Backward-compatible single-score entry point (one predictor call per tick)."""
    preds, probs, extents = predict_agents(env, predictor, other_states, extents)
    if score_key == "conf":
        return score_confidence(env, preds, probs)
    return SCORE_FNS[score_key](env, preds, extents)


# ------------------------------------------------------------------------------- metrics
@dataclass
class RolloutResult:
    seed: int
    lam: float
    score_key: str = "geom"
    steps: int = 0
    collision: bool = False              # any contact with an agent or traffic object
    collision_step: int | None = None
    at_fault_step: int | None = None     # first nuPlan at-fault contact
    collisions: list = field(default_factory=list)   # [{step, type, at_fault}] per contact event
    offroad: bool = False                # sidewalk / boundary / building (nuPlan: drivable-area, separate)
    min_ttc: float = float("inf")
    ttc_violations: int = 0              # steps with nuPlan TTC < 0.95 s
    triggered: bool = False
    trigger_step: int | None = None
    diverged: bool = False               # pre-trigger state differed from the reference run (should never be True)
    stopped_steps: int = 0
    route_completion: float = 0.0
    arrive: bool = False
    peak_decel: float = 0.0
    scores: list = field(default_factory=list)
    score_traces: dict = field(default_factory=dict)  # key -> per-decision-tick score (reference run only)
    agent_tables: list = field(default_factory=list)   # per tick: [[conf, gap, dist, ahead], ...] per agent
    feats: list = field(default_factory=list)          # pre-deployment scenario covariates (shift weighting)
    ttc_trace: list = field(default_factory=list)   # per-step nuPlan TTC (s), capped at TTC_CAP
    ego_trace: list = field(default_factory=list)   # per-step ego [x, y]; kept in memory, not saved


TTC_CAP = 99.0


@dataclass(frozen=True)
class HarmDef:
    """
    What counts as the harmful event the trigger must pre-empt. Evaluated post hoc from the
    stored trace, so one set of rollouts can be scored under every candidate definition.

    The HEADLINE definition is nuPlan's (see nuplan_metrics.py): an ego AT-FAULT collision,
    or TTC < 0.95 s at any step (TTC is not evaluated while ego is stopped, and ignores
    tracks behind ego). Collisions the ego suffers while stopped or from behind are not
    at-fault — they are what a fallback *causes*, and are scored on the cost side
    (`induced_collision` in sweep_scenario), never as a miss.

      ttc_s         : TTC below this is a near-miss
      sustain       : ...only if it holds this many consecutive steps (1 = instantaneous, nuPlan)
      use_ttc       : False -> collisions only
      at_fault_only : count only nuPlan at-fault collisions (True for the headline)
    """
    ttc_s: float = 0.95
    sustain: int = 1
    use_ttc: bool = True
    at_fault_only: bool = True

    def first_harm(self, r: "RolloutResult") -> int | None:
        cands = []
        cstep = r.at_fault_step if self.at_fault_only else r.collision_step
        if cstep is not None:
            cands.append(cstep)
        if self.use_ttc:
            run = 0
            for i, t in enumerate(r.ttc_trace):
                run = run + 1 if t < self.ttc_s else 0
                if run >= self.sustain:
                    cands.append(i - self.sustain + 1)
                    break
        return min(cands) if cands else None

    def label(self) -> str:
        c = "at-fault collision" if self.at_fault_only else "any collision"
        if not self.use_ttc:
            return c
        return f"{c} or TTC<{self.ttc_s:g}s" + (f" x{self.sustain}" if self.sustain > 1 else "")


NUPLAN_HARM = HarmDef()
DEFAULT_HARM = NUPLAN_HARM


def _agents(env) -> np.ndarray:
    """Every other road user (vehicles, pedestrians, cyclists): (A, 6) = [x, y, heading, speed, L, W]."""
    rows = []
    for o in env.engine.get_objects().values():
        if o is env.agent or not hasattr(o, "heading_theta") or not hasattr(o, "velocity"):
            continue
        rows.append([o.position[0], o.position[1], o.heading_theta, o.speed, o.LENGTH, o.WIDTH])
    return np.asarray(rows, float).reshape(-1, 6)


def _ego(env) -> dict:
    a = env.agent
    return dict(x=a.position[0], y=a.position[1], h=a.heading_theta, v=a.speed, L=a.LENGTH, W=a.WIDTH)


def _trigger_inputs(agents: np.ndarray):
    """[x, y, vx, vy] and [L, W] for the predictor / trigger score."""
    vx = agents[:, 3] * np.cos(agents[:, 2])
    vy = agents[:, 3] * np.sin(agents[:, 2])
    return np.stack([agents[:, 0], agents[:, 1], vx, vy], 1), agents[:, 4:6]


FEATURE_NAMES = ["n_vehicles_50m", "n_pedestrians_50m", "n_cyclists_50m", "ego_speed0",
                 "mean_agent_speed_50m", "min_gap0", "route_length"]


def scenario_features(env) -> list:
    """
    Covariates observable BEFORE deployment (at t = 0), used only to estimate the density
    ratio for shift-weighted calibration (evaluate.py). Nothing here looks at outcomes.
    """
    import nuplan_metrics as nm
    ego = _ego(env)
    p0 = np.array([ego["x"], ego["y"]])
    counts = {"SVehicle": 0, "Pedestrian": 0, "Cyclist": 0}
    speeds = []
    for o in env.engine.get_objects().values():
        if o is env.agent or not hasattr(o, "velocity"):
            continue
        if np.linalg.norm(np.asarray(o.position[:2]) - p0) <= 50.0:
            name = type(o).__name__
            counts[name if name in counts else "SVehicle"] += 1
            speeds.append(float(o.speed))
    agents = _agents(env)
    gap = float(np.min(nm.box_gap(ego, agents))) if len(agents) else 100.0
    return [counts["SVehicle"], counts["Pedestrian"], counts["Cyclist"], float(ego["v"]),
            float(np.mean(speeds)) if speeds else 0.0, min(gap, 100.0),
            float(env.agent.navigation.reference_trajectory.length)]


def _agent_contact(v) -> bool:
    """Contact with a road user or traffic object. MetaDrive's info['crash'] also includes
    sidewalk / boundary / building contact, which nuPlan scores separately (drivable area)."""
    return bool(v.crash_vehicle or v.crash_human or v.crash_object)


def rollout(env, seed: int, lam: float, predictor, decide_every: int = 1,
            latency_steps: int = 0, max_steps: int = 1000,
            score_trace: list | None = None, ref_ego: list | None = None,
            score_key: str = "geom", fire_step: int | None = None,
            record_all: bool = False) -> RolloutResult:
    """
    One closed-loop episode. The fallback engages the first time the score exceeds lam
    (after `latency_steps` of actuation delay) and stays engaged. lam = inf disables it,
    which is the counterfactual used to label unnecessary stops. predictor=None skips
    scoring entirely (~3x faster) for studies that only need the no-fallback outcome.

    score_key selects which trigger drives the fallback: "geom" (T3/T4 family,
    score_geometric) or "conf" (T1, score_confidence) — see SCORE_FNS. This changes the
    physical rollout (a different score fires the MRM at a different step), so — unlike the
    harm definition, which is rescored post hoc from a stored trace — comparing triggers
    needs its own set of rollouts per trigger, not a single shared sweep.

    score_trace / ref_ego: replay the scores of a reference (lam = inf) run instead of
    calling the predictor. Exact, not an approximation: with common random numbers the sim
    is deterministic, so a lam-rollout is identical to the reference until its own trigger,
    and after the trigger no score is needed. `ref_ego` verifies that identity step by step;
    any mismatch sets `diverged` (reported by run_sweep; must be zero). A replayed trace was
    necessarily recorded under ONE score_key; reusing it under a different key would silently
    mix triggers, so score_trace and score_key are the caller's joint responsibility.

    fire_step: force the fallback at exactly this step, with no scoring at all. A rollout's
    physical outcome depends only on WHEN the fallback fires, not on which score or threshold
    caused it (pre-trigger runs are identical — test_score_trace_reuse_is_exact). So one
    reference run that records every score (record_all=True) plus one fire_step rollout per
    decision tick lets ANY trigger, threshold, online method or latency be evaluated exactly
    after the fact (src/evaluate.py). This supersedes the "each trigger needs its own
    rollouts" note above.
    """
    import nuplan_metrics as nm

    env.reset(seed=seed)
    live = predictor is not None and score_trace is None and fire_step is None
    if live and hasattr(predictor, "begin_scenario"):
        predictor.begin_scenario(env)
    elif live and hasattr(predictor, "reseed"):
        predictor.reseed(seed)
    pol = env.engine.get_policy(env.agent.name)
    pol.fallback_engaged = False
    pol._mrm_t = None          # never inherit manoeuvre state from a previous episode
    r = RolloutResult(seed=seed, lam=lam, score_key=score_key)
    r.feats = scenario_features(env)
    fire_at = fire_step
    prev_speed = env.agent.speed
    in_contact = False

    for step in range(max_steps):
        if (live or score_trace is not None) and step % decide_every == 0 and not r.triggered:
            if live and record_all:
                states, extents = _trigger_inputs(_agents(env))
                preds, probs, ext = predict_agents(env, predictor, states, extents)
                sc = all_scores(env, preds, probs, ext)
                r.agent_tables.append(agent_table(env, preds, probs, ext, getattr(predictor, "last_is_net", None)))
                # T2: MC-dropout ensemble spread of the nearest vehicles (0 if unavailable)
                sp = getattr(predictor, "last_spread", {}) or {}
                sc["ens"] = float(max(sp.values())) if sp else 0.0
                for k, v in sc.items():
                    r.score_traces.setdefault(k, []).append(round(v, 4))
                u = sc[score_key]
            elif live:
                states, extents = _trigger_inputs(_agents(env))
                u = trigger_score(env, predictor, states, extents, score_key=score_key)
            else:
                u = score_trace[step // decide_every]
            r.scores.append(round(u, 3))
            if u > lam and fire_at is None:
                fire_at = step + latency_steps
        if fire_at is not None and step >= fire_at and not r.triggered:
            r.triggered, r.trigger_step = True, step
            pol.fallback_engaged = True

        _, _, tm, tc, info = env.step([0.0, 0.0])
        r.steps = step + 1
        if live and hasattr(predictor, "observe"):
            predictor.observe(env)
        xy = [round(float(env.agent.position[0]), 3), round(float(env.agent.position[1]), 3)]
        r.ego_trace.append(xy)
        if ref_ego is not None and not r.triggered and step < len(ref_ego):
            if abs(xy[0] - ref_ego[step][0]) > 0.01 or abs(xy[1] - ref_ego[step][1]) > 0.01:
                r.diverged = True

        ego, agents = _ego(env), _agents(env)
        ttc = nm.ttc_nuplan(ego, agents)
        r.ttc_trace.append(round(min(ttc, TTC_CAP), 3))
        r.min_ttc = min(r.min_ttc, ttc)
        r.ttc_violations += int(ttc < nm.LEAST_MIN_TTC)

        contact = _agent_contact(env.agent)
        if contact and not in_contact:                 # rising edge = a new contact event
            ctype, at_fault, _ = nm.classify_collision(ego, agents, env.agent.navigation.current_lateral)
            r.collisions.append(dict(step=step, type=ctype, at_fault=at_fault))
            r.collision = True
            if r.collision_step is None:
                r.collision_step = step
            if at_fault and r.at_fault_step is None:
                r.at_fault_step = step
        in_contact = contact
        r.offroad = r.offroad or bool(env.agent.crash_sidewalk or env.agent.crash_building)

        r.stopped_steps += int(env.agent.speed < STOPPED_SPEED)
        r.peak_decel = max(r.peak_decel, (prev_speed - env.agent.speed) / DT)
        prev_speed = env.agent.speed
        if tm or tc:
            r.route_completion = float(info.get("route_completion", 0.0))
            r.arrive = bool(info.get("arrive_dest"))
            break
    return r


def harmful(r: RolloutResult, harm: HarmDef = DEFAULT_HARM) -> bool:
    """The event the trigger is supposed to pre-empt."""
    return harm.first_harm(r) is not None


def missed(r: RolloutResult, harm: HarmDef = DEFAULT_HARM) -> bool:
    """Harm happened AND the fallback did not fire strictly before it (firing after is no intervention)."""
    h = harm.first_harm(r)
    return h is not None and not (r.triggered and r.trigger_step < h)


def induced_collision(r: RolloutResult, cf: RolloutResult) -> bool:
    """
    The freezing cost made measurable: the fallback run suffered a NOT-at-fault contact
    (ego stopped / rear-ended) that the no-fallback counterfactual did not. nuPlan would
    not blame the ego — but the stop caused it.
    """
    return any(not c["at_fault"] for c in r.collisions) and not any(not c["at_fault"] for c in cf.collisions)


def sweep_scenario(env, seed: int, lams, predictor, harm: HarmDef = DEFAULT_HARM, **kw) -> dict:
    """
    Run one scenario across the lam grid plus the lam = inf counterfactual, and build the
    per-scenario row that src/risk_control.py consumes.

      loss    = 1 if harm happened and the fallback did not fire first    (missed intervention)
      stop    = 1 if the fallback fired although the counterfactual was harmless (unnecessary stop)
      induced = 1 if the stop itself led to a not-at-fault collision      (freezing is unsafe)
    """
    cf = rollout(env, seed, float("inf"), predictor, **kw)
    rows, loss, stop, trig, ind = [], [], [], [], []
    for lam in lams:
        r = rollout(env, seed, float(lam), predictor, score_trace=cf.scores, ref_ego=cf.ego_trace, **kw)
        rows.append(r)
        loss.append(float(missed(r, harm)))
        stop.append(float(r.triggered and not harmful(cf, harm)))
        trig.append(float(r.triggered))
        ind.append(float(induced_collision(r, cf)))
    def slim(x):
        d = asdict(x)
        d.pop("ego_trace")
        return d
    return dict(seed=seed, counterfactual=slim(cf), rollouts=[slim(x) for x in rows],
                loss=loss, stop=stop, triggered=trig, induced=ind,
                diverged=[float(x.diverged) for x in rows])


OUTCOME_KEYS = ("steps", "collision", "collision_step", "at_fault_step", "collisions", "offroad",
                "min_ttc", "ttc_violations", "triggered", "trigger_step", "diverged",
                "stopped_steps", "route_completion", "arrive", "peak_decel", "ttc_trace")


def sweep_scenario_ticks(env, seed: int, predictor, decide_every: int = 5, **kw) -> dict:
    """
    The campaign unit. One live reference run (lambda = inf) records EVERY trigger score at
    every decision tick; then one fallback rollout is forced at each decision tick. Every
    trigger / threshold / online method / latency is then an exact lookup
    (src/evaluate.py) — no approximation, no per-method simulation.
    """
    cf = rollout(env, seed, float("inf"), predictor, decide_every=decide_every,
                 record_all=True, **kw)
    fired = {}
    for k in range(0, cf.steps, decide_every):
        r = rollout(env, seed, float("inf"), None, decide_every=decide_every,
                    fire_step=k, ref_ego=cf.ego_trace)
        fired[str(k)] = {key: getattr(r, key) for key in OUTCOME_KEYS}
    ref = {key: getattr(cf, key) for key in OUTCOME_KEYS}
    ref.update(score_traces=cf.score_traces, feats=cf.feats, agent_tables=cf.agent_tables)
    return dict(seed=seed, decide_every=decide_every, ref=ref, fired=fired,
                n_diverged=sum(int(v["diverged"]) for v in fired.values()))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="av2_test", choices=list(simenv.DBS))
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 5], metavar=("START", "END"))
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--spread", type=float, default=3.0)
    ap.add_argument("--decide_every", type=int, default=1)
    ap.add_argument("--latency_steps", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    simenv.block_torch()
    start, end = args.seeds
    env = simenv.make_env(args.db, start, end - start, policy=make_fallback_policy_cls())
    predictor = ConstantVelocityPredictor(spread=args.spread)
    out = []
    try:
        for seed in range(start, end):
            env.reset(seed=seed)
            if simenv.ego_is_static(env):
                print(f"seed={seed} skipped (static ego)")
                continue
            row = sweep_scenario(env, seed, args.lams, predictor,
                                 decide_every=args.decide_every, latency_steps=args.latency_steps)
            out.append(row)
            cf = row["counterfactual"]
            print(f"seed={seed} cf: harm={cf['collision'] or cf['ttc_violations'] > 0} "
                  f"steps={cf['steps']} minTTC={cf['min_ttc']:.2f} | "
                  f"loss={row['loss']} stop={row['stop']} trig={row['triggered']}")
    finally:
        env.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
