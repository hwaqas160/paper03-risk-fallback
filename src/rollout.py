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
def make_fallback_policy_cls():
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
        MRM_DECEL = 4.0     # m/s², UN R157 §5.5.1 upper bound
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


def ego_plan(env, steps: int = PRED_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Where the nominal plan puts the ego over the horizon (position, heading) at current speed."""
    traj = env.agent.navigation.reference_trajectory
    long, _ = traj.local_coordinates(env.agent.position)
    v = max(env.agent.speed, 1.0)
    longs = [min(long + v * DT * (i + 1), traj.length) for i in range(steps)]
    pos = np.array([traj.position(s, 0) for s in longs])
    head = np.array([traj.heading_theta_at(s) for s in longs])
    return pos, head


def trigger_score(env, predictor, other_states: np.ndarray, extents: np.ndarray) -> float:
    """
    Task-relevant uncertainty: how deeply the predicted region of any agent intrudes into
    the ego's planned corridor over the horizon. 0 = no predicted conflict.

    Intrusion is measured in the ego's path-relative frame at each horizon step: decompose
    the offset to a predicted agent position into LONGITUDINAL (along the planned heading)
    and LATERAL (perpendicular) components, and apply separate margins — a length-based
    longitudinal margin (following distance) and a width-based lateral one (lane clearance).
    A single isotropic radius cannot represent both without either missing straight-ahead
    lead vehicles (radius sized by width alone; length ignored — MISSED 11/12 dev misses at
    lam=0 on 2026-09-22, notes/design.md §11) or over-triggering on adjacent-lane traffic
    (radius sized by the half-diagonal; SUPERSEDED, notes/design.md §1).

    This is the 'plan-intersecting region size' of notes/design.md T3/T4: the score family
    is shared by the conformal and risk-calibrated triggers; only how lam is CHOSEN differs.
    """
    if hasattr(predictor, "predict_env"):                    # AutoBot: needs sim history
        preds, _, extents = predictor.predict_env(env)
        if len(preds) == 0:
            return 0.0
        preds = preds[:, :, :PRED_HORIZON]
    else:
        if len(other_states) == 0:
            return 0.0
        preds = predictor.predict(other_states)              # (A,K,T,2)
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


# ------------------------------------------------------------------------------- metrics
@dataclass
class RolloutResult:
    seed: int
    lam: float
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


def _agent_contact(v) -> bool:
    """Contact with a road user or traffic object. MetaDrive's info['crash'] also includes
    sidewalk / boundary / building contact, which nuPlan scores separately (drivable area)."""
    return bool(v.crash_vehicle or v.crash_human or v.crash_object)


def rollout(env, seed: int, lam: float, predictor, decide_every: int = 1,
            latency_steps: int = 0, max_steps: int = 1000,
            score_trace: list | None = None, ref_ego: list | None = None) -> RolloutResult:
    """
    One closed-loop episode. The fallback engages the first time the score exceeds lam
    (after `latency_steps` of actuation delay) and stays engaged. lam = inf disables it,
    which is the counterfactual used to label unnecessary stops. predictor=None skips
    scoring entirely (~3x faster) for studies that only need the no-fallback outcome.

    score_trace / ref_ego: replay the scores of a reference (lam = inf) run instead of
    calling the predictor. Exact, not an approximation: with common random numbers the sim
    is deterministic, so a lam-rollout is identical to the reference until its own trigger,
    and after the trigger no score is needed. `ref_ego` verifies that identity step by step;
    any mismatch sets `diverged` (reported by run_sweep; must be zero).
    """
    import nuplan_metrics as nm

    env.reset(seed=seed)
    live = predictor is not None and score_trace is None
    if live and hasattr(predictor, "begin_scenario"):
        predictor.begin_scenario(env)
    elif live and hasattr(predictor, "reseed"):
        predictor.reseed(seed)
    pol = env.engine.get_policy(env.agent.name)
    pol.fallback_engaged = False
    pol._mrm_t = None          # never inherit manoeuvre state from a previous episode
    r = RolloutResult(seed=seed, lam=lam)
    fire_at = None
    prev_speed = env.agent.speed
    in_contact = False

    for step in range(max_steps):
        if (live or score_trace is not None) and step % decide_every == 0 and not r.triggered:
            if live:
                states, extents = _trigger_inputs(_agents(env))
                u = trigger_score(env, predictor, states, extents)
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
