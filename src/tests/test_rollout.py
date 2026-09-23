"""
Behavioural checks for the closed-loop harness. Needs the simulator (~2 min, 1 process).

    F:\\CLAUDE\\AI1\\shared\\envs\\unitraj\\Scripts\\python.exe src/tests/test_rollout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

import simenv  # noqa: E402

simenv.block_torch()
from rollout import (ConstantVelocityPredictor, DT, make_fallback_policy_cls,  # noqa: E402
                     score_geometric,
                     rollout, sweep_scenario, trigger_score, ego_plan, score_confidence, predict_agents,
                     sweep_scenario_ticks, SCORE_FNS)


def _env(n=6):
    return simenv.make_env("av2_test", 0, n, policy=make_fallback_policy_cls())


def test_no_torch_in_sim_worker():
    assert "torch" not in sys.modules, "sim worker must not load torch (commit blowup)"


def test_lam_inf_never_triggers_and_lam_zero_always_does(env, pred):
    never = rollout(env, 0, float("inf"), pred)
    assert not never.triggered and never.trigger_step is None
    always = rollout(env, 0, 0.0, pred)
    assert always.triggered and always.trigger_step == 0


def test_fallback_actually_stops_the_car(env, pred):
    """The minimum-risk manoeuvre must brake to a stop — otherwise 'unnecessary stop' is meaningless."""
    braked = rollout(env, 0, 0.0, pred)
    free = rollout(env, 0, float("inf"), pred)
    assert braked.peak_decel > 0.5, f"no deceleration recorded: {braked.peak_decel}"
    assert braked.stopped_steps > free.stopped_steps, "braking rollout should spend longer stopped"
    assert braked.route_completion < free.route_completion, "stopping should reduce route completion"


def test_earlier_trigger_for_smaller_lam(env, pred):
    """Monotone in the trigger TIME (not necessarily in the loss) — a sanity check on the score."""
    steps = []
    for lam in (0.0, 1.5, 2.2):
        r = rollout(env, 0, lam, pred)
        steps.append(r.trigger_step if r.triggered else 10 ** 6)
    assert steps == sorted(steps), f"trigger step not monotone in lam: {steps}"


def test_latency_delays_the_trigger(env, pred):
    a = rollout(env, 0, 1.0, pred, latency_steps=0)
    b = rollout(env, 0, 1.0, pred, latency_steps=5)
    assert a.triggered and b.triggered
    assert b.trigger_step - a.trigger_step == 5, (a.trigger_step, b.trigger_step)


def test_score_is_zero_with_no_other_vehicles(env, pred):
    env.reset(seed=0)
    assert trigger_score(env, pred, np.zeros((0, 4)), np.zeros((0, 2))) == 0.0


def test_sweep_row_shapes(env, pred):
    lams = [0.0, 1.0, 2.5]
    row = sweep_scenario(env, 0, lams, pred)
    assert len(row["loss"]) == len(row["stop"]) == len(row["triggered"]) == len(lams)
    assert set(row["loss"]) <= {0.0, 1.0} and set(row["stop"]) <= {0.0, 1.0}
    # an unnecessary stop requires the counterfactual to be harmless
    assert len(row["induced"]) == len(lams) and set(row["induced"]) <= {0.0, 1.0}
    from rollout import RolloutResult, harmful
    if harmful(RolloutResult(**row["counterfactual"])):
        assert sum(row["stop"]) == 0


def test_ttc_finite_and_sane(env, pred):
    r = rollout(env, 0, float("inf"), pred)
    assert r.min_ttc >= 0.0
    assert r.steps > 1 and r.route_completion > 0.0
    assert len(r.ttc_trace) == r.steps
    for c in r.collisions:
        assert c["type"] in {"stopped_ego", "stopped_track", "active_front", "active_rear", "active_lateral"}


def test_rollouts_are_reproducible(env, pred):
    """Same scenario + same lam -> identical rollout (common random numbers across lam)."""
    a = rollout(env, 0, 2.0, pred)
    b = rollout(env, 0, 2.0, pred)
    assert a.scores == b.scores and a.trigger_step == b.trigger_step and a.steps == b.steps


def test_score_trace_reuse_is_exact(env, pred):
    """
    Replaying the lam=inf score trace must reproduce a live lam-rollout up to the trigger:
    no pre-trigger divergence and the same trigger step. AFTER the trigger MetaDrive itself
    is not bit-repeatable (two identical LIVE runs differed in 2/30 scenario-lam pairs on
    2026-09-22), so post-trigger outcomes are not compared — that noise is the simulator's,
    not the reuse's, and LTT does not need a deterministic simulator.
    """
    for seed in (0, 1, 2):
        ref = rollout(env, seed, float("inf"), pred)
        for lam in (0.5, 1.8, 2.3):
            live = rollout(env, seed, lam, pred)
            rep = rollout(env, seed, lam, pred, score_trace=ref.scores, ref_ego=ref.ego_trace)
            assert not rep.diverged, (seed, lam)
            assert live.trigger_step == rep.trigger_step, (seed, lam)
            k = live.trigger_step if live.triggered else live.steps
            assert live.ttc_trace[:k] == rep.ttc_trace[:k], (seed, lam)


def test_trigger_score_sees_lead_vehicle_not_just_width(env, pred):
    """
    A stopped car directly ahead, offset from ego by less than the vehicle LENGTH but more
    than the WIDTH, must score as a threat. A width-only radius (the pre-2026-09-22 scorer)
    misses this and was the diagnosed cause of 11/12 dev misses at the most aggressive
    threshold (notes/design.md #11).
    """
    env.reset(seed=0)
    for _ in range(3):
        env.step([0.0, 0.0])
    plan, head = ego_plan(env)
    ahead = plan[10] + 3.0 * np.array([np.cos(head[10]), np.sin(head[10])])  # 3 m ahead: inside
    # length-based margin, outside a width-based one (car width ~1.9-2.3 m)
    states = np.array([[ahead[0], ahead[1], 0.0, 0.0]])
    extents = np.array([[4.5, 1.9]])
    u = trigger_score(env, pred, states, extents)
    assert u > 0.0, "lead vehicle 3 m ahead was not scored as a threat"


def test_confidence_score_ignores_geometry(env, pred):
    """
    T1 (score_confidence) sees only mode probability, never distance. The stub predictor has
    no real notion of confidence, so it reports UNIFORM probability over its K modes -- the
    LOWEST possible max-probability (1/K) a proper distribution can have, which makes
    score_confidence = 1 - 1/K, a CONSTANT at its highest attainable value, identical whether
    or not any agent is actually nearby. That is a real, disclosed limitation of running T1
    against this stub (notes/design.md, predict_agents docstring), not a bug: the score is
    doing exactly what "1 - max mode probability" says, the stub just has nothing informative
    to report. score_geometric, by contrast, correctly reacts to the same scene.
    """
    env.reset(seed=0)
    for _ in range(3):
        env.step([0.0, 0.0])
    plan, head = ego_plan(env)
    ahead = plan[10] + 3.0 * np.array([np.cos(head[10]), np.sin(head[10])])
    states = np.array([[ahead[0], ahead[1], 0.0, 0.0]])
    extents = np.array([[4.5, 1.9]])
    stub = ConstantVelocityPredictor(spread=3.0, noise=0.0, bias=0.0, k=6)
    preds, probs, ext2 = predict_agents(env, stub, states, extents)
    assert probs.shape == (1, stub.k)
    assert np.allclose(probs, 1.0 / stub.k), "stub predictor must report uniform (uninformative) confidence"
    u_conf = score_confidence(env, preds, probs)
    u_geom = score_geometric(env, preds, ext2)
    expected = 1.0 - 1.0 / stub.k
    assert abs(u_conf - expected) < 1e-9, f"expected the constant 1-1/K={expected:.4f}, got {u_conf}"
    assert u_geom > 0.0, "the same lead vehicle should register on the geometric score"
    # move the agent far away: score_confidence must not change (it never looks at position)
    far_states = np.array([[ahead[0] + 500.0, ahead[1] + 500.0, 0.0, 0.0]])
    preds2, probs2, _ = predict_agents(env, stub, far_states, extents)
    assert score_confidence(env, preds2, probs2) == u_conf, "T1 must be invariant to agent distance"


def test_score_key_selects_which_trigger_fires(env, pred):
    """rollout(..., score_key=...) actually changes which score drives the fallback."""
    a = rollout(env, 0, 2.0, pred, score_key="geom")
    b = rollout(env, 0, 2.0, pred, score_key="conf")
    assert a.score_key == "geom" and b.score_key == "conf"
    # the stub's confidence score is identically 0 (uninformative), so lam=2.0 never fires
    assert not b.triggered


def test_fire_step_reproduces_lambda_trigger(env, pred):
    """
    The campaign design rests on this: a rollout forced to fire at step k is the same
    rollout as a lambda-triggered one that happens to fire at k (up to the simulator's
    documented post-trigger noise, which is why only trigger step + pre-trigger trace are
    compared, as in test_score_trace_reuse_is_exact).
    """
    for seed in (0, 1):
        ref = rollout(env, seed, float("inf"), pred, decide_every=5, record_all=True)
        assert set(ref.score_traces) == set(SCORE_FNS), ref.score_traces.keys()
        for lam in (0.5, 1.8):
            live = rollout(env, seed, lam, pred, decide_every=5)
            if not live.triggered:
                continue
            forced = rollout(env, seed, float("inf"), None, decide_every=5,
                             fire_step=live.trigger_step, ref_ego=ref.ego_trace)
            assert forced.trigger_step == live.trigger_step
            assert not forced.diverged
            k = live.trigger_step
            assert forced.ttc_trace[:k] == live.ttc_trace[:k]


def test_sweep_scenario_ticks_shape(env, pred):
    row = sweep_scenario_ticks(env, 0, pred, decide_every=5)
    assert row["n_diverged"] == 0
    ticks = sorted(int(k) for k in row["fired"])
    assert ticks[0] == 0 and all(b - a == 5 for a, b in zip(ticks, ticks[1:]))
    assert len(row["ref"]["score_traces"]["geom"]) == len(ticks)
    assert len(row["ref"]["feats"]) == 7


if __name__ == "__main__":
    test_no_torch_in_sim_worker()
    print("PASS test_no_torch_in_sim_worker")
    env, pred = _env(), ConstantVelocityPredictor(spread=3.0)
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and name != "test_no_torch_in_sim_worker":
                fn(env, pred)
                print("PASS", name)
    finally:
        env.close()
