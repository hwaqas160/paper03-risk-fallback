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
                     rollout, sweep_scenario, trigger_score)


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
