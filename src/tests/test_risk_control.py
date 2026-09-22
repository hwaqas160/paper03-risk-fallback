"""
Synthetic validity checks for src/risk_control.py — no simulator needed.

Model: each scenario has a latent hazard h ~ Beta(1, 12) and the trigger score is a noisy
view of h. Under threshold lam, a harmful event is missed if the scenario is hazardous and
the score never exceeded lam. A fraction of scenarios are made NON-monotone (triggering
causes the harm, e.g. a hard stop in a live lane), so CRC's assumption is broken but LTT's
is not.

    python -m pytest src/tests/test_risk_control.py -q     (or run the file directly)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk_control import conformal_risk_control, hb_pvalue, learn_then_test  # noqa: E402

LAMS = np.linspace(0.0, 1.0, 41)


def simulate(n, rng, nonmono_frac=0.05):
    h = rng.beta(1, 12, size=n)
    harmful = rng.random(n) < 6 * h                       # ~ 40% hazardous
    score = np.clip(h * 4 + rng.normal(0, 0.15, n), 0, 1)
    triggered = score[:, None] > LAMS[None, :]             # (n, J)
    loss = (harmful[:, None] & ~triggered).astype(float)  # missed intervention
    nonmono = rng.random(n) < nonmono_frac
    loss[nonmono] = triggered[nonmono].astype(float)       # the stop itself causes harm
    return loss, triggered


def test_hb_pvalue_basic():
    assert hb_pvalue(0.2, 1000, 0.1) == 1.0
    assert hb_pvalue(0.0, 1000, 0.1) < 1e-6
    assert hb_pvalue(0.09, 100, 0.1) > hb_pvalue(0.09, 10000, 0.1)


def test_ltt_controls_risk_over_repeats():
    """P(test risk at lam_hat > alpha) must be <= delta (up to Monte-Carlo slack)."""
    alpha, delta, reps = 0.10, 0.10, 300
    rng = np.random.default_rng(0)
    big_loss, _ = simulate(200_000, rng)
    true_risk = big_loss.mean(axis=0)
    fails = 0
    for _ in range(reps):
        loss, _ = simulate(2000, rng)
        r = learn_then_test(LAMS, loss, alpha, delta)
        if r["idx"] is not None and true_risk[r["idx"]] > alpha:
            fails += 1
    assert fails / reps <= delta + 0.03, f"LTT violated: {fails}/{reps}"


def test_crc_reports_nonmonotonicity():
    rng = np.random.default_rng(1)
    loss, _ = simulate(2000, rng, nonmono_frac=0.05)
    r = conformal_risk_control(LAMS, loss, alpha=0.10)
    assert r["monotone_violations"] > 0.0
    loss_m, _ = simulate(2000, rng, nonmono_frac=0.0)
    assert conformal_risk_control(LAMS, loss_m, alpha=0.10)["monotone_violations"] == 0.0


def test_ltt_returns_none_when_unachievable():
    loss = np.ones((500, len(LAMS)))
    assert learn_then_test(LAMS, loss, 0.1, 0.1)["lam_hat"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
