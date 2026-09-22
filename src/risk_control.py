"""
Decision-level risk calibration for the fallback trigger (Paper 03, contribution 1).

A trigger fires when an uncertainty score u_t exceeds a threshold lam. For each
calibration scenario i and each candidate lam_j we run the CLOSED LOOP under lam_j and
record a bounded loss  L[i, j] in [0, 1]  (default: missed intervention = a harmful event
occurred with no fallback before it). The loss depends on the whole rollout, so it need
not be monotone in lam — hence Learn-then-Test is the primary method and Conformal Risk
Control is the monotone special case.

Conventions
-----------
lams   : (J,)    candidate thresholds, sorted ASCENDING (small lam = trigger often = safe)
losses : (N, J)  per-scenario closed-loop loss under each lam, values in [0, 1]
stops  : (N, J)  optional; 1 if the fallback fired and was unnecessary (the counterfactual
                 lam = inf rollout was harmless) — the over-conservatism the paper measures

References
----------
Angelopoulos, Bates, Candès, Jordan, Lei (2021) "Learn then Test".
Bates, Angelopoulos, Lei, Malik, Jordan (2021) "Distribution-free, risk-controlling
    prediction sets" — Hoeffding–Bentkus p-value.
Angelopoulos, Bates, Fisch, Lei, Schuster (2022) "Conformal Risk Control".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

__all__ = [
    "hb_pvalue",
    "learn_then_test",
    "conformal_risk_control",
    "risk_coverage_curve",
]


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b), with 0 log 0 = 0."""
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """
    Hoeffding–Bentkus p-value for H0: R > alpha, given the empirical mean of n i.i.d.
    losses in [0, 1]. Small p => evidence that the true risk is <= alpha.
    """
    if risk_hat >= alpha:
        return 1.0
    p_hoeffding = np.exp(-n * _h1(risk_hat, alpha))
    p_bentkus = np.e * binom.cdf(np.ceil(n * risk_hat), n, alpha)
    return float(min(p_hoeffding, p_bentkus, 1.0))


def learn_then_test(lams: np.ndarray, losses: np.ndarray, alpha: float, delta: float) -> dict:
    """
    Fixed-sequence LTT. Test from the most conservative lam (smallest) upward and stop at
    the first non-rejection; every rejected lam controls risk at level alpha with
    family-wise error <= delta. Returns the most permissive valid lam.

    With probability >= 1 - delta over calibration scenarios, for every returned-valid lam,
    E_test[loss(lam)] <= alpha — for ANY loss in [0, 1], monotone or not.
    """
    lams = np.asarray(lams, float)
    losses = np.asarray(losses, float)
    assert np.all(np.diff(lams) > 0), "lams must be strictly ascending"
    n, J = losses.shape
    assert J == len(lams)
    r_hat = losses.mean(axis=0)
    pvals = np.array([hb_pvalue(r, n, alpha) for r in r_hat])

    valid = np.zeros(J, bool)
    for j in range(J):
        if pvals[j] > delta:
            break
        valid[j] = True
    idx = int(np.flatnonzero(valid)[-1]) if valid.any() else None
    return dict(
        lam_hat=float(lams[idx]) if idx is not None else None,  # None => must always fall back
        idx=idx, valid=valid, pvals=pvals, risk_hat=r_hat, n=n, alpha=alpha, delta=delta,
    )


def conformal_risk_control(lams: np.ndarray, losses: np.ndarray, alpha: float, B: float = 1.0) -> dict:
    """
    Conformal Risk Control for a loss NON-DECREASING in lam (per scenario).
    lam_hat = largest lam with (n * R_hat(lam) + B) / (n + 1) <= alpha
    gives E[loss_test(lam_hat)] <= alpha in expectation over calibration + test.

    Validity needs per-scenario monotonicity; `monotone_violations` reports how often the
    closed-loop data break it, which is itself a result worth reporting.
    """
    lams = np.asarray(lams, float)
    losses = np.asarray(losses, float)
    n = losses.shape[0]
    r_hat = losses.mean(axis=0)
    ok = (n * r_hat + B) / (n + 1) <= alpha
    # guard: require every smaller lam to satisfy too, so lam_hat sits below the first failure
    first_fail = int(np.argmin(ok)) if not ok.all() else len(lams)
    idx = first_fail - 1 if first_fail > 0 else None
    viol = float(np.mean(np.any(np.diff(losses, axis=1) < 0, axis=1)))
    return dict(
        lam_hat=float(lams[idx]) if idx is not None else None,
        idx=idx, risk_hat=r_hat, bound=(n * r_hat + B) / (n + 1),
        monotone_violations=viol, n=n, alpha=alpha,
    )


def risk_coverage_curve(losses: np.ndarray, triggered: np.ndarray, stops: np.ndarray | None = None) -> dict:
    """
    Selective-prediction view over the lam grid (Exp 3).
      autonomy  = fraction of scenarios in which the fallback never fired  ("coverage")
      risk      = mean closed-loop loss                                     (missed interventions)
      unnecessary_stop = mean over scenarios of `stops`                     (over-conservatism)
    triggered, stops : (N, J) in {0, 1}
    """
    out = dict(
        autonomy=1.0 - np.asarray(triggered, float).mean(axis=0),
        risk=np.asarray(losses, float).mean(axis=0),
    )
    if stops is not None:
        out["unnecessary_stop"] = np.asarray(stops, float).mean(axis=0)
    return out
