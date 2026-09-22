"""
Split conformal prediction for marginal trajectory forecasting, plus the label-free
recalibration methods for Paper 01 (H3).

Everything here is numpy-only and model-agnostic: it consumes arrays of predicted
trajectories + ground truth + (optionally) per-scene features. No torch, no UniTraj.

Conventions
-----------
pred_trajs : (N, K, T, 2)   K predicted modes, T horizon steps, (x, y)
pred_probs : (N, K)         mode probabilities (softmax); may be None
gt          : (N, T, 2)     ground-truth future
gt_mask     : (N, T) bool   valid horizon steps (True = valid); default all-valid
feats       : (N, D)        per-scene shift factors (H2/H3); optional

The nonconformity score (pre-registered in notes/falsification.md):

    s_i = min_k  max_{t valid}  || pred_trajs[i,k,t] - gt[i,t] ||_2

i.e. the worst-timestep error of the mode that best covers the ground truth.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "nonconformity_scores",
    "split_conformal_quantile",
    "coverage_and_efficiency",
    "SplitConformal",
    "WeightedSplitConformal",
    "NormalizedSplitConformal",
    "GroupConditionalConformal",
    "domain_classifier_weights",
]


# --------------------------------------------------------------------------------------
# core score + coverage
# --------------------------------------------------------------------------------------
def _pairwise_disp_err(pred_trajs: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, K, T) L2 displacement error per mode per step."""
    diff = pred_trajs - gt[:, None, :, :]              # (N, K, T, 2)
    return np.linalg.norm(diff, axis=-1)              # (N, K, T)


def nonconformity_scores(
    pred_trajs: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """s_i = min_k max_{t valid} ||pred - gt||. Shape (N,)."""
    err = _pairwise_disp_err(pred_trajs, gt)          # (N, K, T)
    if gt_mask is None:
        per_mode_worst = err.max(axis=-1)            # (N, K)
    else:
        m = gt_mask[:, None, :]                       # (N, 1, T)
        masked = np.where(m, err, -np.inf)
        per_mode_worst = masked.max(axis=-1)         # (N, K)
    return per_mode_worst.min(axis=-1)               # (N,)


def split_conformal_quantile(cal_scores: np.ndarray, alpha: float,
                             weights: np.ndarray | None = None) -> float:
    """
    Finite-sample-valid conformal quantile.

    Unweighted: the ceil((n+1)(1-alpha))/n empirical quantile (Vovk).
    Weighted:   Tibshirani et al. (2019) weighted conformal — weights are normalised
                over calibration points plus a point mass at +inf for the test point.
    """
    n = len(cal_scores)
    if n == 0:
        return np.inf
    if weights is None:
        q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(cal_scores, q_level, method="higher"))

    w = np.asarray(weights, dtype=float)
    order = np.argsort(cal_scores)
    s_sorted = cal_scores[order]
    w_sorted = w[order]
    # normalise with test-point weight = mean cal weight (exchangeable-in-weight proxy)
    w_test = w.mean()
    cw = np.cumsum(w_sorted)
    total = cw[-1] + w_test
    target = (1.0 - alpha) * total
    idx = int(np.searchsorted(cw, target, side="left"))
    if idx >= n:
        return np.inf
    return float(s_sorted[idx])


def coverage_and_efficiency(
    test_scores: np.ndarray,
    q_hat: float,
) -> dict:
    """
    Given nonconformity scores on a test set and a radius q_hat, the region covers the
    ground truth iff s_i <= q_hat (by construction of the score).
    'Efficiency' = the radius itself (mean region 'size' proxy) and the score distribution.
    """
    covered = test_scores <= q_hat
    return {
        "coverage": float(np.mean(covered)),
        "q_hat": float(q_hat),
        "mean_score": float(np.mean(test_scores)),
        "median_score": float(np.median(test_scores)),
        "n": int(len(test_scores)),
    }


# --------------------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------------------
class SplitConformal:
    """Vanilla split conformal. Calibrate on source, apply anywhere."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat_: float | None = None
        self.cal_scores_: np.ndarray | None = None

    def calibrate(self, cal_scores: np.ndarray) -> "SplitConformal":
        self.cal_scores_ = np.asarray(cal_scores, dtype=float)
        self.q_hat_ = split_conformal_quantile(self.cal_scores_, self.alpha)
        return self

    def evaluate(self, test_scores: np.ndarray) -> dict:
        assert self.q_hat_ is not None, "call calibrate() first"
        return coverage_and_efficiency(np.asarray(test_scores, float), self.q_hat_)


class WeightedSplitConformal:
    """
    H3 method (a): importance-weighted split conformal.
    Weights w_i ~ p_target(x_i) / p_source(x_i), estimated from scene features by a
    domain classifier (see domain_classifier_weights). Uses ONLY unlabelled target x.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat_: float | None = None

    def calibrate(self, cal_scores: np.ndarray, cal_weights: np.ndarray) -> "WeightedSplitConformal":
        cal_scores = np.asarray(cal_scores, float)
        cal_weights = np.asarray(cal_weights, float)
        self.q_hat_ = split_conformal_quantile(cal_scores, self.alpha, weights=cal_weights)
        return self

    def evaluate(self, test_scores: np.ndarray) -> dict:
        assert self.q_hat_ is not None
        return coverage_and_efficiency(np.asarray(test_scores, float), self.q_hat_)


class NormalizedSplitConformal:
    """
    H3 method (b): scene-adaptive normalised nonconformity.
    Divide the score by a per-scene scale sigma(x) >= sigma_min, fit on the SOURCE
    calibration set (features -> score) by ridge regression on |score|. The conformal
    guarantee is preserved because normalisation is a fixed function of x.
    """

    def __init__(self, alpha: float = 0.1, ridge: float = 1.0, sigma_min: float = 1e-2):
        self.alpha = alpha
        self.ridge = ridge
        self.sigma_min = sigma_min
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.feat_mean_: np.ndarray | None = None
        self.feat_std_: np.ndarray | None = None
        self.q_hat_: float | None = None

    def _sigma(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.feat_mean_) / self.feat_std_
        raw = z @ self.w_ + self.b_
        return np.maximum(np.abs(raw), self.sigma_min)

    def calibrate(self, cal_scores: np.ndarray, cal_feats: np.ndarray,
                  fit_scores: np.ndarray | None = None,
                  fit_feats: np.ndarray | None = None) -> "NormalizedSplitConformal":
        cal_scores = np.asarray(cal_scores, float)
        cal_feats = np.asarray(cal_feats, float)
        fs = cal_scores if fit_scores is None else np.asarray(fit_scores, float)
        ff = cal_feats if fit_feats is None else np.asarray(fit_feats, float)

        self.feat_mean_ = ff.mean(0)
        self.feat_std_ = ff.std(0) + 1e-8
        z = (ff - self.feat_mean_) / self.feat_std_
        d = z.shape[1]
        A = z.T @ z + self.ridge * np.eye(d)
        self.w_ = np.linalg.solve(A, z.T @ fs)
        self.b_ = float(fs.mean() - (z.mean(0) @ self.w_))

        norm_cal = cal_scores / self._sigma(cal_feats)
        self.q_hat_ = split_conformal_quantile(norm_cal, self.alpha)
        return self

    def evaluate(self, test_scores: np.ndarray, test_feats: np.ndarray) -> dict:
        assert self.q_hat_ is not None
        norm_test = np.asarray(test_scores, float) / self._sigma(np.asarray(test_feats, float))
        out = coverage_and_efficiency(norm_test, self.q_hat_)
        # report the effective radius in raw units too (median sigma * q_hat)
        out["q_hat_raw_equiv"] = float(self.q_hat_ * np.median(self._sigma(test_feats)))
        return out


class GroupConditionalConformal:
    """
    H3 method (c): bin scenes by a discrete group id (e.g. agent-density tertile),
    quantile within group, route test scenes to their group's q_hat.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_by_group_: dict = {}
        self.global_q_: float | None = None

    def calibrate(self, cal_scores: np.ndarray, cal_groups: np.ndarray) -> "GroupConditionalConformal":
        cal_scores = np.asarray(cal_scores, float)
        cal_groups = np.asarray(cal_groups)
        self.global_q_ = split_conformal_quantile(cal_scores, self.alpha)
        for g in np.unique(cal_groups):
            s = cal_scores[cal_groups == g]
            self.q_by_group_[g] = (split_conformal_quantile(s, self.alpha)
                                   if len(s) >= 50 else self.global_q_)
        return self

    def evaluate(self, test_scores: np.ndarray, test_groups: np.ndarray) -> dict:
        test_scores = np.asarray(test_scores, float)
        test_groups = np.asarray(test_groups)
        q = np.array([self.q_by_group_.get(g, self.global_q_) for g in test_groups])
        covered = test_scores <= q
        return {
            "coverage": float(np.mean(covered)),
            "q_hat": float(np.mean(q)),
            "mean_score": float(np.mean(test_scores)),
            "n": int(len(test_scores)),
        }


# --------------------------------------------------------------------------------------
# domain classifier for importance weights (H3 method a, and the H2 reweighting analysis)
# --------------------------------------------------------------------------------------
def domain_classifier_weights(
    source_feats: np.ndarray,
    target_feats: np.ndarray,
    clip: float = 20.0,
    C: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Fit logistic regression source(0) vs target(1) on standardised features, return
    density-ratio weights w(x) = p(target|x) / p(source|x) for the SOURCE points.

    Uses only feature vectors — no labels, no ground-truth futures. Requires sklearn.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs = np.asarray(source_feats, float)
    Xt = np.asarray(target_feats, float)
    X = np.vstack([Xs, Xt])
    y = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    p_t = clf.predict_proba(scaler.transform(Xs))[:, 1]
    p_t = np.clip(p_t, 1e-4, 1 - 1e-4)
    w = p_t / (1.0 - p_t)
    w = w / w.mean()
    return np.clip(w, 1.0 / clip, clip)


if __name__ == "__main__":
    # smoke test on synthetic data
    rng = np.random.default_rng(0)
    N, K, T = 4000, 6, 60
    gt = np.cumsum(rng.normal(0, 0.3, size=(N, T, 2)), axis=1)
    # one "good" mode near GT + noise, plus 5 distractors
    good = gt + rng.normal(0, 0.5, size=(N, T, 2))
    bad = rng.normal(0, 5, size=(N, K - 1, T, 2))
    pred = np.concatenate([good[:, None], bad], axis=1)
    s = nonconformity_scores(pred, gt)
    cal, test = s[:2000], s[2000:]
    for a in (0.05, 0.10, 0.20):
        sc = SplitConformal(alpha=a).calibrate(cal)
        r = sc.evaluate(test)
        print(f"alpha={a}  nominal={1-a:.2f}  empirical={r['coverage']:.3f}  q_hat={r['q_hat']:.2f}")
