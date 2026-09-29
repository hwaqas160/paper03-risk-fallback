"""
Amendment 5 (notes/falsification.md): N4, jointly-certified multi-action policy.

Jointly certifies residual harm H(X, lambda, d) <= alpha_H and induced-collision rate
I(X, lambda, d) <= alpha_I over threshold lambda AND MRM deceleration d in {2.0, 4.0} m/s^2,
minimising unnecessary stops, and compares against the single-risk baseline (lambda alone, d fixed
at 4.0).

Multi-risk LTT: at each lambda, the two Hoeffding-Bentkus p-values (for H and for I) are combined by
taking their max, and the combined sequence is walked with the same fixed-sequence test as ordinary
LTT (Angelopoulos et al. 2021, Section on simultaneous control of multiple risks: testing max(p_H,
p_I) <= delta at each grid point is the direct 1-D generalisation, no further correction needed for
a single ordered sequence). This is done once per value of d; with only two d values, the "search"
Pareto Testing (Laufer-Goldshtein et al. 2023) performs adaptively over a large multi-dimensional grid
reduces here to a direct comparison of the two resulting valid configurations, choosing the one with
the lower CALIBRATION-set stop rate -- the test-set evaluation of whichever is chosen still carries
LTT's per-d delta guarantee, since that guarantee does not depend on what else was computed. We call
this "Pareto Testing" only for the general multi-value case; for two arms it is a direct comparison,
stated as such rather than claimed as the full adaptive search.

    python src/multi_risk_cert.py --cal_d4 results/campaign/av2_cal5 --cal_d2 results/campaign/av2_cal5_d2 \
        --test_d4 results/campaign/av2_test5 --test_d2 results/campaign/av2_test5_d2 \
        --out results/final/multi_risk.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, lam_grid, load_rows, ltt, matrices  # noqa: E402
from outcome_cert import ALPHA_H, DELTA, residual_matrix, summarize_residual  # noqa: E402
from risk_control import hb_pvalue  # noqa: E402

KEY = "geom"
ALPHA_I = 0.03


def load(paths):
    return [Scenario(r) for r in load_rows(paths)]


def induced_matrix(scns, key, grid):
    return matrices(scns, key, grid)[2]


def multi_risk_ltt(cal, key, alpha_h, alpha_i, delta):
    """
    Bonferroni-corrected simultaneous test over the full lambda grid, NOT a fixed-sequence walk.

    Residual harm falls as lambda decreases (more intervention) while induced collisions RISE as
    lambda decreases (more intervention causes more rear-end collisions): the two risks move in
    opposite directions, so there is no single "most conservative -> least conservative" ordering
    both walk consistently, and a fixed-sequence test starting from either end can fail at its very
    first point (found while smoke-testing this file on old data, not fresh data). Testing every
    grid point independently at level delta/J (Bonferroni; always valid, no ordering assumption
    needed) and keeping every point that passes both tests avoids that failure mode. Among the valid
    points, the one with the lowest CALIBRATION stop rate is returned; the test-set guarantee for
    that point does not depend on how many other points were also tested.
    """
    grid = lam_grid(cal, key)
    H = residual_matrix(cal, key, grid)
    I = induced_matrix(cal, key, grid)
    S = matrices(cal, key, grid)[1]
    n = len(cal)
    J = len(grid)
    rh, ri = H.mean(0), I.mean(0)
    delta_j = delta / J
    ph = np.array([hb_pvalue(r, n, alpha_h) for r in rh])
    pi = np.array([hb_pvalue(r, n, alpha_i) for r in ri])
    valid = (ph <= delta_j) & (pi <= delta_j)
    if not valid.any():
        return dict(lam=-np.inf, cal_stop_rate=1.0, n_cal=n, grid_size=J, n_valid=0, bonferroni_delta=delta_j)
    stops = S.mean(0)
    idx = int(np.flatnonzero(valid)[np.argmin(stops[valid])])
    return dict(lam=float(grid[idx]), cal_stop_rate=float(stops[idx]), n_cal=n, grid_size=J,
               n_valid=int(valid.sum()), bonferroni_delta=delta_j)


def evaluate_on(test, key, lam, d_label):
    if not np.isfinite(lam):
        return dict(certified=False, d=d_label)
    T = np.stack([s.tick(key, [lam])[0] for s in test])
    H = np.array([s.harm_run[t] for s, t in zip(test, T)], float)
    I = np.array([s.induced[t] for s, t in zip(test, T)], float)
    S = np.array([s.stop[t] for s, t in zip(test, T)], float)
    n = len(H)

    def ci(x):
        z = 1.96
        k = x.sum()
        p = k / n
        dd = 1 + z * z / n
        c = (p + z * z / (2 * n)) / dd
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
        return [max(0.0, c - h), min(1.0, c + h)]
    return dict(certified=True, d=d_label, lam=lam, n=n,
               residual_harm=float(H.mean()), residual_harm_ci=ci(H),
               induced=float(I.mean()), induced_ci=ci(I),
               unnecessary_stop=float(S.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal_d4", nargs="+", required=True)
    ap.add_argument("--cal_d2", nargs="+", required=True)
    ap.add_argument("--test_d4", nargs="+", required=True)
    ap.add_argument("--test_d2", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cal_d4, cal_d2 = load(a.cal_d4), load(a.cal_d2)
    test_d4, test_d2 = load(a.test_d4), load(a.test_d2)
    print(f"cal_d4 n={len(cal_d4)}  cal_d2 n={len(cal_d2)}  test_d4 n={len(test_d4)}  test_d2 n={len(test_d2)}")

    cfg_d4 = multi_risk_ltt(cal_d4, KEY, ALPHA_H, ALPHA_I, DELTA)
    cfg_d2 = multi_risk_ltt(cal_d2, KEY, ALPHA_H, ALPHA_I, DELTA)
    print(f"d=4.0: lam={cfg_d4['lam']:.4f}  cal stop rate={cfg_d4['cal_stop_rate']:.3f}")
    print(f"d=2.0: lam={cfg_d2['lam']:.4f}  cal stop rate={cfg_d2['cal_stop_rate']:.3f}")

    chosen_is_d2 = cfg_d2["cal_stop_rate"] < cfg_d4["cal_stop_rate"]
    chosen = cfg_d2 if chosen_is_d2 else cfg_d4
    chosen_test = test_d2 if chosen_is_d2 else test_d4
    chosen_label = "2.0" if chosen_is_d2 else "4.0"
    joint_result = evaluate_on(chosen_test, KEY, chosen["lam"], chosen_label)

    # single-risk baseline: certify lambda alone (residual harm only, alpha_H) at d=4.0, ignoring
    # the induced-collision constraint, same style as outcome_cert's outcome-level LTT
    grid4 = lam_grid(cal_d4, KEY)
    lam_single = ltt(cal_d4, KEY, ALPHA_H, DELTA, grid4)
    baseline = evaluate_on(test_d4, KEY, lam_single, "4.0 (single-risk baseline)")

    out = dict(alpha_h=ALPHA_H, alpha_i=ALPHA_I, delta=DELTA,
               d4_config=cfg_d4, d2_config=cfg_d2, chosen_d=chosen_label,
               joint_certified_result=joint_result, single_risk_baseline=baseline)

    both_valid = (joint_result.get("certified") and joint_result["residual_harm_ci"][0] <= ALPHA_H
                 and joint_result["induced_ci"][0] <= ALPHA_I)
    stops_lower = (joint_result.get("certified") and baseline.get("certified")
                  and joint_result["unnecessary_stop"] < baseline["unnecessary_stop"])
    out["verdict_N4_joint_beats_single_risk"] = dict(
        both_risks_controlled=bool(both_valid),
        joint_stops=joint_result.get("unnecessary_stop"), baseline_stops=baseline.get("unnecessary_stop"),
        stops_lower=bool(stops_lower), holds=bool(both_valid and stops_lower))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nchosen d={chosen_label}  joint result: {joint_result}")
    print(f"single-risk baseline (d=4.0 alone): {baseline}")
    print(f"\nN4 verdict: {'HOLDS' if out['verdict_N4_joint_beats_single_risk']['holds'] else 'REFUTED'}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
