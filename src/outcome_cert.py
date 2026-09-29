"""
Amendment 5 (notes/falsification.md): outcome-level certification. N1-N3.

The alarm-level certificate (evaluate.ltt on `miss`) bounds whether the alert preceded harm. It says
nothing about whether harm occurs somewhere in the run the vehicle actually drives. This module
certifies RESIDUAL HARM directly: `Scenario.harm_run` at the firing tick, 1 if the nuPlan harm
definition fires anywhere in the executed run (before or after the fallback engages).

Only LTT is used as the primary calibrator (N3's premise): residual harm is not guaranteed monotone
in lambda (firing earlier can itself create harm), so CRC's guarantee does not apply without checking
it, and MultiRisk-style DP guarantees assume monotonicity too. CRC is still computed and its violation
frequency measured, precisely so N3 has a number to refute or support, not skipped because it "should"
fail.

    python src/outcome_cert.py --cal results/campaign/av2_cal5 --test results/campaign/av2_test5 \
        --require_tables=False --out results/final/outcome_cert.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, lam_grid, load_rows, ltt, matrices, summarize, tuned  # noqa: E402
from risk_control import conformal_risk_control, learn_then_test  # noqa: E402

KEY = "geom"
ALPHA_H, DELTA = 0.05, 0.10


def load(paths, require_tables=False):
    sc = [Scenario(r) for r in load_rows(paths)]
    return [s for s in sc if "conf_conflict" in s.traces] if require_tables else sc


def residual_matrix(scns, key, grid):
    """H(X, lambda): 1 if harm occurs anywhere in the executed run (before or after firing)."""
    T = np.stack([s.tick(key, grid) for s in scns])
    return np.stack([s.harm_run[t] for s, t in zip(scns, T)]).astype(float)


def nonmonotone_fraction(H: np.ndarray) -> float:
    """Share of scenarios whose residual-harm row is not monotone non-decreasing in lambda.
    The grid is ascending (smallest = fires soonest/most intervention, inf = never fires), so a
    well-behaved loss only ever goes UP as lambda grows (less intervention -> no less safe).
    A violation (some diff < 0) means firing later avoided harm that firing sooner caused, or
    vice versa -- the case CRC's monotonicity assumption does not cover. Matches the sign
    convention risk_control.conformal_risk_control uses for its own `monotone_violations`."""
    return float(np.mean(np.any(np.diff(H, axis=1) < 0, axis=1)))


def ltt_on_residual(cal_scns, grid, alpha=ALPHA_H, delta=DELTA):
    H = residual_matrix(cal_scns, KEY, grid)
    r = learn_then_test(grid, H, alpha, delta)
    return (-np.inf if r["lam_hat"] is None else float(r["lam_hat"])), H


def crc_on_residual(cal_scns, grid, alpha=ALPHA_H):
    H = residual_matrix(cal_scns, KEY, grid)
    r = conformal_risk_control(grid, H, alpha)
    return (-np.inf if r["lam_hat"] is None else float(r["lam_hat"])), H


def summarize_residual(scns, lam):
    T = np.stack([s.tick(KEY, [lam])[0] for s in scns])
    H = np.array([s.harm_run[t] for s, t in zip(scns, T)], float)
    S = np.array([s.stop[t] for s, t in zip(scns, T)], float)
    I = np.array([s.induced[t] for s, t in zip(scns, T)], float)
    n = len(H)
    z = 1.96

    def ci(x):
        k = x.sum()
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [max(0.0, c - h), min(1.0, c + h)]
    return dict(n=n, residual_harm=float(H.mean()), residual_harm_ci=ci(H),
                unnecessary_stop=float(S.mean()), induced=float(I.mean()))


def violation_freq(pool, lam_fn, key, reps=200, alpha=ALPHA_H, seed=0):
    """Re-split `pool` into cal/test 200 times (same size split each time), re-certify on each
    cal half with `lam_fn`, measure the fraction of splits whose TEST residual-harm rate exceeds
    alpha. This is the same validity check used for H1-H3, applied to residual harm."""
    rng = np.random.default_rng(seed)
    n = len(pool)
    half = n // 2
    bad = 0
    for _ in range(reps):
        perm = rng.permutation(n)
        cal = [pool[i] for i in perm[:half]]
        test = [pool[i] for i in perm[half:]]
        grid = lam_grid(cal, key)
        lam = lam_fn(cal, grid)
        if not np.isfinite(lam):
            bad += 1
            continue
        T = np.stack([s.tick(key, [lam])[0] for s in test])
        H = np.mean([s.harm_run[t] for s, t in zip(test, T)])
        bad += int(H > alpha)
    return bad / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--require_tables", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cal = load(a.cal, a.require_tables)
    test = load(a.test, a.require_tables)
    grid = lam_grid(cal, KEY)
    print(f"cal n={len(cal)} test n={len(test)}")

    # --- N1/N2: outcome-level LTT vs the alarm-level certificate, both re-fit on THIS fresh cal ---
    lam_alarm = ltt(cal, KEY, ALPHA_H, DELTA)                    # re-fit alarm-level LTT on fresh cal
    lam_outcome, H_cal = ltt_on_residual(cal, grid)
    nonmono = nonmonotone_fraction(H_cal)
    alarm_on_test = summarize(test, KEY, lam_alarm) if np.isfinite(lam_alarm) else None
    alarm_residual = summarize_residual(test, lam_alarm) if np.isfinite(lam_alarm) else None
    outcome_on_test = summarize_residual(test, lam_outcome) if np.isfinite(lam_outcome) else None

    # --- N3: CRC on residual harm, compared to LTT's validity ---
    lam_crc, _ = crc_on_residual(cal, grid)
    crc_on_test = summarize_residual(test, lam_crc) if np.isfinite(lam_crc) else None

    def stop_diff_ci(lam_a, lam_b, B=10000, seed=0):
        if not (np.isfinite(lam_a) and np.isfinite(lam_b)):
            return None
        Ta = np.stack([s.tick(KEY, [lam_a])[0] for s in test])
        Tb = np.stack([s.tick(KEY, [lam_b])[0] for s in test])
        Sa = np.array([s.stop[t] for s, t in zip(test, Ta)], float)
        Sb = np.array([s.stop[t] for s, t in zip(test, Tb)], float)
        d = Sa - Sb
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boots = d[idx].mean(1)
        return dict(diff=float(d.mean()), ci=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))])

    out = dict(
        n_cal=len(cal), n_test=len(test), alpha_h=ALPHA_H, delta=DELTA,
        nonmonotone_fraction_cal=nonmono,
        lam_alarm=lam_alarm, lam_outcome=lam_outcome, lam_crc=lam_crc,
        alarm_level_on_test=dict(miss=alarm_on_test["miss"] if alarm_on_test else None,
                                 miss_ci=alarm_on_test["miss_ci"] if alarm_on_test else None,
                                 unnecessary_stop=alarm_on_test["unnecessary_stop"] if alarm_on_test else None,
                                 residual_harm=alarm_residual["residual_harm"] if alarm_residual else None,
                                 residual_harm_ci=alarm_residual["residual_harm_ci"] if alarm_residual else None),
        outcome_level_ltt_on_test=outcome_on_test,
        outcome_level_crc_on_test=crc_on_test,
        stop_rate_diff_alarm_minus_outcome=stop_diff_ci(lam_alarm, lam_outcome),
    )

    print(f"validity check (this can take a while: {a.reps} resplits x 2 calibrators)...")
    out["validity_ltt_outcome"] = violation_freq(cal + test, lambda c, g: ltt_on_residual(c, g)[0], KEY, a.reps)
    out["validity_crc_outcome"] = violation_freq(cal + test, lambda c, g: crc_on_residual(c, g)[0], KEY, a.reps)

    # --- verdicts, mechanical, per Amendment 5 ---
    v = {}
    if alarm_residual and outcome_on_test:
        stop_diff = out["stop_rate_diff_alarm_minus_outcome"]
        distinguishable = stop_diff is not None and not (stop_diff["ci"][0] <= 0 <= stop_diff["ci"][1])
        v["N1_outcome_cert_valid_and_distinct"] = dict(
            violation=out["validity_ltt_outcome"], valid=out["validity_ltt_outcome"] <= DELTA,
            stop_diff=stop_diff, distinguishable=distinguishable,
            holds=bool(out["validity_ltt_outcome"] <= DELTA and distinguishable))
    if alarm_residual:
        lo, hi = alarm_residual["residual_harm_ci"]
        v["N2_alarm_cert_violates_outcome_target"] = dict(
            residual_harm=alarm_residual["residual_harm"], ci=[lo, hi],
            holds=bool(lo > ALPHA_H))
    v["N3_crc_unsafe_on_nonmonotone_loss"] = dict(
        crc_violation=out["validity_crc_outcome"], ltt_violation=out["validity_ltt_outcome"],
        holds=bool(out["validity_crc_outcome"] > DELTA and out["validity_ltt_outcome"] <= DELTA))
    out["verdicts"] = v

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    print(f"\nnon-monotone fraction (cal): {nonmono:.3f}")
    print(f"lambda: alarm={lam_alarm:.4f} outcome-LTT={lam_outcome:.4f} outcome-CRC={lam_crc:.4f}")
    print(f"alarm-level cert on test:   miss={alarm_on_test['miss']:.3f}  residual harm={alarm_residual['residual_harm']:.3f} "
          f"CI={alarm_residual['residual_harm_ci']}  stops={alarm_on_test['unnecessary_stop']:.3f}")
    print(f"outcome-level LTT on test:  residual harm={outcome_on_test['residual_harm']:.3f} "
          f"CI={outcome_on_test['residual_harm_ci']}  stops={outcome_on_test['unnecessary_stop']:.3f}")
    if crc_on_test:
        print(f"outcome-level CRC on test: residual harm={crc_on_test['residual_harm']:.3f} "
              f"CI={crc_on_test['residual_harm_ci']}  stops={crc_on_test['unnecessary_stop']:.3f}")
    print(f"validity (200 resplits): LTT-outcome={out['validity_ltt_outcome']:.3f}  CRC-outcome={out['validity_crc_outcome']:.3f}  (valid if <= {DELTA})")
    print("\nverdicts:")
    for k, val in v.items():
        print(f"  {k}: {'HOLDS' if val['holds'] else 'REFUTED'}  {json.dumps({x: y for x, y in val.items() if x != 'holds'}, default=float)}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
