"""
Pre-registered few-label recalibration study (notes/falsification.md, Amendment 4B).

For each k in {40, 60}: draw k labelled target scenarios uniformly at random (200 draws, seed 0),
evaluate on the remaining target scenarios. alpha = 0.05, delta = 0.10, score `geom`. Primary
analysis uses ALL non-static scenarios; the avoidable-only subset (no harm at the first decision
tick) is a labelled sensitivity. Methods:
  M1  AV2-certified LTT threshold, no target labels
  M2  target-only tuned (largest lam whose empirical miss on the k scenes <= alpha)
  M3  target-only LTT on the k scenes
  M4  AV2 + k weighted LTT: source weight 1, each target scene weight N_src/k, Kish n_eff
  M5  Luo-style class-conditional certificate on the k scenes at matched eps = alpha / pi_hat_target
Violation = realised test miss > alpha in a draw. R1-R4 are evaluated mechanically at the end.

    python src/recalibration_study.py --cal results/campaign/av2_cal --test results/campaign/av2_test \
        --target results/campaign/waymo_val --name waymo --out results/final/recal_waymo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, lam_grid, load_rows, matrices, summarize  # noqa: E402
from risk_control import hb_pvalue, learn_then_test  # noqa: E402

KEY = "geom"
ALPHA, DELTA = 0.05, 0.10


def load(paths, need_tables):
    sc = [Scenario(r) for r in load_rows(paths)]
    return [s for s in sc if "conf_conflict" in s.traces] if need_tables else sc


def ltt_lambda(grid, M, alpha=ALPHA, delta=DELTA):
    r = learn_then_test(grid, M, alpha, delta)
    return -np.inf if r["lam_hat"] is None else float(r["lam_hat"])


def tuned_idx(M, alpha=ALPHA):
    ok = np.flatnonzero(M.mean(0) <= alpha)
    return int(ok[-1]) if len(ok) else -1


def weighted_ltt_idx(M, w, alpha=ALPHA, delta=DELTA):
    w = np.asarray(w, float) / np.mean(w)
    n_eff = int(np.floor(w.sum() ** 2 / (w ** 2).sum()))
    risk = (w[:, None] * M).sum(0) / w.sum()
    best = -1
    for j in range(M.shape[1]):
        if hb_pvalue(float(risk[j]), max(n_eff, 1), alpha) > delta:
            break
        best = j
    return best, n_eff


def luo_idx(M_unsafe, eps):
    """Largest grid index with (misses among unsafe + 1) / (n_unsafe + 1) <= eps (prefix property)."""
    n = M_unsafe.shape[0]
    k = M_unsafe.sum(0)
    ok = (k + 1) / (n + 1) <= eps
    if not ok[0]:
        return -1
    return int(np.argmin(ok)) - 1 if not ok.all() else M_unsafe.shape[1] - 1


def run_one(cal, target, test_ref_lam_idx_fn=None, ks=(40, 60), reps=200, seed=0):
    grid = lam_grid(cal + target, KEY)
    Mc, Sc = matrices(cal, KEY, grid)[:2]
    Mt, St = matrices(target, KEY, grid)[:2]
    unsafe_t = np.array([s.ref_harmful for s in target])
    n_src = len(cal)
    lam1 = ltt_lambda(grid, Mc)
    j1 = -1 if not np.isfinite(lam1) else int(np.searchsorted(grid, lam1))
    rng = np.random.default_rng(seed)

    def at(j, rows):
        if j < 0:
            j = 0                        # nothing certifiable -> most conservative grid point (fires first)
        return float(Mt[rows, j].mean()), float(St[rows, j].mean())

    out = dict(n_target=len(target), n_cal=n_src, base_rate_target=float(unsafe_t.mean()),
               grid_lam1=lam1, M1_full_target=None)
    full = np.arange(len(target))
    m, s = at(j1, full)
    summ = summarize(target, KEY, lam1) if np.isfinite(lam1) else None
    out["M1_full_target"] = dict(miss=m, unnecessary_stop=s, miss_ci=None if summ is None else summ["miss_ci"],
                                 certified=bool(np.isfinite(lam1)))
    for k in ks:
        res = {n: dict(miss=[], stop=[], viol=[], certified=[], n_eff=[]) for n in ("M1", "M2", "M3", "M4", "M5")}
        for _ in range(reps):
            perm = rng.permutation(len(target))
            tr, te = perm[:k], perm[k:]
            picks = {}
            picks["M1"] = j1
            picks["M2"] = tuned_idx(Mt[tr])
            lam3 = ltt_lambda(grid, Mt[tr])
            picks["M3"] = -1 if not np.isfinite(lam3) else int(np.searchsorted(grid, lam3))
            w = np.r_[np.ones(n_src), np.full(k, n_src / k)]
            j4, ne = weighted_ltt_idx(np.vstack([Mc, Mt[tr]]), w)
            picks["M4"] = j4
            pi_hat = float(unsafe_t[tr].mean())
            eps = min(0.99, ALPHA / pi_hat) if pi_hat > 0 else 0.99
            us = tr[unsafe_t[tr]]
            picks["M5"] = luo_idx(Mt[us], eps) if len(us) else -1
            for name, j in picks.items():
                m, s = at(j, te)
                res[name]["miss"].append(m)
                res[name]["stop"].append(s)
                res[name]["viol"].append(m > ALPHA)
                res[name]["certified"].append(j >= 0)
                if name == "M4":
                    res[name]["n_eff"].append(ne)
        out[f"k{k}"] = {n: dict(miss_mean=float(np.mean(v["miss"])), stop_mean=float(np.mean(v["stop"])),
                                violation=float(np.mean(v["viol"])), certified=float(np.mean(v["certified"])),
                                n_eff_mean=(float(np.mean(v["n_eff"])) if v["n_eff"] else None))
                        for n, v in res.items()}
    return out


def verdicts(all_res, in_domain_stop):
    r = all_res
    v = {}
    v["R1_M2_violates_ge_50pct_both_k"] = dict(
        values=[r["k40"]["M2"]["violation"], r["k60"]["M2"]["violation"]],
        holds=bool(r["k40"]["M2"]["violation"] >= 0.5 and r["k60"]["M2"]["violation"] >= 0.5))
    v["R2_M4_k60_violation_le_delta"] = dict(value=r["k60"]["M4"]["violation"],
                                             holds=bool(r["k60"]["M4"]["violation"] <= DELTA))
    gap = r["k60"]["M4"]["stop_mean"] - in_domain_stop
    v["R3_M4_k60_stops_ge_10pts_above_in_domain"] = dict(value=gap, in_domain_stop=in_domain_stop, holds=bool(gap >= 0.10))
    ci = r["M1_full_target"]["miss_ci"]
    v["R4_M1_target_miss_CI_excludes_alpha"] = dict(
        ci=ci, holds=bool(ci is not None and not (ci[0] <= ALPHA <= ci[1])))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True, help="AV2 test rows (in-domain stop rate for R3)")
    ap.add_argument("--target", nargs="+", required=True)
    ap.add_argument("--name", default="target")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cal = load(a.cal, True)
    test = load(a.test, True)
    target = load(a.target, False)
    lam = ltt_lambda(lam_grid(cal, KEY), matrices(cal, KEY, lam_grid(cal, KEY))[0])
    in_dom = summarize(test, KEY, lam)["unnecessary_stop"] if np.isfinite(lam) else float("nan")
    res_all = run_one(cal, target, reps=a.reps)
    avoid = [s for s in target if s.miss[0] == 0]
    res_av = run_one(cal, avoid, reps=a.reps)
    out = dict(name=a.name, alpha=ALPHA, delta=DELTA, in_domain_stop=in_dom, all_scenarios=res_all,
               avoidable_only=res_av, verdicts_primary_all_scenarios=verdicts(res_all, in_dom))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    def show(tag, r):
        print(f"\n== {a.name} / {tag}: n_target={r['n_target']}  base harm rate={r['base_rate_target']:.3f} ==")
        print(f"M1 (AV2-certified, no labels): miss={r['M1_full_target']['miss']:.3f} "
              f"CI={r['M1_full_target']['miss_ci']}  stops={r['M1_full_target']['unnecessary_stop']:.3f}")
        for k in ("k40", "k60"):
            print(f" {k}:")
            for n, v in r[k].items():
                ne = f" n_eff={v['n_eff_mean']:.0f}" if v["n_eff_mean"] else ""
                print(f"   {n}  miss={v['miss_mean']:.3f}  P(miss>alpha)={v['violation']:.2f}  "
                      f"stops={v['stop_mean']:.3f}  certified={v['certified']:.2f}{ne}")
    show("ALL scenarios (primary)", res_all)
    show("avoidable-only (sensitivity)", res_av)
    print("\n== pre-registered verdicts (primary = all scenarios) ==")
    for k, v in out["verdicts_primary_all_scenarios"].items():
        print(f"  {k}: {'HOLDS' if v['holds'] else 'REFUTED'}  {json.dumps({x: y for x, y in v.items() if x != 'holds'}, default=float)}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
