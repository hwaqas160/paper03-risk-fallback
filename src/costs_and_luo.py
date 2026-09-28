"""
Post-hoc analyses (notes/falsification.md Amendment 4). Two questions, both answered exactly from
the stored per-tick outcomes:

  1. What does the intervention itself do?  A full outcome partition per scenario: harm before
     the alert (miss), alert but harm persists (too late), harm prevented, new harm caused by the
     intervention, induced not-at-fault collision, unnecessary stop without collision.
  2. How does a Luo et al. (IJRR 2024) class-conditional certificate compare?  Their guarantee is
     Pr[no alert | situation unsafe] <= eps (Mondrian split conformal on the UNSAFE class). Here
     "unsafe" = harm in the no-fallback reference run, "alert" = the trigger fires strictly before
     that harm. Ours is the MARGINAL miss probability over all scenarios. Both are evaluated on
     identical scenarios; the conditional FNR of every method is also reported.

    python src/costs_and_luo.py --cal results/campaign/av2_cal --test results/campaign/av2_test \
        --shift results/campaign/ns_val --out results/costs_luo.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (Scenario, lam_grid, load_rows, ltt, matrices, summarize, tuned)  # noqa: E402

KEY = "geom"


def load(paths, need_tables=True):
    sc = [Scenario(r) for r in load_rows(paths)]
    return [s for s in sc if "conf_conflict" in s.traces] if need_tables else sc


def breakdown(scns, lam, key=KEY):
    T = np.stack([s.tick(key, [lam])[0] for s in scns])
    parts = dict(miss=0, too_late=0, prevented=0, new_harm=0, induced_only=0, stop_clean=0, no_fire=0)
    for s, t in zip(scns, T):
        fired = t < s.n_ticks
        ref_h = s.ref_harmful
        if ref_h:
            if s.miss[t]:
                parts["miss"] += 1
            elif s.harm_run[t]:
                parts["too_late"] += 1
            else:
                parts["prevented"] += 1
        elif fired:
            if s.harm_run[t]:
                parts["new_harm"] += 1
            elif s.induced[t]:
                parts["induced_only"] += 1
            else:
                parts["stop_clean"] += 1
        else:
            parts["no_fire"] += 1
    n = len(scns)
    out = {k: v / n for k, v in parts.items()}
    out["residual_harm"] = out["miss"] + out["too_late"] + out["new_harm"]
    out["induced_any"] = float(np.mean([s.induced[t] for s, t in zip(scns, T)]))
    out["n"] = n
    return out


def cond_fnr(scns, lam, key=KEY):
    uns = [s for s in scns if s.ref_harmful]
    if not uns:
        return float("nan"), 0
    M = matrices(uns, key, np.array([lam]))[0][:, 0]
    return float(M.mean()), len(uns)


def luo_lambda(cal, eps, key=KEY):
    """Largest lambda with (misses among UNSAFE calibration scenarios + 1) / (M + 1) <= eps."""
    uns = [s for s in cal if s.ref_harmful]
    grid = lam_grid(cal, key)
    M = matrices(uns, key, grid)[0]
    k = M.sum(0)
    ok = (k + 1) / (len(uns) + 1) <= eps
    if not ok[0]:
        return -np.inf
    first_fail = int(np.argmin(ok)) if not ok.all() else len(grid)
    return float(grid[first_fail - 1])


def row(name, scns, lam, key=KEY):
    s = summarize(scns, key, lam)
    c, m = cond_fnr(scns, lam, key)
    b = breakdown(scns, lam, key)
    return dict(method=name, lam=lam, miss=s["miss"], cond_fnr=c, n_unsafe=m,
                unnecessary_stop=s["unnecessary_stop"], induced=s["induced"], **{f"b_{k}": v for k, v in b.items()})


def show(title, rows):
    print(f"\n== {title} ==")
    print(f"{'method':<40}{'marg.miss':>10}{'cond.FNR':>9}{'unnec':>7}{'induced':>8}{'residual':>9}"
          f"{'too_late':>9}{'prevent':>8}{'new_harm':>9}")
    for r in rows:
        print(f"{r['method']:<40}{r['miss']:>10.3f}{r['cond_fnr']:>9.3f}{r['unnecessary_stop']:>7.3f}{r['induced']:>8.3f}"
              f"{r['b_residual_harm']:>9.3f}{r['b_too_late']:>9.3f}{r['b_prevented']:>8.3f}{r['b_new_harm']:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--shift", nargs="+", default=None)
    ap.add_argument("--target", action="append", metavar="NAME=DIR")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--out", default="results/costs_luo.json")
    a = ap.parse_args()

    cal, test = load(a.cal), load(a.test)
    base = float(np.mean([s.ref_harmful for s in cal]))
    eps_matched = min(0.99, a.alpha / base)
    lam_ltt = ltt(cal, KEY, a.alpha, a.delta)
    lam_tuned = tuned(cal, KEY, a.alpha)
    lam_luo05 = luo_lambda(cal, a.alpha)
    lam_luoM = luo_lambda(cal, eps_matched)
    print(f"cal n={len(cal)} base harm rate={base:.3f}; unsafe cal n={sum(s.ref_harmful for s in cal)}; "
          f"eps_matched=alpha/base={eps_matched:.3f}")

    def table(scns):
        return [row("never fire", scns, np.inf),
                row("LTT (marginal, ours)", scns, lam_ltt),
                row("tuned geometric (no cert.)", scns, lam_tuned),
                row(f"Luo-style conditional eps={a.alpha:.2f}", scns, lam_luo05),
                row(f"Luo-style conditional eps={eps_matched:.2f} (matched)", scns, lam_luoM),
                row("always fire", scns, -np.inf)]

    out = dict(base_rate_cal=base, eps_matched=eps_matched, av2_test=table(test))
    show("Argoverse 2 test (certified on av2_cal)", out["av2_test"])
    targets = []
    if a.shift:
        targets.append(("nuScenes", load(a.shift, need_tables=False)))
    for spec in a.target or []:
        nm, _, pth = spec.partition("=")
        targets.append((nm, load([pth], need_tables=False)))
    for nm, scns in targets:
        out[nm] = table(scns)
        show(f"{nm}: thresholds certified on av2_cal, deployed unchanged "
             f"(base harm rate {np.mean([s.ref_harmful for s in scns]):.3f}, unsafe n={sum(s.ref_harmful for s in scns)})", out[nm])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
