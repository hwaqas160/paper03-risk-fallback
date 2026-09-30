"""
Amendment 6 post-hoc robustness analyses (notes/falsification.md). Reading rules are fixed there.

  R6.1  alignment:   python src/robustness.py alignment --out results/final/r6_alignment.json
  R6.2  replication: python src/robustness.py replicate --cal results/campaign/av2_cal5 \
            --test results/campaign/av2_test5 --name heldout --out results/final/r6_heldout.json
  R6.3  second predictor: same as R6.2 with the *_gpu arms, --name gpu

Everything reuses the validated functions in evaluate.py. Nothing is re-simulated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (Scenario, crc, lam_grid, load_rows, ltt, matrices, online, oracle,  # noqa: E402
                      summarize, tuned, validity)
from rollout import HarmDef  # noqa: E402

DELTA = 0.10
ETAS = (0.01, 0.05, 0.1, 0.5)


def load(paths, harm=None, drop_ens=False):
    out = []
    for r in load_rows(paths):
        s = Scenario(r) if harm is None else Scenario(r, harm)
        if "conf_conflict" not in s.traces:
            continue
        if drop_ens and "ens" in s.traces:
            del s.traces["ens"]
            del s.cummax["ens"]
        out.append(s)
    return out


def per_scenario(scns, key, lam):
    M, S, I = matrices(scns, key, np.array([lam]))[:3]
    return M[:, 0], S[:, 0], I[:, 0]


def paired_diff(a, b, B=10000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return dict(diff=float(d.mean()), ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])


def wilson(x):
    n = len(x)
    p = float(np.mean(x))
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def row(test, key, lam):
    if not np.isfinite(lam) and lam > 0:
        lam = np.inf
    m, s, i = per_scenario(test, key, lam)
    return dict(lam=float(lam), miss=float(m.mean()), miss_ci=wilson(m), unnecessary_stop=float(s.mean()),
                stop_ci=wilson(s), induced=float(i.mean()), _m=m, _s=s)


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------ R6.1
def alignment(a):
    defs = [("collision_only", HarmDef(use_ttc=False), [0.01, 0.02]),
            ("ttc_0.5s", HarmDef(ttc_s=0.5), [0.05]),
            ("ttc_0.95s (headline)", HarmDef(), [0.05]),
            ("ttc_1.5s", HarmDef(ttc_s=1.5), [0.05])]
    out = {}
    for name, harm, alphas in defs:
        cal = load(a.cal, harm)
        test = load(a.test, harm)
        base = float(np.mean([s.ref_harmful for s in test]))
        for alpha in alphas:
            lam_ltt = ltt(cal, "geom", alpha, DELTA)
            if not np.isfinite(lam_ltt) and alpha == 0.01 and name == "collision_only":
                out[f"{name} @ alpha={alpha}"] = dict(base_rate_test=base, ltt_certified=False)
                continue
            res = {"LTT geometric": row(test, "geom", lam_ltt),
                   "tuned geometric": row(test, "geom", tuned(cal, "geom", alpha)),
                   "tuned T1 confidence": row(test, "conf_conflict", tuned(cal, "conf_conflict", alpha)),
                   "tuned T2 ensemble": row(test, "ens", tuned(cal, "ens", alpha))}
            ok = [k for k in ("tuned T1 confidence", "tuned T2 ensemble") if res[k]["miss"] <= alpha]
            best = min(ok, key=lambda k: res[k]["unnecessary_stop"]) if ok else None
            verdict = None
            if best:
                pdiff = paired_diff(res["LTT geometric"]["_s"], res[best]["_s"])
                survives = pdiff["ci"][1] < 0
                verdict = dict(best_baseline=best, stop_diff_ltt_minus_best=pdiff, survives=bool(survives))
            else:
                verdict = dict(best_baseline=None, note="no T1/T2 baseline met miss <= alpha on test", survives=None)
            out[f"{name} @ alpha={alpha}"] = dict(base_rate_test=base, ltt_certified=bool(np.isfinite(lam_ltt)),
                                                   methods={k: strip(v) for k, v in res.items()}, verdict=verdict)
            if name == "collision_only" and np.isfinite(lam_ltt):
                break
    return out


# ------------------------------------------------------------------------------ R6.2 / R6.3
def replicate(a):
    cal = load(a.cal, drop_ens=True)
    test = load(a.test, drop_ens=True)
    alpha = a.alpha
    lam0 = tuned(cal, "geom", alpha)
    rng = np.random.default_rng(0)
    orders = [rng.permutation(len(test)) for _ in range(20)]
    best = None
    for eta in ETAS:
        runs = [online(test, "geom", lam0 if np.isfinite(lam0) else 0.0, eta, alpha, o) for o in orders]
        r = dict(eta=eta, miss=float(np.mean([x["miss"] for x in runs])),
                 unnecessary_stop=float(np.mean([x["unnecessary_stop"] for x in runs])),
                 induced=float(np.mean([x["induced"] for x in runs])))
        if best is None or (r["miss"] <= alpha and (best["miss"] > alpha or r["unnecessary_stop"] < best["unnecessary_stop"])):
            best = r
    res = {"never": row(test, "geom", np.inf),
           "LTT geometric": row(test, "geom", ltt(cal, "geom", alpha, DELTA)),
           "CRC geometric": row(test, "geom", crc(cal, "geom", alpha)),
           "tuned geometric": row(test, "geom", lam0),
           "tuned T1 confidence": row(test, "conf_conflict", tuned(cal, "conf_conflict", alpha)),
           "oracle": row(test, "geom", oracle(test, "geom", alpha))}
    val = validity(cal + test, alpha, DELTA, reps=a.reps)
    ltt_v = val["T4 LTT (ours)"]["violation_freq"]
    out = dict(name=a.name, n_cal=len(cal), n_test=len(test), alpha=alpha, delta=DELTA,
               methods={k: strip(v) for k, v in res.items()}, cdt_best_eta=best, validity=val,
               missing=["T2 ensemble (no MC dropout)", "T3 open-loop conformal and ACI (no open-loop error file)"],
               verdict=dict(ltt_validity=ltt_v, ltt_test_miss=res["LTT geometric"]["miss"],
                            replicates=bool(ltt_v <= DELTA and res["LTT geometric"]["miss"] <= alpha)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["alignment", "replicate"])
    ap.add_argument("--cal", nargs="+", default=["results/campaign/av2_cal"])
    ap.add_argument("--test", nargs="+", default=["results/campaign/av2_test"])
    ap.add_argument("--name", default="heldout")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = alignment(a) if a.what == "alignment" else replicate(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
