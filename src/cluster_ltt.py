"""
Amendment 15 (notes/falsification.md): cluster-aware Learn-then-Test for the alert certificate under spatial dependence.

The Hoeffding-Bentkus p-value of LTT assumes n independent losses. When scenarios come in spatial groups (single-linkage components of ego
start points within EPS metres, per city), the calibration losses are dependent within a group. The cluster-aware certificate keeps the same
empirical risk but evaluates the p-value at the cluster effective sample size n_eff = N^2 / sum_g n_g^2 (the Kish effective size with groups as the
units of replication), so a calibration set dominated by a few large groups certifies less, and may refuse.  It is the cluster analogue of the
density-ratio-weighted LTT of Section III-D and is a heuristic with no finite-sample guarantee, except for the strictly valid 'cluster-Hoeffding'
variant (weighted Hoeffding bound for independent group means with weights n_g / N), which is reported alongside.

    python src/cluster_ltt.py describe --out results/final/r15_groups.json     (outcome-free group structure)
    python src/cluster_ltt.py run      --out results/final/r15_cluster_ltt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, lam_grid, matrices, summarize  # noqa: E402
from risk_control import hb_pvalue  # noqa: E402
from spatial_groups import components, ego_xy  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def build_groups(eps):
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    ld = lambda p: [s for s in (Scenario(r) for r in load_rows([p])) if "conf_conflict" in s.traces]
    cal, test = ld("results/campaign/av2_cal"), ld("results/campaign/av2_test")
    pool = cal + test
    xy = {"av2_cal": ego_xy("av2_cal", [s.seed for s in cal]), "av2_test": ego_xy("av2_test", [s.seed for s in test])}
    pts = np.stack([xy["av2_cal"][s.seed] for s in cal] + [xy["av2_test"][s.seed] for s in test])
    city = np.array([lab["av2_cal"][str(s.seed)] for s in cal] + [lab["av2_test"][str(s.seed)] for s in test])
    grp = np.zeros(len(pool), int)
    off = 0
    for c in sorted(set(city)):
        m = city == c
        g = components(pts[m], eps)
        grp[m] = g + off
        off += g.max() + 1
    return cal, test, pool, grp


def n_eff_of(grp_sub):
    sizes = np.bincount(np.unique(grp_sub, return_inverse=True)[1]).astype(float)
    return float(len(grp_sub) ** 2 / (sizes ** 2).sum())


def ltt_cluster(cal, grp_cal, key, alpha=ALPHA, delta=DELTA, mode="kish"):
    """Fixed-sequence LTT from the most conservative threshold; p-value at the cluster effective sample size.
    mode 'kish': Hoeffding-Bentkus at floor(n_eff); mode 'hoeffding': weighted Hoeffding on group means (strictly valid for independent groups)."""
    grid = lam_grid(cal, key)
    M = matrices(cal, key, grid)[0]                       # (N, G) miss indicators
    ne = n_eff_of(grp_cal)
    ids, inv = np.unique(grp_cal, return_inverse=True)
    lam = -np.inf
    for j in range(len(grid)):
        risk = float(M[:, j].mean())
        if mode == "kish":
            p = hb_pvalue(risk, max(int(np.floor(ne)), 1), alpha)
        else:
            p = 1.0 if risk >= alpha else float(np.exp(-2.0 * ne * (alpha - risk) ** 2))
        if p > delta:
            break
        lam = float(grid[j])
    return lam, ne


def group_halves(pool, grp, reps, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(grp)
    idx = {g: np.flatnonzero(grp == g) for g in ids}
    for _ in range(reps):
        order = rng.permutation(ids)
        cal_i, tot = [], 0
        for g in order:
            if tot >= len(pool) // 2:
                break
            cal_i += idx[g].tolist()
            tot += len(idx[g])
        mask = np.zeros(len(pool), bool)
        mask[cal_i] = True
        yield mask


def describe(a):
    out = {}
    for eps in a.eps:
        cal, test, pool, grp = build_groups(eps)
        ne = [n_eff_of(grp[m]) for m in group_halves(pool, grp, 200)]
        sizes = np.bincount(grp)
        out[str(eps)] = dict(n=len(pool), n_groups=int(len(sizes)), largest=int(sizes.max()), n_eff_calibration_half=dict(
            mean=float(np.mean(ne)), median=float(np.median(ne)), p05=float(np.quantile(ne, .05)), p95=float(np.quantile(ne, .95)), min=float(np.min(ne))))
        print(eps, out[str(eps)], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


def run(a):
    from review_fixes import wilson_lower
    res = {}
    for eps in a.eps:
        cal0, test0, pool, grp = build_groups(eps)
        row = {}
        for variant in ("plain", "kish", "hoeffding"):
            miss, stop, refused, neffs = [], [], 0, []
            for mask in group_halves(pool, grp, a.reps):
                cal = [s for s, m in zip(pool, mask) if m]
                test = [s for s, m in zip(pool, mask) if not m]
                if variant == "plain":
                    from evaluate import ltt
                    lam, ne = ltt(cal, "geom", ALPHA, DELTA), float(len(cal))
                else:
                    lam, ne = ltt_cluster(cal, grp[mask], "geom", mode=variant)
                neffs.append(ne)
                if not np.isfinite(lam):
                    refused += 1
                    miss.append(np.nan)
                    stop.append(np.nan)
                    continue
                s_ = summarize(test, "geom", lam)
                miss.append(s_["miss"])
                stop.append(s_["unnecessary_stop"])
            m = np.array(miss)
            ok = ~np.isnan(m)
            nt = len(pool) - len(pool) // 2
            unc = float(np.mean(m[ok] > ALPHA)) if ok.any() else None
            cor = float(np.mean([wilson_lower(x, nt) > ALPHA for x in m[ok]])) if ok.any() else None
            row[variant] = dict(uncorrected=unc, corrected=cor, refused=int(refused), n_resplits=int(ok.sum()),
                                mean_stop=float(np.nanmean(stop)) if ok.any() else None, mean_miss=float(np.nanmean(m)) if ok.any() else None,
                                mean_n_eff=float(np.mean(neffs)), stops=[None if np.isnan(x) else float(x) for x in stop])
            print(eps, variant, {k: v for k, v in row[variant].items() if k != "stops"}, flush=True)
        # random (non-clustered) half resplits: validity must not be lost
        rng = np.random.default_rng(1)
        rnd = {}
        for variant in ("plain", "kish", "hoeffding"):
            miss, refused = [], 0
            rng = np.random.default_rng(1)
            for _ in range(a.reps):
                p_ = rng.permutation(len(pool))
                mask = np.zeros(len(pool), bool)
                mask[p_[: len(pool) // 2]] = True
                cal = [s for s, m in zip(pool, mask) if m]
                test = [s for s, m in zip(pool, mask) if not m]
                if variant == "plain":
                    from evaluate import ltt
                    lam = ltt(cal, "geom", ALPHA, DELTA)
                else:
                    lam, _ = ltt_cluster(cal, grp[mask], "geom", mode=variant)
                if not np.isfinite(lam):
                    refused += 1
                    miss.append(np.nan)
                    continue
                miss.append(summarize(test, "geom", lam)["miss"])
            m = np.array(miss)
            ok = ~np.isnan(m)
            rnd[variant] = dict(uncorrected=float(np.mean(m[ok] > ALPHA)) if ok.any() else None, refused=int(refused), n_resplits=int(ok.sum()))
            print(eps, "random", variant, rnd[variant], flush=True)
        row["random_resplits"] = rnd
        # paired stop difference on resplits where both certify
        p, k = np.array([np.nan if x is None else x for x in row["plain"]["stops"]]), np.array([np.nan if x is None else x for x in row["kish"]["stops"]])
        both = ~np.isnan(p) & ~np.isnan(k)
        row["paired_stop_kish_minus_plain"] = dict(mean=float(np.mean(k[both] - p[both])) if both.any() else None, n=int(both.sum()))
        for v in ("plain", "kish", "hoeffding"):
            row[v].pop("stops")
        res[str(eps)] = row
    Path(a.out).write_text(json.dumps(dict(alpha=ALPHA, delta=DELTA, reps=a.reps, results=res), indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["describe", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[50.0, 150.0])
    ap.add_argument("--reps", type=int, default=200)
    a_ = ap.parse_args()
    {"describe": describe, "run": run}[a_.cmd](a_)
