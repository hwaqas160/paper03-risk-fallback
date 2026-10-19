"""
Amendment 13 (notes/falsification.md): certifiability frontier on the 10 Hz arm (A13-1) and collision-only harm with a doubled calibration set (A13-2).
Stored rollouts only; nothing is re-simulated.

    python src/cswc3.py frontier --out results/final/r13_frontier.json
    python src/cswc3.py collonly --out results/final/r13_collonly.json
    python src/cswc3.py outvalid --out results/final/r14_outvalid.json     (Amendment 14)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cswc as C  # noqa: E402
import cswc2 as C2  # noqa: E402
import hz10 as H  # noqa: E402
from risk_control import hb_pvalue  # noqa: E402

DELTA = 0.10
TARGETS = (0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.10, 0.12, 0.15)
LATS = (0, 2, 3, 5, 8, 10)                     # steps at 10 Hz: 0, 0.2, 0.3, 0.5, 0.8, 1.0 s
MAX_STOP = 0.50
SCORES = (("TTC", "ttc"), ("RSS", "rss"), ("geometric", "geom"))


def cert_multi(calib, key, train, lat, targets):
    """Same procedure as hz10.certify, evaluated for several targets from one pair of outcome matrices."""
    grid = H.qgrid(calib, key)
    Oc, Sc = H.mats(calib, key, grid, lat)
    Ot, _ = H.mats(train, key, grid, lat)
    risk = Ot.mean(0)
    order = sorted(range(len(grid)), key=lambda j: (risk[j], -grid[j] if np.isfinite(grid[j]) else -1e18))
    out = {}
    for t in targets:
        cert = []
        for j in order:
            if hb_pvalue(float(Oc[:, j].mean()), len(calib), t) <= DELTA:
                cert.append(j)
            else:
                break
        out[t] = float(grid[min(cert, key=lambda j: Sc[:, j].mean())]) if cert else None
    return out


def evaluate(test, key, lam, lat):
    if lam is None:
        return None
    O, S = H.mats(test, key, [lam], lat)
    return dict(lam=lam, o_res=float(O.mean()), stop=float(S.mean()))


def useful(e, t):
    return bool(e is not None and e["o_res"] <= t and e["stop"] <= MAX_STOP)


def split(cal10, seed):
    perm = np.random.default_rng(seed).permutation(len(cal10))
    h = len(cal10) // 2
    return [cal10[i] for i in perm[:h]], [cal10[i] for i in perm[h:]]


def frontier(a):
    cal10, info_c = H.load_arm("av2_cal5_10hz", "av2_cal5")
    test10, info_t = H.load_arm("av2_test5_10hz", "av2_test5")
    print("reproducibility", info_c, info_t, flush=True)
    train, calib = split(cal10, 0)
    out = dict(reproducibility=dict(cal=info_c, test=info_t), n_train=len(train), n_calib=len(calib), n_test=len(test10),
               targets=TARGETS, lats_steps=LATS, max_stop=MAX_STOP, cells={}, frontier={})
    for lat in LATS:
        cells = {}
        for name, key in SCORES:
            lam_t = cert_multi(calib, key, train, lat, TARGETS)
            cells[name] = {str(t): dict(evaluate(test10, key, lam_t[t], lat) or dict(lam=None), useful=useful(evaluate(test10, key, lam_t[t], lat), t)) for t in TARGETS}
        model = H.fit_learned(train, H.y_cswc, lat)
        for s in train:
            s["learned_cm"] = {"L": np.maximum.accumulate(s["learned_oof"]["v"])}
        for s in calib + test10:
            s["learned_cm"] = {"L": np.maximum.accumulate(model.predict_proba(s["feats"])[:, 1])}
        lam_t = cert_multi(calib, "L", train, lat, TARGETS)
        cells["learned"] = {str(t): dict(evaluate(test10, "L", lam_t[t], lat) or dict(lam=None), useful=useful(evaluate(test10, "L", lam_t[t], lat), t)) for t in TARGETS}
        out["cells"][str(lat)] = cells
        fr = {}
        for name, c in cells.items():
            ok = [t for t in TARGETS if c[str(t)]["useful"]]
            fr[name] = min(ok) if ok else None
        out["frontier"][str(lat)] = fr
        print(f"lat {lat / 10:.1f} s frontier:", fr, flush=True)
    inf = float("inf")
    f = lambda lat, n: (out["frontier"][str(lat)][n] if out["frontier"][str(lat)][n] is not None else inf)
    a_ok = all(f(l, "TTC") <= min(f(l, "RSS"), f(l, "geometric"), f(l, "learned")) for l in (0, 2, 3, 5))
    b_ok = all(f(l, "TTC") > min(f(l, "RSS"), f(l, "learned")) for l in (8, 10))
    out["claim_A13_1"] = dict(a=bool(a_ok), b=bool(b_ok), holds=bool(a_ok and b_ok))
    print("claim_A13_1", out["claim_A13_1"], flush=True)
    # descriptive: certifiability probability at 0.075 over 50 resplits (score-only triggers)
    prob = {str(l): {n: 0 for n, _ in SCORES} for l in LATS}
    for seed in range(int(os.environ.get('P13_SEEDS', '50'))):
        tr, ca = split(cal10, seed)
        for lat in LATS:
            for name, key in SCORES:
                lam = cert_multi(ca, key, tr, lat, (0.075,))[0.075]
                prob[str(lat)][name] += int(useful(evaluate(test10, key, lam, lat), 0.075))
        print("seed", seed, flush=True)
    out["prob_useful_at_0.075_over_50_splits"] = {l: {n: v / max(1, int(os.environ.get('P13_SEEDS', '50'))) for n, v in d.items()} for l, d in prob.items()}
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


def collonly(a):
    from rollout import HarmDef
    harm = HarmDef(use_ttc=False)
    cal, cal5, test, test5 = C2.sets(harm)
    ev = test + test5
    out = dict(base_rate_cal=float(np.mean([s.ref_harmful for s in cal])), n_calib=len(cal), n_order=len(cal5), n_test=len(ev), targets={})
    for tg in (0.02, 0.03):
        res_t = {}
        for lat in (0, 1):
            res = {}
            for name, key in (("geometric", "geom"), ("TTC", "ttc")):
                grid = C.lam_grid(cal, key)
                lam, det = C.certify_ordered(cal, key, grid, tg, cal5, lat=lat)
                res[name] = dict(lam=lam, **det, **({"test": C.evaluate_at(ev, key, lam, lat)} if lam is not None else {}))
                t = res[name].get("test")
                print(f"target {tg} lat {lat} {name:9s}", f"O_res {t['o_res']:.4f} stop {t['stop']:.3f}" if t else "refused", flush=True)
            d = None
            if all(res[n].get("test") for n in res):
                d = C2.stops_cmp(ev, "ttc", res["TTC"]["lam"], "geom", res["geometric"]["lam"], lat)
            nontriv = all(res[n].get("test") and res[n]["test"]["stop"] < 0.95 and res[n]["test"]["o_res"] <= tg for n in res)
            res_t[str(lat)] = dict(methods=res, stop_diff_ttc_minus_geom=d, both_certify_nontrivially=bool(nontriv),
                                   ttc_fewer=bool(nontriv and d is not None and d["ci"][1] < 0))
            print("   diff TTC - geometric", d, "nontrivial", nontriv, flush=True)
        out["targets"][str(tg)] = res_t
    p = out["targets"]["0.02"]
    out["claim_A13_2"] = bool(p["0"]["ttc_fewer"] and p["1"]["ttc_fewer"])
    print("claim_A13_2", out["claim_A13_2"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------------------------------------ Amendment 14
def _load_arm(arm, drop_div):
    from evaluate import load_rows
    scns = C.load(arm)
    if drop_div:
        bad = {r["seed"] for r in load_rows([C.CAMP + arm]) if r.get("n_diverged", 0) > 0}
        scns = [s for s in scns if s.seed not in bad]
    return scns


def _verdict(unc, cor):
    return "valid" if unc <= DELTA else ("indeterminate" if cor <= DELTA else "violated")


def _score_and_certify(sets_, pool_idx_resplits, groups=None, reps=200):
    """sets_ = (cal, cal5, test, test5). Returns per-score realized result and random / group resplit violation."""
    from review_fixes import wilson_lower
    cal, cal5, test, test5 = sets_
    train = cal
    C.oof_traces(train)
    model = C.fit(train)
    for g in (cal5, test, test5):
        C.score(model, g)
    ev = test + test5
    pool = cal5 + test + test5
    out = {}
    for name, key, okey in (("learned", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC", "ttc", None)):
        grid = C.lam_grid(cal5, key)
        lam, det = C.certify_ordered(cal5, key, grid, C.T_RES, train, order_key=okey)
        e = C.evaluate_at(ev, key, lam) if lam is not None else None
        entry = dict(realized=None if e is None else dict(o_res=e["o_res"], stop=e["stop"], n=e["n"]))

        def resplit_stats(masks):
            m = []
            for mask in masks:
                c_ = [s for s, k in zip(pool, mask) if k]
                t_ = [s for s, k in zip(pool, mask) if not k]
                g_ = C.lam_grid(c_, key)
                l_, _ = C.certify_ordered(c_, key, g_, C.T_RES, train, order_key=okey)
                m.append(C.evaluate_at(t_, key, l_)["o_res"] if l_ is not None else np.nan)
            m = np.array(m)
            ok = ~np.isnan(m)
            nt = int(round(np.mean([sum(1 for k in mask if not k) for mask in masks])))
            unc = float(np.mean(m[ok] > C.T_RES))
            cor = float(np.mean([wilson_lower(x, nt) > C.T_RES for x in m[ok]]))
            return dict(uncorrected=unc, corrected=cor, refused=int((~ok).sum()), n_resplits=int(ok.sum()), verdict=_verdict(unc, cor))

        n = len(pool)
        rng = np.random.default_rng(0)
        rand_masks = []
        for _ in range(reps):
            p_ = rng.permutation(n)
            mk = np.zeros(n, bool); mk[p_[: n // 2]] = True
            rand_masks.append(mk)
        entry["random_resplits"] = resplit_stats(rand_masks)
        if groups is not None:
            ids = np.unique(groups)
            idx = {g: np.flatnonzero(groups == g) for g in ids}
            rng = np.random.default_rng(0)
            gmasks = []
            for _ in range(reps):
                order = rng.permutation(ids)
                mk = np.zeros(n, bool); tot = 0
                for g in order:
                    if tot >= n // 2:
                        break
                    mk[idx[g]] = True; tot += len(idx[g])
                gmasks.append(mk)
            entry["group_resplits"] = resplit_stats(gmasks)
        out[name] = entry
        r = entry["realized"]
        print(name, "realized", None if r is None else (round(r["o_res"], 4), round(r["stop"], 3)), "random", entry["random_resplits"],
              "group", entry.get("group_resplits"), flush=True)
    return out


def outvalid(a):
    import json as _json
    from spatial_groups import components, ego_xy
    lab = _json.loads(Path("results/final/av2_city_labels.json").read_text())
    out = dict(target=C.T_RES, delta=DELTA, reps=200)
    # --- A14-1: all scenarios, with spatial groups (eps = 150 m, per city)
    sets_all = tuple(_load_arm(x, False) for x in ("av2_cal", "av2_cal5", "av2_test", "av2_test5"))
    cal, cal5, test, test5 = sets_all
    pool = cal5 + test + test5
    xy_c = ego_xy("av2_cal", [s.seed for s in cal5])
    xy_t = ego_xy("av2_test", [s.seed for s in test + test5])
    pts = np.stack([xy_c[s.seed] for s in cal5] + [xy_t[s.seed] for s in test + test5])
    city = np.array([lab["av2_cal"][str(s.seed)] for s in cal5] + [lab["av2_test"][str(s.seed)] for s in test + test5])
    grp = np.zeros(len(pool), int)
    off = 0
    for c in sorted(set(city)):
        m = city == c
        g = components(pts[m], 150.0)
        grp[m] = g + off
        off += g.max() + 1
    sizes = np.bincount(grp)
    out["groups"] = dict(n=len(pool), n_groups=int(len(sizes)), largest=int(sizes.max()), share_largest=float(sizes.max() / len(pool)))
    print("groups", out["groups"], flush=True)
    feasible = sizes.max() / len(pool) <= 0.25
    out["all"] = _score_and_certify(sets_all, None, groups=grp if feasible else None)
    out["group_analysis_feasible"] = bool(feasible)
    # --- A14-2: drift-free subset
    sets_df = tuple(_load_arm(x, True) for x in ("av2_cal", "av2_cal5", "av2_test", "av2_test5"))
    out["drift_free_sizes"] = [len(x) for x in sets_df]
    out["all_sizes"] = [len(x) for x in sets_all]
    print("sizes all", out["all_sizes"], "drift-free", out["drift_free_sizes"], flush=True)
    out["drift_free"] = _score_and_certify(sets_df, None, groups=None)
    # --- claims
    a1 = feasible and all(out["all"][k]["group_resplits"]["uncorrected"] <= DELTA for k in out["all"])
    a2 = all(out["drift_free"][k]["realized"] is not None and out["drift_free"][k]["realized"]["o_res"] <= C.T_RES
             and out["drift_free"][k]["random_resplits"]["uncorrected"] <= DELTA for k in out["drift_free"])
    out["claim_A14_1"] = bool(a1)
    out["claim_A14_2"] = bool(a2)
    print("claim_A14_1", a1, "claim_A14_2", a2, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frontier", "collonly", "outvalid"])
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    {"frontier": frontier, "collonly": collonly, "outvalid": outvalid}[a_.cmd](a_)
