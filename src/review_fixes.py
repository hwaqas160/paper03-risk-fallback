"""
Amendment 11 (notes/falsification.md), phase 1: strict-review fixes computed from stored rollouts.

    python src/review_fixes.py p1 --out results/final/r11_p1_physics.json
    python src/review_fixes.py p2 --out results/final/r11_p2_validity.json
    python src/review_fixes.py p3 --out results/final/r11_p3_online.json
    python src/review_fixes.py p4 --out results/final/r11_p4_confidence.json

Nothing is re-simulated. Rules are fixed in the amendment before any of this is run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (Scenario, aci, crc, load_rows, lam_grid, ltt, matrices, online, pred_error_scores,  # noqa: E402
                      summarize, tuned)

ALPHA, DELTA = 0.05, 0.10
Z1 = 1.645                      # one-sided 95 %
DE = 5                          # decision interval in simulator steps


def load(path):
    out = []
    for r in load_rows([path]):
        s = Scenario(r)
        if "conf_conflict" not in s.traces:
            continue
        attach_physics(s, r)
        out.append(s)
    return out


def attach_physics(s, r):
    """T0a (time-to-collision) and T0b (headway distance) traces from stored per-step TTC and per-tick agent tables."""
    ref = r["ref"]
    n = s.n_ticks
    ttc = ref.get("ttc_trace", [])
    u_ttc = np.zeros(n)
    for i in range(n):
        k = i * DE
        t = ttc[k] if k < len(ttc) else 99.0
        u_ttc[i] = max(0.0, 3.0 - min(t, 3.0))
    tabs = ref.get("agent_tables") or []
    u_d = np.full(n, -100.0)
    for i in range(min(n, len(tabs))):
        d = [a[2] for a in tabs[i] if a[3] == 1]
        if d:
            u_d[i] = -min(min(d), 100.0)
    for key, v in (("ttc", u_ttc), ("dist", u_d)):
        s.traces[key] = v
        s.cummax[key] = np.maximum.accumulate(v)


def wilson_lower(p, n, z=Z1):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def paired(a, b, B=10000, seed=0):
    d = a - b
    bs = d[np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))].mean(1)
    return dict(diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])


def tri(misses, ntest):
    """uncorrected / corrected violation frequency and the tri-state verdict of Amendment 11, P2."""
    m = np.asarray(misses, float)
    ok = ~np.isnan(m)
    unc = float(np.mean(m[ok] > ALPHA))
    cor = float(np.mean([wilson_lower(x, ntest) > ALPHA for x in m[ok]]))
    verdict = "valid" if unc <= DELTA else ("indeterminate" if cor <= DELTA else "violated")
    return dict(uncorrected=unc, corrected=cor, n_resplits=int(ok.sum()), refused=int((~ok).sum()), verdict=verdict)


# ------------------------------------------------------------------------------------------------ P1
def p1(a):
    cal, test = load("results/campaign/av2_cal"), load("results/campaign/av2_test")
    methods = {"LTT geometric (learned predictor)": ("geom", ltt(cal, "geom", ALPHA, DELTA)),
               "tuned geometric": ("geom", tuned(cal, "geom", ALPHA)),
               "T0a TTC, LTT": ("ttc", ltt(cal, "ttc", ALPHA, DELTA)),
               "T0a TTC, tuned": ("ttc", tuned(cal, "ttc", ALPHA)),
               "T0b headway distance, LTT": ("dist", ltt(cal, "dist", ALPHA, DELTA)),
               "T0b headway distance, tuned": ("dist", tuned(cal, "dist", ALPHA)),
               "T1 confidence (most conflicting), tuned": ("conf_conflict", tuned(cal, "conf_conflict", ALPHA)),
               "T1 variant (nearest ahead), tuned": ("conf_ahead", tuned(cal, "conf_ahead", ALPHA))}
    out = dict(n_cal=len(cal), n_test=len(test), methods={})
    stops = {}
    for k, (key, lam) in methods.items():
        out["methods"][k] = dict(lam=lam, **summarize(test, key, lam))
        stops[k] = matrices(test, key, np.array([lam]))[1][:, 0]
    meeting = [k for k in ("T0a TTC, LTT", "T0b headway distance, LTT")
               if out["methods"][k]["miss"] <= ALPHA]
    out["claim_P1"] = dict(physics_meeting_target=meeting)
    if meeting:
        best = min(meeting, key=lambda k: out["methods"][k]["unnecessary_stop"])
        d = paired(stops["LTT geometric (learned predictor)"], stops[best])
        out["claim_P1"].update(best_physics=best, stop_diff_geom_minus_physics=d, geometric_advantage=bool(d["ci"][1] < 0))
    else:
        out["claim_P1"].update(note="no certified physics trigger met the target on test", geometric_advantage=None)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for k, v in out["methods"].items():
        print(f"{k:42s} miss {v['miss']:.3f} [{v['miss_ci'][0]:.3f},{v['miss_ci'][1]:.3f}] stop {v['unnecessary_stop']:.3f}")
    print(out["claim_P1"])


def p1b(a):
    """POST HOC (not pre-registered): P1 under the alternative harm definitions of Amendment 6, descriptive."""
    from rollout import HarmDef
    out = {}
    for name, harm, alpha in (("collision only, alpha=0.01", HarmDef(use_ttc=False), 0.01), ("TTC < 0.5 s", HarmDef(ttc_s=0.5), ALPHA),
                              ("TTC < 0.95 s (headline)", HarmDef(), ALPHA), ("TTC < 1.5 s", HarmDef(ttc_s=1.5), ALPHA)):
        def ld(path):
            res = []
            for r in load_rows([path]):
                s = Scenario(r, harm)
                if "conf_conflict" not in s.traces:
                    continue
                attach_physics(s, r); res.append(s)
            return res
        cal, test = ld("results/campaign/av2_cal"), ld("results/campaign/av2_test")
        row = {}
        for k, key in (("LTT geometric", "geom"), ("T0a TTC", "ttc"), ("T0b distance", "dist"), ("T1 confidence", "conf_conflict")):
            lam = ltt(cal, key, alpha, DELTA) if key != "conf_conflict" else tuned(cal, key, alpha)
            row[k] = dict(lam=lam, **summarize(test, key, lam)) if np.isfinite(lam) else dict(lam=lam, certified=False)
        out[name] = dict(base_rate=float(np.mean([s.ref_harmful for s in test])), methods=row)
        print(name, {k: (round(v.get("miss", float("nan")), 3), round(v.get("unnecessary_stop", float("nan")), 3)) for k, v in row.items()})
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


def p5(a):
    """TTC trigger (T0a) in the main-paper format: Table-I row, validity, latency, and shift targets (descriptive, post hoc)."""
    cal, test = load("results/campaign/av2_cal"), load("results/campaign/av2_test")
    out = {}
    for name, key, fn in (("T0a TTC, LTT", "ttc", lambda c: ltt(c, "ttc", ALPHA, DELTA)), ("T0a TTC, tuned", "ttc", lambda c: tuned(c, "ttc", ALPHA))):
        lam = fn(cal)
        out[name] = dict(lam=lam, **summarize(test, key, lam))
        out[name]["latency"] = {str(l): summarize(test, key, lam, l) for l in (1, 2)}
    # validity over the same resplits as the other methods
    pool = cal + test
    rng = np.random.default_rng(0)
    n = len(pool)
    m = {"T0a TTC, LTT": [], "T0a TTC, tuned": []}
    for _ in range(a.reps):
        p_ = rng.permutation(n)
        c_ = [pool[i] for i in p_[: n // 2]]; t_ = [pool[i] for i in p_[n // 2:]]
        for k, f in (("T0a TTC, LTT", lambda c: ltt(c, "ttc", ALPHA, DELTA)), ("T0a TTC, tuned", lambda c: tuned(c, "ttc", ALPHA))):
            lam = f(c_)
            m[k].append(summarize(t_, "ttc", lam)["miss"] if np.isfinite(lam) else np.nan)
    for k in m:
        out[k]["validity"] = tri(m[k], n - n // 2)
    # shift targets
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    cal5, test5 = load("results/campaign/av2_cal5"), load("results/campaign/av2_test5")
    rows = [(s_, lab["av2_cal"][str(s_.seed)]) for s_ in cal + cal5] + [(s_, lab["av2_test"][str(s_.seed)]) for s_ in test + test5]
    shift = {}
    for c in sorted({k for _, k in rows}):
        src = [s_ for s_, k in rows if k != c]; tgt = [s_ for s_, k in rows if k == c]
        lam = ltt(src, "ttc", ALPHA, DELTA)
        shift[c] = summarize(tgt, "ttc", lam)
    lam = ltt(cal + cal5, "ttc", ALPHA, DELTA)
    for name, arm in (("nuscenes", "results/campaign/ns_val"), ("waymo", "results/campaign/waymo_val")):
        shift[name] = summarize(load(arm), "ttc", lam)
    out["shift"] = shift
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for k in ("T0a TTC, LTT", "T0a TTC, tuned"):
        v = out[k]
        print(f"{k:16s} miss {v['miss']:.3f} stop {v['unnecessary_stop']:.3f} [{v['stop_ci'][0]:.3f},{v['stop_ci'][1]:.3f}] induced {v['induced']:.3f} auton {v['autonomy']:.3f} "
              f"valid unc {v['validity']['uncorrected']:.3f} cor {v['validity']['corrected']:.3f} {v['validity']['verdict']} | lat1 miss {v['latency']['1']['miss']:.3f} lat2 {v['latency']['2']['miss']:.3f}")
    print({c: (round(v["miss"], 3), round(v["unnecessary_stop"], 3)) for c, v in shift.items()})


# ------------------------------------------------------------------------------------------------ P2
def resplit_methods(pool, reps, seed=0, extra=None):
    """Same protocol as evaluate.validity (one permutation per repetition), returning per-resplit test miss."""
    rng = np.random.default_rng(seed)
    n = len(pool)
    names = ["tuned geometric", "LTT", "CRC", "T1 tuned confidence"]
    miss = {k: [] for k in names}
    for _ in range(reps):
        p = rng.permutation(n)
        cal = [pool[i] for i in p[: n // 2]]
        test = [pool[i] for i in p[n // 2:]]
        for name, key, fn in (("tuned geometric", "geom", lambda c: tuned(c, "geom", ALPHA)),
                              ("LTT", "geom", lambda c: ltt(c, "geom", ALPHA, DELTA)),
                              ("CRC", "geom", lambda c: crc(c, "geom", ALPHA)),
                              ("T1 tuned confidence", "conf_conflict", lambda c: tuned(c, "conf_conflict", ALPHA))):
            lam = fn(cal)
            miss[name].append(summarize(test, key, lam)["miss"] if np.isfinite(lam) else np.nan)
        if extra:
            for name, f in extra.items():
                miss.setdefault(name, []).append(f(cal, test))
    return miss, n - n // 2


def lat_miss(l):
    def f(cal, test):
        g = lam_grid(cal, "geom")
        from risk_control import learn_then_test
        r = learn_then_test(g, matrices(cal, "geom", g, l)[0], ALPHA, DELTA)
        lam = -np.inf if r["lam_hat"] is None else float(r["lam_hat"])
        return summarize(test, "geom", lam, l)["miss"] if np.isfinite(lam) else np.nan
    return f


def p2(a):
    out = {}
    cal, test = load("results/campaign/av2_cal"), load("results/campaign/av2_test")
    pool = cal + test
    extra = {"LTT latency-aware, 1 tick": lat_miss(1), "LTT latency-aware, 2 ticks": lat_miss(2),
             "LTT zero-latency certificate deployed at 1 tick": None}
    # naive (zero-latency certificate deployed under latency) needs the zero-latency lambda; add inline
    def naive(l):
        def f(c, t):
            lam = ltt(c, "geom", ALPHA, DELTA)
            return summarize(t, "geom", lam, l)["miss"] if np.isfinite(lam) else np.nan
        return f
    extra["LTT zero-latency certificate deployed at 1 tick"] = naive(1)
    extra["LTT zero-latency certificate deployed at 2 ticks"] = naive(2)
    miss, nt = resplit_methods(pool, a.reps, extra=extra)
    out["random_resplits_av2_pool"] = {k: tri(v, nt) for k, v in miss.items()}
    # --- fresh data, second predictor, Wayformer
    for name, c_, t_ in (("fresh held-out (AutoBot)", "av2_cal5", "av2_test5"), ("second predictor (AutoBot 0.854)", "av2_cal5_gpu", "av2_test5_gpu"),
                         ("Wayformer", "av2_cal5_way", "av2_test5_way")):
        pl = load(f"results/campaign/{c_}") + load(f"results/campaign/{t_}")
        for s in pl:
            s.traces.pop("ens", None); s.cummax.pop("ens", None)
        m, nt2 = resplit_methods(pl, a.reps)
        out[name] = {k: tri(v, nt2) for k, v in m.items()}
    # --- group-wise resplits (150 m), reuse the grouping code
    from spatial_groups import components, ego_xy
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    xy = {db: ego_xy(db, [s.seed for s in sc]) for db, sc in (("av2_cal", cal), ("av2_test", test))}
    pts = np.stack([xy["av2_cal"][s.seed] for s in cal] + [xy["av2_test"][s.seed] for s in test])
    city = np.array([lab["av2_cal"][str(s.seed)] for s in cal] + [lab["av2_test"][str(s.seed)] for s in test])
    grp = np.zeros(len(pool), int); off = 0
    for c in sorted(set(city)):
        mk = city == c
        g = components(pts[mk], 150.0); grp[mk] = g + off; off += g.max() + 1
    rng = np.random.default_rng(0)
    ids = np.unique(grp); idx = {g: np.flatnonzero(grp == g) for g in ids}
    gm, nts = [], []
    for _ in range(a.reps):
        order = rng.permutation(ids); ci, tot = [], 0
        for g in order:
            if tot >= len(pool) // 2:
                break
            ci += idx[g].tolist(); tot += len(idx[g])
        mask = np.zeros(len(pool), bool); mask[ci] = True
        c_ = [s for s, m_ in zip(pool, mask) if m_]; t_ = [s for s, m_ in zip(pool, mask) if not m_]
        lam = ltt(c_, "geom", ALPHA, DELTA)
        gm.append(summarize(t_, "geom", lam)["miss"] if np.isfinite(lam) else np.nan); nts.append(len(t_))
    out["group_resplits_150m (LTT)"] = tri(gm, int(np.mean(nts)))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for sec, d in out.items():
        print("==", sec)
        for k, v in d.items():
            print(f"  {k:52s} uncorrected {v['uncorrected']:.3f}  corrected {v['corrected']:.3f}  -> {v['verdict']}")


# ------------------------------------------------------------------------------------------------ P3
def avg(runs):
    keys = ("miss", "unnecessary_stop", "induced", "autonomy", "route_completion")
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def pick_cdt(av):
    best = None
    for e, x in av.items():
        if best is None or (x["miss"] <= ALPHA and x["unnecessary_stop"] < best[1]["unnecessary_stop"]) \
                or (best[1]["miss"] > ALPHA and x["miss"] < best[1]["miss"]):
            best = (e, x)
    return best[0]


def p3(a):
    cal0, test0 = load("results/campaign/av2_cal"), load("results/campaign/av2_test")
    pool = cal0 + test0
    pe = pred_error_scores("results/preds/av2cal_from_av2_cpu_v2.npz")
    etas = (0.01, 0.05, 0.1, 0.5)
    rng = np.random.default_rng(0)
    n = len(pool)
    res = {"CDT": [], "ACI": []}
    for rep in range(a.reps):
        p = rng.permutation(n)
        cal = [pool[i] for i in p[: n // 2]]; test = [pool[i] for i in p[n // 2:]]
        oc = [rng.permutation(len(cal)) for _ in range(a.orders)]
        ot = [rng.permutation(len(test)) for _ in range(a.orders)]
        lam0 = tuned(cal, "geom", ALPHA); lam0 = lam0 if np.isfinite(lam0) else 0.0
        av = {e: avg([online(cal, "geom", lam0, e, ALPHA, o) for o in oc]) for e in etas}
        e = pick_cdt(av)
        res["CDT"].append(avg([online(test, "geom", lam0, e, ALPHA, o) for o in ot])["miss"])
        av = {g: avg([aci(cal, pe, ALPHA, g, o) for o in oc]) for g in etas}
        g = min(av, key=lambda k: abs(av[k]["miss"] - ALPHA))
        res["ACI"].append(avg([aci(test, pe, ALPHA, g, o) for o in ot])["miss"])
        if (rep + 1) % 10 == 0:
            print(f"  {rep + 1}/{a.reps}", flush=True)
    out = {k: dict(mean_test_miss=float(np.mean(v)), **tri(v, n - n // 2)) for k, v in res.items()}
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for k, v in out.items():
        print(k, v)


# ------------------------------------------------------------------------------------------------ P4
def p4(a):
    z = np.load("results/preds/av2cal_from_av2_cpu_v2.npz")
    P, T, G, M = z["pred_probs"], z["pred_trajs"], z["gt"], z["gt_mask"].astype(bool)
    k = 29                                           # 3 s at 10 Hz
    ok = M[:, k]
    top = P.argmax(1)
    err_all = np.linalg.norm(T[:, :, k, :] - G[:, None, k, :], axis=-1)       # (N, K)
    err_top = err_all[np.arange(len(P)), top]
    err_best = err_all.min(1)
    ptop = P.max(1)
    ptop, err_top, err_best = ptop[ok], err_top[ok], err_best[ok]
    correct = (err_top < 2.0).astype(float)
    bins = np.linspace(ptop.min(), ptop.max(), 11)
    ece, rows = 0.0, []
    for i in range(10):
        m = (ptop >= bins[i]) & (ptop <= bins[i + 1] if i == 9 else ptop < bins[i + 1])
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - ptop[m].mean())
            rows.append(dict(lo=float(bins[i]), hi=float(bins[i + 1]), n=int(m.sum()), conf=float(ptop[m].mean()), acc=float(correct[m].mean())))
    from sklearn.metrics import roc_auc_score
    y = (err_best > 2.0).astype(int)
    out = dict(n=int(ok.sum()), top_mode_within_2m=float(correct.mean()), mean_top_probability=float(ptop.mean()),
               ece_top_mode=float(ece), auroc_1_minus_ptop_vs_best_mode_error_gt2m=float(roc_auc_score(y, 1 - ptop)),
               share_best_mode_error_gt2m=float(y.mean()), reliability_bins=rows)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print({k: v for k, v in out.items() if k != "reliability_bins"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["p1", "p1b", "p2", "p3", "p4", "p5"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--orders", type=int, default=5)
    a = ap.parse_args()
    if a.what == "p3" and a.reps == 200:
        a.reps = 100
    {"p1": p1, "p1b": p1b, "p2": p2, "p3": p3, "p4": p4, "p5": p5}[a.what](a)


if __name__ == "__main__":
    main()
