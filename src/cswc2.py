"""
Amendment 12 (notes/falsification.md): supervised baseline (A12-1), target sweep (A12-2), collision-only harm (A12-3),
joint certification (A12-4), multiplicity (A12-5). Stored rollouts only.

    python src/cswc2.py sup     --out results/final/r12_sup.json
    python src/cswc2.py sweep   --out results/final/r12_sweep.json
    python src/cswc2.py collonly --out results/final/r12_collonly.json
    python src/cswc2.py joint   --out results/final/r12_joint.json
    python src/cswc2.py bonf    --out results/final/r12_bonf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cswc as C  # noqa: E402
from cswc import (CAMP, T_RES, DELTA, evaluate_at, fm, get_targets, labels, lam_grid, outcome_mat, set_trace, stops_of)  # noqa: E402
from evaluate import NUPLAN_HARM, Scenario, _rr, load_rows  # noqa: E402
from review_fixes import attach_physics, paired  # noqa: E402
from risk_control import hb_pvalue  # noqa: E402

DE = 5
H_AHEAD = 4        # supervised baseline: harm within 4 ticks (2 s) after the action takes effect


def attach_htick(scns, arm):
    seeds = {r["seed"]: r for r in load_rows([CAMP + arm])}
    for s in scns:
        h = NUPLAN_HARM.first_harm(_rr(seeds[s.seed]["ref"]))
        s.h_step = None if h is None else int(h)


def labels_sup(s, lat=0):
    """Factual time-to-harm label: the no-fallback run is harmful and its first harm falls within H_AHEAD ticks after the action takes effect."""
    n = s.n_ticks
    y = np.zeros(n)
    if getattr(s, "h_step", None) is not None:
        t = np.arange(n)
        y[(t + lat + H_AHEAD) * DE >= s.h_step] = 1.0
    return y


def fit_lab(train, label_fn, lat, cols=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([fm(s, cols) for s in train])
    y = np.concatenate([label_fn(s, lat) for s in train])
    return HistGradientBoostingClassifier(**C.HGB).fit(X, y)


def oof_lab(train, label_fn, lat, key, folds=5, seed=0, cols=None):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(train))
    for f in range(folds):
        m = fit_lab([s for s, k in zip(train, fold) if k != f], label_fn, lat, cols)
        for s, k in zip(train, fold):
            if k == f:
                set_trace(s, key, m.predict_proba(fm(s, cols))[:, 1])


def run_lab(train, calib, tests, label_fn, lat, tag, target, kind="res"):
    oof_lab(train, label_fn, lat, tag + "_oof")
    m = fit_lab(train, label_fn, lat)
    for g in [calib] + list(tests):
        for s in g:
            set_trace(s, tag, m.predict_proba(fm(s))[:, 1])
    grid = lam_grid(calib, tag)
    return C.certify_ordered(calib, tag, grid, target, train, order_key=tag + "_oof", kind=kind, lat=lat)


def load_h(arm, harm=None):
    out = []
    for r in load_rows([CAMP + arm]):
        s = Scenario(r) if harm is None else Scenario(r, harm)
        if "conf_conflict" not in s.traces or "conf_ahead" not in s.traces:
            continue
        attach_physics(s, r)
        tabs = r["ref"].get("agent_tables") or []
        n = s.n_ticks
        near, gap = np.zeros(n), np.full(n, 100.0)
        for i in range(min(n, len(tabs))):
            near[i] = sum(1 for a in tabs[i] if a[2] < 30.0)
            if tabs[i]:
                gap[i] = min(a[1] for a in tabs[i])
        s.near, s.gap = near, gap
        h = (harm or NUPLAN_HARM).first_harm(_rr(r["ref"]))
        s.h_step = None if h is None else int(h)
        out.append(s)
    return out


def sets(harm=None):
    return load_h("av2_cal", harm), load_h("av2_cal5", harm), load_h("av2_test", harm), load_h("av2_test5", harm)


def stops_cmp(ev, key_a, lam_a, key_b, lam_b, lat):
    return paired(stops_of(ev, key_a, lam_a, lat), stops_of(ev, key_b, lam_b, lat))


# ------------------------------------------------------------------------------------------------ A12-1
def sup(a):
    cal, cal5, test, test5 = sets()
    ev = test + test5
    out = {}
    for lat in (0, 1, 2):
        res = {}
        for name, fn in (("CSWC", labels), ("supervised time-to-harm", labels_sup)):
            tag = "k_" + name[:3]
            lam, det = run_lab(cal, cal5, [test, test5], fn, lat, tag, T_RES)
            res[name] = dict(lam=lam, tag=tag, **det, **({"test": evaluate_at(ev, tag, lam, lat)} if lam is not None else {}))
            t = res[name].get("test")
            print(f"lat {lat} {name:25s}", f"O_res {t['o_res']:.3f} stop {t['stop']:.3f}" if t else "refused", flush=True)
        ok = all(res[n].get("test") and res[n]["test"]["o_res"] <= T_RES for n in res)
        d = stops_cmp(ev, res["CSWC"]["tag"], res["CSWC"]["lam"], res["supervised time-to-harm"]["tag"], res["supervised time-to-harm"]["lam"], lat) if ok else None
        out[str(lat)] = dict(methods=res, both_meet_target=ok, stop_diff_cswc_minus_sup=d)
        print("  diff", d, flush=True)
    out["claim_A12_1"] = bool(all(out[l]["stop_diff_cswc_minus_sup"] and out[l]["stop_diff_cswc_minus_sup"]["ci"][1] < 0 for l in ("1", "2")))
    print("claim_A12_1", out["claim_A12_1"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------------------------------------ A12-2
def sweep(a):
    cal, cal5, test, test5 = sets()
    ev = test + test5
    targets = (0.05, 0.06, 0.075, 0.09, 0.10)
    out = {}
    for lat in (0, 1, 2):
        C.oof_traces(cal, lat=lat)
        model = C.fit(cal, lat)
        for g in (cal5, test, test5):
            C.score(model, g)
        for name, key, okey in (("CSWC", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC", "ttc", None)):
            grid = lam_grid(cal5, key)
            for tg in targets:
                lam, det = C.certify_ordered(cal5, key, grid, tg, cal, order_key=okey, lat=lat)
                e = evaluate_at(ev, key, lam, lat) if lam is not None else None
                out[f"{lat}|{name}|{tg}"] = dict(certified=lam is not None, o_res=None if e is None else e["o_res"], stop=None if e is None else e["stop"])
                print(f"lat {lat} {name:9s} target {tg}: " + (f"O {e['o_res']:.3f} stop {e['stop']:.3f}" if e else "refused"), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------------------------------------ A12-3
def collonly(a):
    from rollout import HarmDef
    harm = HarmDef(use_ttc=False)
    cal, cal5, test, test5 = sets(harm)
    ev = test + test5
    tg = 0.02
    out = {"base_rate_cal": float(np.mean([s.ref_harmful for s in cal])), "target": tg}
    for lat in (0, 1):
        res = {}
        lam, det = run_lab(cal, cal5, [test, test5], labels, lat, "c_cswc", tg)
        res["CSWC"] = dict(lam=lam, tag="c_cswc", **det, **({"test": evaluate_at(ev, "c_cswc", lam, lat)} if lam is not None else {}))
        for name, key in (("geometric", "geom"), ("TTC", "ttc")):
            grid = lam_grid(cal5, key)
            l_, d_ = C.certify_ordered(cal5, key, grid, tg, cal, lat=lat)
            res[name] = dict(lam=l_, tag=key, **d_, **({"test": evaluate_at(ev, key, l_, lat)} if l_ is not None else {}))
        comp = {}
        if res["CSWC"].get("test"):
            for name in ("geometric", "TTC"):
                if res[name].get("test"):
                    comp[name] = stops_cmp(ev, "c_cswc", res["CSWC"]["lam"], res[name]["tag"], res[name]["lam"], lat)
        ok = bool(res["CSWC"].get("test") and res["CSWC"]["test"]["o_res"] <= tg and len(comp) == 2 and all(v["ci"][1] < 0 for v in comp.values()))
        out[str(lat)] = dict(methods=res, stop_diff_cswc_minus=comp, cswc_meets_and_fewer_than_both=ok)
        print("lat", lat, {k: (round(v["test"]["o_res"], 4), round(v["test"]["stop"], 3)) if v.get("test") else "refused" for k, v in res.items()}, comp, ok, flush=True)
    out["claim_A12_3"] = bool(out["1"]["cswc_meets_and_fewer_than_both"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------------------------------------ A12-4
def joint_mats(scns, key, grid, lat):
    T = C.tick_idx(scns, key, grid, lat)
    H = np.stack([s.harm_run[t].astype(float) for s, t in zip(scns, T)])
    I = np.stack([s.induced[t] for s, t in zip(scns, T)])
    S = np.stack([s.stop[t] for s, t in zip(scns, T)])
    return H, I, S


def certify_joint(cal, key, grid, tH, tI, order_scns, order_key, lat):
    Hc, Ic, Sc = joint_mats(cal, key, grid, lat)
    Ho, Io, _ = joint_mats(order_scns, order_key or key, grid, lat)
    ord_risk = np.maximum(Ho.mean(0) / tH, Io.mean(0) / tI)           # fixed rule: ascending maximum normalized train risk
    order = sorted(range(len(grid)), key=lambda j: (ord_risk[j], -grid[j] if np.isfinite(grid[j]) else (-1e18 if grid[j] > 0 else 1e18)))
    n, cert = len(cal), []
    for j in order:
        p = max(hb_pvalue(float(Hc[:, j].mean()), n, tH), hb_pvalue(float(Ic[:, j].mean()), n, tI))
        if p <= DELTA:
            cert.append(j)
        else:
            break
    if not cert:
        return None
    jb = min(cert, key=lambda j: Sc[:, j].mean())
    return float(grid[jb])


def joint(a):
    cal, cal5, test, test5 = sets()
    ev = test + test5
    tH, tI = 0.075, 0.03
    out = {}
    for lat in (0, 1):
        C.oof_traces(cal, lat=lat)
        model = C.fit(cal, lat)
        for g in (cal5, test, test5):
            C.score(model, g)
        res = {}
        for name, key, okey in (("CSWC", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC", "ttc", None)):
            grid = lam_grid(cal5, key)
            lam = certify_joint(cal5, key, grid, tH, tI, cal, okey, lat)
            if lam is None:
                res[name] = dict(certified=False)
            else:
                H, I, S = joint_mats(ev, key, np.array([lam]), lat)
                res[name] = dict(certified=True, lam=lam, H=float(H.mean()), I=float(I.mean()), stop=float(S.mean()),
                                 meets=bool(H.mean() <= tH and I.mean() <= tI))
            print(f"lat {lat} {name:9s}", res[name], flush=True)
        out[str(lat)] = res
    out["claim_A12_4"] = bool(all(out[l]["CSWC"].get("meets") for l in ("0", "1")))
    print("claim_A12_4", out["claim_A12_4"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


# ------------------------------------------------------------------------------------------------ A12-5
def bonf(a):
    """Bonferroni-adjusted bootstrap intervals (level 1 - 0.05/12) for the paired stop-difference comparisons."""
    from review_fixes import ALPHA, load as rload
    m = 12
    q = 0.05 / m / 2
    def pair(x, y, B=20000, seed=0):
        d = x - y
        b = d[np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))].mean(1)
        return dict(diff=float(d.mean()), ci=[float(np.quantile(b, q)), float(np.quantile(b, 1 - q))])
    cal, cal5, test, test5 = sets()
    ev = test + test5
    out = {}
    # P1: TTC LTT vs geometric LTT alert level on av2_test
    from evaluate import ltt, matrices
    calP, testP = rload("results/campaign/av2_cal"), rload("results/campaign/av2_test")
    sg = matrices(testP, "geom", np.array([ltt(calP, "geom", ALPHA, DELTA)]))[1][:, 0]
    st = matrices(testP, "ttc", np.array([ltt(calP, "ttc", ALPHA, DELTA)]))[1][:, 0]
    out["P1 geometric minus TTC (alert level)"] = pair(sg, st)
    for lat in (0, 1, 2):
        C.oof_traces(cal, lat=lat)
        model = C.fit(cal, lat)
        for g in (cal5, test, test5):
            C.score(model, g)
        lam = {}
        for name, key, okey in (("CSWC", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC", "ttc", None)):
            grid = lam_grid(cal5, key)
            lam[name], _ = C.certify_ordered(cal5, key, grid, T_RES, cal, order_key=okey, lat=lat)
        S = {n: stops_of(ev, {"CSWC": "cswc", "geometric": "geom", "TTC": "ttc"}[n], lam[n], lat) for n in lam}
        for other in ("geometric", "TTC"):
            out[f"outcome lat {lat}: CSWC minus {other}"] = pair(S["CSWC"], S[other])
    p = Path(a.out)
    for k, v in out.items():
        print(f"{k:48s} diff {100 * v['diff']:+.1f} CI [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] excludes 0: {v['ci'][0] > 0 or v['ci'][1] < 0}")
    p.write_text(json.dumps(out, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sup", "sweep", "collonly", "joint", "bonf"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sup": sup, "sweep": sweep, "collonly": collonly, "joint": joint, "bonf": bonf}[a.what](a)


if __name__ == "__main__":
    main()
