"""
Amendment 12, A12-6: latency crossover at 10 Hz decisions, with an RSS-style trigger. Reads the 10 Hz arms written by src/campaign10.py and the stored 2 Hz rows
(for the predictor scores, held at zero order). Nothing is re-simulated here.

    python src/hz10.py --out results/final/r12_hz10.json [--debug]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cswc as C  # noqa: E402
from evaluate import NUPLAN_HARM, Scenario, _rr, load_rows  # noqa: E402
from review_fixes import paired, tri  # noqa: E402
from risk_control import hb_pvalue  # noqa: E402

T_RES, DELTA = 0.075, 0.10
DE = 5
RHO, A_ACC, B_MIN, B_MAX = 0.5, 3.0, 4.0, 8.0          # RSS parameters fixed in Amendment 12
H_STEPS = 20                                             # supervised variant: harm within 2 s after the action takes effect
LATS = (0, 2, 3, 5)
HGB = C.HGB


def load_arm(arm10, arm2):
    old = {}
    for r in load_rows([C.CAMP + arm2]):
        s = Scenario(r)
        if "conf_conflict" in s.traces and "conf_ahead" in s.traces:
            C.attach_physics(s, r)
            tabs = r["ref"].get("agent_tables") or []
            n_ = s.n_ticks
            near, gap = np.zeros(n_), np.full(n_, 100.0)
            for i in range(min(n_, len(tabs))):
                near[i] = sum(1 for a_ in tabs[i] if a_[2] < 30.0)
                if tabs[i]:
                    gap[i] = min(a_[1] for a_ in tabs[i])
            s.near, s.gap = near, gap
            old[r["seed"]] = (s, r)
    rows = []
    for f in sorted(Path(C.CAMP + arm10).glob("part_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in d or d["seed"] not in old:
                continue
            rows.append(d)
    out, dropped = [], 0
    for d in rows:
        s2, r2 = old[d["seed"]]
        n = d["steps"]
        ttc_new, ttc_old = d["ref"]["ttc_trace"], r2["ref"]["ttc_trace"]
        m = min(len(ttc_new), len(ttc_old))
        if m == 0 or len(ttc_new) != len(ttc_old) or np.abs(np.array(ttc_new[:m]) - np.array(ttc_old[:m])).max() > 1e-3:
            dropped += 1
            continue
        out.append(build(d, s2, r2, n))
    return out, dict(rows=len(rows), kept=len(out), dropped_not_reproduced=dropped)


def outcomes(d, n):
    ref = _rr(d["ref"])
    ref_h = NUPLAN_HARM.first_harm(ref)
    ref_nonfault = any(not c["at_fault"] for c in d["ref"]["collisions"])
    harm = np.zeros(n + 1); ind = np.zeros(n + 1); stop = np.zeros(n + 1)
    for k in range(n):
        f = d["fired"][str(k)]
        h = NUPLAN_HARM.first_harm(_rr(f))
        harm[k] = h is not None
        ind[k] = any(not c["at_fault"] for c in f["collisions"]) and not ref_nonfault
        stop[k] = bool(f["triggered"]) and ref_h is None
    harm[n] = ref_h is not None; ind[n] = 0.0; stop[n] = 0.0
    return harm, ind, stop, ref_h


def build(d, s2, r2, n):
    harm, ind, stop, ref_h = outcomes(d, n)
    ttc = np.array(d["ref"]["ttc_trace"], float)
    u_ttc = np.maximum(0.0, 3.0 - np.minimum(ttc, 3.0))
    kin = np.array(d["kin"], float)                       # ego v, lead gap, lead speed per step
    v_r, gap, v_f = kin[:, 0], kin[:, 1], kin[:, 2]
    dmin = v_r * RHO + 0.5 * A_ACC * RHO ** 2 + (v_r + RHO * A_ACC) ** 2 / (2 * B_MIN) - v_f ** 2 / (2 * B_MAX)
    u_rss = np.where(gap >= 99.9, -100.0, dmin - gap)
    idx = np.minimum(np.arange(n) // DE, s2.n_ticks - 1)
    hold = lambda x: np.asarray(x, float)[idx]
    cols = {"geom": hold(s2.traces["geom"]), "conf_conflict": hold(s2.traces["conf_conflict"]), "conf_ahead": hold(s2.traces["conf_ahead"]),
            "dist": hold(s2.traces["dist"]), "near": hold(s2.near), "gap": hold(s2.gap), "ttc": u_ttc, "rss": u_rss}
    feats = np.stack([cols["geom"], np.maximum.accumulate(cols["geom"]), cols["conf_conflict"], cols["conf_ahead"], u_ttc, np.maximum.accumulate(u_ttc),
                      cols["dist"], np.maximum.accumulate(cols["dist"]), cols["near"], cols["gap"], u_rss, np.maximum.accumulate(u_rss), np.arange(n, dtype=float)], 1)
    feats = np.hstack([feats, np.repeat(np.asarray(s2.feats, float)[None, :], n, 0)])
    safe = np.flatnonzero(harm[:n] == 0)
    first_safe = int(safe[0]) if len(safe) else None
    return dict(seed=d["seed"], n=n, harm=harm, ind=ind, stop=stop, ref_h=ref_h, cols=cols, feats=feats, first_safe=first_safe,
                cm={k: np.maximum.accumulate(v) for k, v in cols.items()})


def y_cswc(s, lat):
    n = s["n"]; y = np.zeros(n)
    if s["ref_h"] is not None and s["first_safe"] is not None:
        y[max(0, s["first_safe"] - lat):] = 1.0
    return y


def y_sup(s, lat):
    n = s["n"]; y = np.zeros(n)
    if s["ref_h"] is not None:
        y[(np.arange(n) + lat + H_STEPS) >= s["ref_h"]] = 1.0
    return y


def fire_idx(s, trace_cm, lam, lat):
    over = np.flatnonzero(trace_cm > lam)
    return min(int(over[0]) + lat, s["n"]) if len(over) else s["n"]


def mats(scns, key, grid, lat):
    O = np.zeros((len(scns), len(grid))); S = np.zeros_like(O)
    for i, s in enumerate(scns):
        cm = s["cm"][key] if key in s["cm"] else s["learned_cm"][key]
        for j, lam in enumerate(grid):
            t = fire_idx(s, cm, lam, lat)
            O[i, j] = s["harm"][t]; S[i, j] = s["stop"][t]
    return O, S


def qgrid(scns, key, size=200):
    v = np.concatenate([(s["cm"][key] if key in s["cm"] else s["learned_cm"][key]) for s in scns])
    q = np.unique(np.quantile(v, np.linspace(0, 1, size)))
    return np.concatenate([[q[0] - 1e-6], q, [np.inf]])


def certify(calib, key, order_scns, order_key, lat, target):
    grid = qgrid(calib, key)
    Oc, Sc = mats(calib, key, grid, lat)
    Ot, _ = mats(order_scns, order_key, grid, lat)
    risk = Ot.mean(0)
    order = sorted(range(len(grid)), key=lambda j: (risk[j], -grid[j] if np.isfinite(grid[j]) else -1e18))
    cert = []
    for j in order:
        if hb_pvalue(float(Oc[:, j].mean()), len(calib), target) <= DELTA:
            cert.append(j)
        else:
            break
    if not cert:
        return None
    return float(grid[min(cert, key=lambda j: Sc[:, j].mean())])


def fit_learned(train, ylab, lat, folds=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    fm = lambda ss: (np.vstack([s["feats"] for s in ss]), np.concatenate([ylab(s, lat) for s in ss]))
    rng = np.random.default_rng(0)
    fold = rng.integers(0, folds, size=len(train))
    for f in range(folds):
        X, y = fm([s for s, k in zip(train, fold) if k != f])
        m = HistGradientBoostingClassifier(**HGB).fit(X, y)
        for s, k in zip(train, fold):
            if k == f:
                p = m.predict_proba(s["feats"])[:, 1]
                s.setdefault("learned_oof", {})["v"] = p
    X, y = fm(train)
    return HistGradientBoostingClassifier(**HGB).fit(X, y)


def run(a):
    cal10, info_c = load_arm("av2_cal5_10hz", "av2_cal5")
    test10, info_t = load_arm("av2_test5_10hz", "av2_test5")
    print("reproducibility", info_c, info_t, flush=True)
    if a.debug:
        h = len(cal10) // 2
        cal10, test10 = cal10[: 2 * h], cal10[:h]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cal10)); h = len(cal10) // 2
    train = [cal10[i] for i in perm[:h]]; calib = [cal10[i] for i in perm[h:]]
    out = dict(reproducibility=dict(cal=info_c, test=info_t), n_train=len(train), n_calib=len(calib), n_test=len(test10), latencies_steps=LATS, results={})
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    for lat in LATS:
        res = {}
        for variant, ylab in (("learned (counterfactual labels)", y_cswc), ("learned (factual time-to-harm labels)", y_sup)):
            model = fit_learned(train, ylab, lat)
            for s in train:
                s["learned_cm"] = {"L": np.maximum.accumulate(s["learned_oof"]["v"])}
            for s in calib + test10:
                s["learned_cm"] = {"L": np.maximum.accumulate(model.predict_proba(s["feats"])[:, 1])}
            lam = certify(calib, "L", train, "L", lat, T_RES)
            e = dict(lam=lam)
            if lam is not None:
                O, S = mats(test10, "L", [lam], lat)
                e.update(o_res=float(O.mean()), stop=float(S.mean()), stops=S[:, 0])
            res[variant] = e
        for name, key in (("TTC (10 Hz)", "ttc"), ("RSS (10 Hz)", "rss"), ("geometric (2 Hz, held)", "geom")):
            lam = certify(calib, key, train, key, lat, T_RES)
            e = dict(lam=lam)
            if lam is not None:
                O, S = mats(test10, key, [lam], lat)
                e.update(o_res=float(O.mean()), stop=float(S.mean()), stops=S[:, 0])
            res[name] = e
        L = res["learned (counterfactual labels)"]
        comp = {}
        if L.get("o_res") is not None:
            for k, v in res.items():
                if k != "learned (counterfactual labels)" and v.get("o_res") is not None and v["o_res"] <= T_RES:
                    comp[k] = paired(L["stops"], v["stops"])
        for v in res.values():
            v.pop("stops", None)
        out["results"][str(lat)] = dict(methods=res, stop_diff_learned_cf_minus=comp)
        print(f"lat {lat} steps ({lat / 10:.1f} s):", {k: (round(v["o_res"], 3), round(v["stop"], 3)) if v.get("o_res") is not None else "refused" for k, v in res.items()}, flush=True)
        print("   diffs", {k: [round(x, 3) for x in (c["diff"], *c["ci"])] for k, c in comp.items()}, flush=True)
    def holds(lat):
        if str(lat) not in out["results"]:
            return None
        r = out["results"][str(lat)]
        L = r["methods"]["learned (counterfactual labels)"]
        need = ("TTC (10 Hz)", "RSS (10 Hz)", "geometric (2 Hz, held)")
        c = r["stop_diff_learned_cf_minus"]
        return bool(L.get("o_res") is not None and L["o_res"] <= T_RES and all(k in c and c[k]["ci"][1] < 0 for k in need))
    out["claim_A12_6"] = dict(lat3=holds(3), lat5=holds(5), holds=bool(holds(3) and holds(5)))
    print(out["claim_A12_6"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--lats", type=int, nargs="+", default=None, help="post hoc latencies in steps")
    a_ = ap.parse_args()
    if a_.lats:
        LATS = tuple(a_.lats)
    run(a_)
