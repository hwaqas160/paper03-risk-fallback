"""
Amendment 13 (notes/falsification.md): certifiability frontier on the 10 Hz arm (A13-1) and collision-only harm with a doubled calibration set (A13-2).
Stored rollouts only; nothing is re-simulated.

    python src/cswc3.py frontier --out results/final/r13_frontier.json
    python src/cswc3.py collonly --out results/final/r13_collonly.json
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frontier", "collonly"])
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    {"frontier": frontier, "collonly": collonly}[a_.cmd](a_)
