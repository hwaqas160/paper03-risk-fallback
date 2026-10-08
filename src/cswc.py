"""
Amendment 11 / 11a (notes/falsification.md): counterfactual safe-window certification (CSWC), prevalence-corrected
certificate (N-A) and outcome audit (N-C). Everything reads stored rollouts; nothing is re-simulated.

    python src/cswc.py indist --out results/final/r11_cswc_indist.json
    python src/cswc.py na     --out results/final/r11_na_prevalence.json
    python src/cswc.py shift  --out results/final/r11_cswc_shift.json
    python src/cswc.py nc     --out results/final/r11_nc_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, lam_grid, load_rows, ltt, summarize  # noqa: E402
from review_fixes import attach_physics, paired, tri, wilson  # noqa: E402
from risk_control import hb_pvalue, learn_then_test  # noqa: E402

ALPHA, DELTA = 0.05, 0.10
T_RES, T_TOT = 0.075, 0.12
CAMP = "results/campaign/"
HGB = dict(max_iter=200, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=0, early_stopping=False)
FEATS = ["geom", "geom_cm", "conf_conflict", "conf_ahead", "ttc", "ttc_cm", "dist", "dist_cm", "n_near", "min_gap", "tick"]


# ------------------------------------------------------------------------------------------------ data
def load(arm):
    out = []
    for r in load_rows([CAMP + arm]):
        s = Scenario(r)
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
        out.append(s)
    return out


def window_first(s):
    """First safe firing tick F(X), or None (unavoidable or only 'never' is safe)."""
    safe = np.flatnonzero(~s.harm_run[: s.n_ticks])
    return int(safe[0]) if len(safe) else None


def feature_matrix(s):
    n = s.n_ticks
    cols = [s.traces["geom"], s.cummax["geom"], s.traces["conf_conflict"], s.traces["conf_ahead"], s.traces["ttc"],
            s.cummax["ttc"], s.traces["dist"], s.cummax["dist"], s.near, s.gap, np.arange(n, dtype=float)]
    X = np.stack(cols, 1)
    return np.hstack([X, np.repeat(np.asarray(s.feats, float)[None, :], n, 0)])


def labels(s, lat=0):
    n = s.n_ticks
    F = window_first(s)
    y = np.zeros(n)
    if s.ref_harmful and F is not None:
        y[max(0, F - lat):] = 1.0              # firing decided at t acts at t + lat
    return y


def fit(train, lat=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([feature_matrix(s) for s in train])
    y = np.concatenate([labels(s, lat) for s in train])
    return HistGradientBoostingClassifier(**HGB).fit(X, y)


def oof_traces(train, key="cswc_oof", folds=5, seed=0, lat=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(train))
    for f in range(folds):
        tr = [s for s, k in zip(train, fold) if k != f]
        m = fit(tr, lat)
        for s, k in zip(train, fold):
            if k == f:
                set_trace(s, key, m.predict_proba(feature_matrix(s))[:, 1])


def set_trace(s, key, v):
    s.traces[key] = np.asarray(v, float)
    s.cummax[key] = np.maximum.accumulate(s.traces[key])


def score(model, scns, key="cswc"):
    for s in scns:
        set_trace(s, key, model.predict_proba(feature_matrix(s))[:, 1])


# ------------------------------------------------------------------------------------------------ outcome certification
def tick_idx(scns, key, grid, lat=0):
    return np.stack([s.tick(key, grid, lat) for s in scns])           # (N, G)


def outcome_mat(scns, key, grid, kind="res", lat=0):
    T = tick_idx(scns, key, grid, lat)
    O = np.zeros(T.shape)
    S = np.zeros(T.shape)
    for i, s in enumerate(scns):
        h = s.harm_run[T[i]].astype(float)
        if kind == "tot":
            h = np.maximum(h, s.induced[T[i]])
        O[i] = h
        S[i] = s.stop[T[i]]
    return O, S


def certify_ordered(cal, key, grid, target, order_scns, order_key=None, kind="res", lat=0):
    """Fixed-sequence LTT along a PRE-SPECIFIED order (ascending estimated risk on `order_scns`), then the certified
    threshold with the fewest calibration unnecessary stops. Returns (lam or None, details)."""
    ok = order_key or key
    Oc, Sc = outcome_mat(cal, key, grid, kind, lat)
    Ot, _ = outcome_mat(order_scns, ok, grid, kind, lat)
    risk_order = Ot.mean(0)
    order = sorted(range(len(grid)), key=lambda j: (risk_order[j], -grid[j] if np.isfinite(grid[j]) else (-1e18 if grid[j] > 0 else 1e18)))
    n = len(cal)
    cert = []
    for j in order:
        if hb_pvalue(float(Oc[:, j].mean()), n, target) <= DELTA:
            cert.append(j)
        else:
            break
    if not cert:
        return None, dict(certified=0)
    jbest = min(cert, key=lambda j: Sc[:, j].mean())
    return float(grid[jbest]), dict(certified=len(cert), cal_risk=float(Oc[:, jbest].mean()), cal_stop=float(Sc[:, jbest].mean()))


def evaluate_at(scns, key, lam, lat=0):
    O, S = outcome_mat(scns, key, np.array([lam]), "res", lat)
    Ot, _ = outcome_mat(scns, key, np.array([lam]), "tot", lat)
    s = summarize(scns, key, lam, lat)
    return dict(o_res=float(O.mean()), o_res_ci=wilson(int(O.sum()), len(O)), o_tot=float(Ot.mean()), o_tot_ci=wilson(int(Ot.sum()), len(Ot)),
                stop=float(S.mean()), miss=s["miss"], induced=s["induced"], fired=s["fired"], n=len(scns))


# ------------------------------------------------------------------------------------------------ N-A
def bbse(src_cal, lam_pilot, tgt, B=1000, seed=0):
    """Black-box shift estimation of the target harm prevalence from the pilot alert (unlabeled target scores only)."""
    def alert(scns):
        return np.array([float(s.cummax["geom"][-1] > lam_pilot) for s in scns])
    a_s, h_s = alert(src_cal), np.array([float(s.ref_harmful) for s in src_cal])
    a_t = alert(tgt)
    rng = np.random.default_rng(seed)
    def est(a_s_, h_s_, a_t_):
        tpr = a_s_[h_s_ == 1].mean(); fpr = a_s_[h_s_ == 0].mean()
        d = tpr - fpr
        return np.nan if d <= 1e-6 else float(np.clip((a_t_.mean() - fpr) / d, 0.0, 1.0))
    pi = est(a_s, h_s, a_t)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(a_s), len(a_s)); j = rng.integers(0, len(a_t), len(a_t))
        bs.append(est(a_s[i], h_s[i], a_t[j]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return dict(pi_hat=pi, pi_up=float(np.quantile(bs, 0.95)) if len(bs) else float("nan"), pi_src=float(h_s.mean()),
                tpr=float(a_s[h_s == 1].mean()), fpr=float(a_s[h_s == 0].mean()), target_alert_rate=float(a_t.mean()))


def class_conditional_lam(src_cal, eps):
    H = [s for s in src_cal if s.ref_harmful]
    if eps >= 1.0:
        return float("inf")
    g = lam_grid(src_cal, "geom")
    M = np.stack([s.miss[s.tick("geom", g)] for s in H])
    r = learn_then_test(g, M, eps, DELTA)
    return -np.inf if r["lam_hat"] is None else float(r["lam_hat"])


def na_target(src_cal, tgt):
    lam_ltt = ltt(src_cal, "geom", ALPHA, DELTA)
    est = bbse(src_cal, lam_ltt, tgt)
    eps = min(1.0, ALPHA / est["pi_up"]) if np.isfinite(est["pi_up"]) and est["pi_up"] > 0 else 1.0
    lam_pc = class_conditional_lam(src_cal, eps)
    res = dict(bbse=est, eps=eps, true_prevalence=float(np.mean([s.ref_harmful for s in tgt])))
    res["unweighted_ltt"] = dict(lam=lam_ltt, **summarize(tgt, "geom", lam_ltt))
    res["prevalence_corrected"] = dict(lam=lam_pc, **summarize(tgt, "geom", lam_pc))
    return res


def get_targets():
    cal, cal5 = load("av2_cal"), load("av2_cal5")
    test, test5 = load("av2_test"), load("av2_test5")
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    pool = [(s, lab["av2_cal"][str(s.seed)]) for s in cal] + [(s, lab["av2_cal"][str(s.seed)]) for s in cal5] \
        + [(s, lab["av2_test"][str(s.seed)]) for s in test] + [(s, lab["av2_test"][str(s.seed)]) for s in test5]
    return cal, cal5, test, test5, pool


def na(a):
    cal, cal5, test, test5, pool = get_targets()
    out = {}
    cities = sorted({c for _, c in pool})
    for c in cities:
        src = [s for s, k in pool if k != c]
        tgt = [s for s, k in pool if k == c]
        out[c] = na_target(src, tgt)
    src = cal + cal5
    out["nuscenes"] = na_target(src, load("ns_val"))
    out["waymo"] = na_target(src, load("waymo_val"))
    rows = {}
    for k, v in out.items():
        rows[k] = dict(true_pi=v["true_prevalence"], pi_hat=v["bbse"]["pi_hat"], pi_up=v["bbse"]["pi_up"], eps=v["eps"],
                       ltt_miss=v["unweighted_ltt"]["miss"], ltt_stop=v["unweighted_ltt"]["unnecessary_stop"],
                       pc_miss=v["prevalence_corrected"]["miss"], pc_stop=v["prevalence_corrected"]["unnecessary_stop"])
        print(f"{k:14s} true pi {rows[k]['true_pi']:.3f} est {rows[k]['pi_hat']:.3f} up {rows[k]['pi_up']:.3f} eps {rows[k]['eps']:.2f} | "
              f"LTT miss {rows[k]['ltt_miss']:.3f} stop {rows[k]['ltt_stop']:.3f} | PC miss {rows[k]['pc_miss']:.3f} stop {rows[k]['pc_stop']:.3f}")
    n_ltt = sum(r["ltt_miss"] <= ALPHA for r in rows.values())
    n_pc = sum(r["pc_miss"] <= ALPHA for r in rows.values())
    verdict = dict(n_targets=len(rows), ltt_meets=n_ltt, pc_meets=n_pc, claim_holds=bool(n_pc >= 7 and n_pc > n_ltt))
    print(verdict)
    Path(a.out).write_text(json.dumps(dict(targets=out, rows=rows, verdict_NA=verdict), indent=2, default=float))


# ------------------------------------------------------------------------------------------------ CSWC in distribution
def indist(a):
    cal, cal5, test, test5, pool = get_targets()
    train = cal
    oof_traces(train)
    model = fit(train)
    for grp in (cal5, test, test5):
        score(model, grp)
    score(model, train, key="cswc")                 # in-sample scores of the training split (for reporting only)
    out = dict(n_train=len(train), n_cal=len(cal5), n_test=len(test) + len(test5), windows={})
    # --- safe-window descriptives on the training split
    F = [window_first(s) for s in train]
    out["windows"] = dict(share_ref_harmful=float(np.mean([s.ref_harmful for s in train])),
                          unavoidable=float(np.mean([f is None and s.ref_harmful for f, s in zip(F, train)])),
                          noncontiguous=float(np.mean([np.diff(np.flatnonzero(~s.harm_run[: s.n_ticks])).max(initial=1) > 1 for s in train])))
    ev = test + test5
    res = {}
    for name, key, okey in (("CSWC", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC (T0a)", "ttc", None)):
        grid = lam_grid(cal5, key)
        lam, det = certify_ordered(cal5, key, grid, T_RES, train, order_key=okey)
        lam_t, det_t = certify_ordered(cal5, key, grid, T_TOT, train, order_key=okey, kind="tot")
        entry = dict(res_target=dict(lam=lam, **det), tot_target=dict(lam=lam_t, **det_t))
        if lam is not None:
            entry["res_target"]["test"] = evaluate_at(ev, key, lam)
            entry["res_target"]["test_av2_test"] = evaluate_at(test, key, lam)
            entry["res_target"]["test_av2_test5"] = evaluate_at(test5, key, lam)
        if lam_t is not None:
            entry["tot_target"]["test"] = evaluate_at(ev, key, lam_t)
        res[name] = entry
        t = entry["res_target"].get("test")
        print(f"{name:10s} O_res<=.075: lam={lam} certified={det.get('certified')} test O_res "
              f"{t['o_res']:.3f} O_tot {t['o_tot']:.3f} stop {t['stop']:.3f} miss {t['miss']:.3f}" if t else f"{name}: refused at O_res target", flush=True)
    out["methods"] = res
    # --- claim (ii): paired stops among those meeting the target
    meeting = [k for k, e in res.items() if e["res_target"].get("test") and e["res_target"]["test"]["o_res"] <= T_RES]
    out["claim_ii"] = dict(meeting=meeting)
    if "CSWC" in meeting:
        lam_c = res["CSWC"]["res_target"]["lam"]
        _, Sc = outcome_mat(ev, "cswc", np.array([lam_c]))
        comp = {}
        for k in meeting:
            if k == "CSWC":
                continue
            key = "geom" if k == "geometric" else "ttc"
            _, Sk = outcome_mat(ev, key, np.array([res[k]["res_target"]["lam"]]))
            comp[k] = paired(Sc[:, 0], Sk[:, 0])
        out["claim_ii"]["stop_diff_cswc_minus"] = comp
        out["claim_ii"]["cswc_fewer_stops_than_all"] = bool(comp and all(v["ci"][1] < 0 for v in comp.values()))
    else:
        out["claim_ii"]["cswc_fewer_stops_than_all"] = None
    # --- claim (i): violation over resplits of the pooled calibrate + test scenarios
    pool2 = cal5 + test + test5
    rng = np.random.default_rng(0)
    n = len(pool2)
    miss_o = []
    for _ in range(a.reps):
        p = rng.permutation(n)
        c_ = [pool2[i] for i in p[: n // 2]]; t_ = [pool2[i] for i in p[n // 2:]]
        g = lam_grid(c_, "cswc")
        lam, _ = certify_ordered(c_, "cswc", g, T_RES, train, order_key="cswc_oof")
        miss_o.append(evaluate_at(t_, "cswc", lam)["o_res"] if lam is not None else np.nan)
    m = np.array(miss_o)
    from review_fixes import wilson_lower
    nt = n - n // 2
    unc = float(np.nanmean(m > T_RES))
    cor = float(np.nanmean([wilson_lower(x, nt) > T_RES for x in m[~np.isnan(m)]]))
    out["claim_i"] = dict(uncorrected=unc, corrected=cor, refused=int(np.isnan(m).sum()), n_resplits=int((~np.isnan(m)).sum()),
                          verdict="valid" if unc <= DELTA else ("indeterminate" if cor <= DELTA else "violated"),
                          realized_o_res_test=res["CSWC"]["res_target"].get("test", {}).get("o_res"))
    print("claim_i", out["claim_i"]); print("claim_ii", out["claim_ii"])
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    import pickle
    pickle.dump(None, open("results/final/.cswc_done", "wb"))


# ------------------------------------------------------------------------------------------------ CSWC shift targets
def shift(a):
    cal, cal5, test, test5, pool = get_targets()
    rows = {}
    targets = {}
    for c in sorted({k for _, k in pool}):
        src = [s for s, k in pool if k != c]
        tgt = [s for s, k in pool if k == c]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(src)); ntr = int(0.6 * len(src))
        targets[c] = ([src[i] for i in perm[:ntr]], [src[i] for i in perm[ntr:]], tgt, src)
    nsv, wmv = load("ns_val"), load("waymo_val")
    targets["nuscenes"] = (cal, cal5, nsv, cal + cal5)
    targets["waymo"] = (cal, cal5, wmv, cal + cal5)
    for name, (train, calib, tgt, src_all) in targets.items():
        model = fit(train)
        score(model, calib); score(model, tgt)
        oof_traces(train)
        grid = lam_grid(calib, "cswc")
        lam0, d0 = certify_ordered(calib, "cswc", grid, T_RES, train, order_key="cswc_oof")
        lam_ltt = ltt(src_all, "geom", ALPHA, DELTA)
        est = bbse(src_all, lam_ltt, tgt)
        pi_s = est["pi_src"]; pi_up = est["pi_up"]
        scale = min(1.0, pi_s / pi_up) if np.isfinite(pi_up) and pi_up > 0 else 1.0
        lam1, d1 = certify_ordered(calib, "cswc", grid, T_RES * scale, train, order_key="cswc_oof")
        row = dict(true_pi=float(np.mean([s.ref_harmful for s in tgt])), pi_src=pi_s, pi_up=pi_up, scale=scale,
                   uncorrected=dict(lam=lam0, **(evaluate_at(tgt, "cswc", lam0) if lam0 is not None else {})),
                   corrected=dict(lam=lam1, **(evaluate_at(tgt, "cswc", lam1) if lam1 is not None else {})))
        rows[name] = row
        u, c1 = row["uncorrected"], row["corrected"]
        print(f"{name:14s} scale {scale:.2f} | uncorrected O_res {u.get('o_res', float('nan')):.3f} stop {u.get('stop', float('nan')):.3f} | "
              f"corrected O_res {c1.get('o_res', float('nan')):.3f} stop {c1.get('stop', float('nan')):.3f}", flush=True)
    meets = lambda k: sum(1 for r in rows.values() if r[k].get("o_res") is not None and r[k]["o_res"] <= T_RES)
    verdict = dict(n_targets=len(rows), uncorrected_meets=meets("uncorrected"), corrected_meets=meets("corrected"),
                   refused_uncorrected=sum(r["uncorrected"].get("o_res") is None for r in rows.values()),
                   refused_corrected=sum(r["corrected"].get("o_res") is None for r in rows.values()))
    verdict["claim_iii"] = bool(verdict["corrected_meets"] > verdict["uncorrected_meets"])
    print(verdict)
    Path(a.out).write_text(json.dumps(dict(rows=rows, verdict_iii=verdict), indent=2, default=float))


# ------------------------------------------------------------------------------------------------ N-C
def cp_upper(k, n, delta=DELTA):
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def nc(a):
    cal, cal5, test, test5, pool = get_targets()
    lam = ltt(cal, "geom", ALPHA, DELTA)
    O, _ = outcome_mat(cal5, "geom", np.array([lam]), "res"); Ot, _ = outcome_mat(cal5, "geom", np.array([lam]), "tot")
    ub_res, ub_tot = cp_upper(int(O.sum()), len(O)), cp_upper(int(Ot.sum()), len(Ot))
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    tg = {"av2_test5 (in distribution)": test5}
    for c in sorted({lab["av2_test"][str(s.seed)] for s in test}):
        tg[f"city {c} (test rows)"] = [s for s in test if lab["av2_test"][str(s.seed)] == c]
    tg["nuscenes"], tg["waymo"] = load("ns_val"), load("waymo_val")
    rows = {}
    for k, g in tg.items():
        e = evaluate_at(g, "geom", lam)
        rows[k] = dict(n=len(g), o_res=e["o_res"], o_tot=e["o_tot"], covered_res=bool(e["o_res"] <= ub_res), covered_tot=bool(e["o_tot"] <= ub_tot))
        print(f"{k:34s} O_res {e['o_res']:.3f} (UB {ub_res:.3f}) covered {rows[k]['covered_res']} | O_tot {e['o_tot']:.3f} (UB {ub_tot:.3f}) covered {rows[k]['covered_tot']}")
    cov = float(np.mean([r["covered_res"] for r in rows.values()]))
    verdict = dict(lam=lam, cal5_res=float(O.mean()), ub_res=ub_res, cal5_tot=float(Ot.mean()), ub_tot=ub_tot, n_targets=len(rows),
                   coverage_res=cov, claim_holds=bool(cov >= 0.9))
    print(verdict)
    Path(a.out).write_text(json.dumps(dict(rows=rows, verdict_NC=verdict), indent=2, default=float))


def latency_outcome(a):
    from review_fixes import wilson_lower
    cal, cal5, test, test5, pool = get_targets()
    train = cal
    ev = test + test5
    pool2 = cal5 + test + test5
    out = {}
    for lat in (1, 2):
        oof_traces(train, lat=lat)
        model = fit(train, lat)
        for g_ in (cal5, test, test5):
            score(model, g_)
        res = {}
        for name, key, okey in (("CSWC", "cswc", "cswc_oof"), ("geometric", "geom", None), ("TTC (T0a)", "ttc", None)):
            grid = lam_grid(cal5, key)
            lam, det = certify_ordered(cal5, key, grid, T_RES, train, order_key=okey, lat=lat)
            e = dict(lam=lam, **det)
            if lam is not None:
                e["test"] = evaluate_at(ev, key, lam, lat)
            # violation over resplits (outcome loss at latency lat)
            rng = np.random.default_rng(0)
            n = len(pool2)
            mm = []
            for _ in range(a.reps):
                p = rng.permutation(n)
                c_ = [pool2[i] for i in p[: n // 2]]; t_ = [pool2[i] for i in p[n // 2:]]
                g = lam_grid(c_, key)
                l_, _ = certify_ordered(c_, key, g, T_RES, train, order_key=okey, lat=lat)
                mm.append(evaluate_at(t_, key, l_, lat)["o_res"] if l_ is not None else np.nan)
            m = np.array(mm); nt = n - n // 2
            ok = ~np.isnan(m)
            unc = float(np.mean(m[ok] > T_RES)); cor = float(np.mean([wilson_lower(x, nt) > T_RES for x in m[ok]]))
            e["validity"] = dict(uncorrected=unc, corrected=cor, refused=int((~ok).sum()), verdict="valid" if unc <= DELTA else ("indeterminate" if cor <= DELTA else "violated"))
            res[name] = e
            t = e.get("test")
            print(f"lat {lat} {name:10s} " + (f"O_res {t['o_res']:.3f} stop {t['stop']:.3f} | viol unc {unc:.3f} cor {cor:.3f} {e['validity']['verdict']}" if t else "refused"), flush=True)
        meeting = [k for k, v in res.items() if v.get("test") and v["test"]["o_res"] <= T_RES]
        comp = {}
        if "CSWC" in meeting:
            _, Sc = outcome_mat(ev, "cswc", np.array([res["CSWC"]["lam"]]), "res", lat)
            for k in meeting:
                if k == "CSWC":
                    continue
                key = "geom" if k == "geometric" else "ttc"
                _, Sk = outcome_mat(ev, key, np.array([res[k]["lam"]]), "res", lat)
                comp[k] = paired(Sc[:, 0], Sk[:, 0])
        out[str(lat)] = dict(methods=res, meeting=meeting, stop_diff_cswc_minus=comp,
                             cswc_fewer_than_all=bool("CSWC" in meeting and len(meeting) == 3 and all(v["ci"][1] < 0 for v in comp.values())))
        print(f"lat {lat}", out[str(lat)]["stop_diff_cswc_minus"], out[str(lat)]["cswc_fewer_than_all"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["indist", "na", "shift", "nc", "latency"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    {"indist": indist, "na": na, "shift": shift, "nc": nc, "latency": latency_outcome}[a.what](a)


if __name__ == "__main__":
    main()
