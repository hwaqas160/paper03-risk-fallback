"""
Exact offline evaluation of every trigger, calibrator and online method from tick-sweep rows.

Why exact: a rollout's physical outcome depends only on WHEN the fallback fires
(test_fire_step_reproduces_lambda_trigger). Each campaign row stores one reference run with
every trigger's score trace, plus one rollout forced to fire at each decision tick. Any
rule "fire at the first tick where score_key > lam (+ latency)" is therefore a lookup, so
every method below is evaluated on identical physics, with no approximation and no extra
simulation — including online methods (CDT, ACI) whose lam changes scenario by scenario.

Methods (notes/design.md Sec. 2 and 16):
  never / always          sanity bounds
  tuned[key]              deployed practice: the most permissive lam whose EMPIRICAL
                          calibration miss rate is <= alpha. No guarantee.
                            key=conf  -> T1 (confidence threshold)
                            key=geom  -> tuned version of our score (isolates the calibrator)
  cp_open                 T3, open-loop conformal: inflate every predicted mode by the
                          split-conformal radius q of the predictor's own error on held-out
                          logs, fire when it touches the ego. Guarantee on the PREDICTION.
  ltt[key] / crc[key]     T4 (ours): Learn-then-Test / Conformal Risk Control on the
                          CLOSED-LOOP miss indicator. Guarantee on the DECISION.
  ltt_w[key]              ours + shift repair: density-ratio-weighted LTT
  cdt[key]                Conformal Decision Theory (Lekeufack et al., 2024), episode-level
                          online update lam <- lam + eta (alpha - L)
  aci                     Adaptive Conformal Inference (Gibbs & Candes, 2021) on cp_open's
                          miscoverage level
  oracle[key]             best lam in hindsight on the TEST set (not deployable; upper bound)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from risk_control import hb_pvalue, learn_then_test
from rollout import NUPLAN_HARM, HarmDef, RolloutResult


# ------------------------------------------------------------------------------ data
def _rr(d: dict) -> RolloutResult:
    keep = {k: v for k, v in d.items() if k in RolloutResult.__dataclass_fields__}
    return RolloutResult(seed=-1, lam=float("inf"), **keep)


class Scenario:
    """One scenario's exact outcome for every possible firing tick (last index = never fire)."""

    def __init__(self, row: dict, harm: HarmDef = NUPLAN_HARM):
        ref = row["ref"]
        de = row["decide_every"]
        self.seed = row["seed"]
        self.feats = np.asarray(ref.get("feats", []), float)
        self.traces = {k: np.asarray(v, float) for k, v in ref["score_traces"].items()}
        self.cummax = {k: np.maximum.accumulate(v) for k, v in self.traces.items()}
        self.n_ticks = len(next(iter(self.traces.values())))
        ref_r = _rr(ref)
        ref_harm = harm.first_harm(ref_r)
        ref_nonfault = any(not c["at_fault"] for c in ref["collisions"])
        miss, stop, ind, rc = [], [], [], []
        for i in range(self.n_ticks):
            f = row["fired"].get(str(i * de))
            if f is None:                          # tick past the episode end: never fires
                miss.append(ref_harm is not None); stop.append(False); ind.append(False)
                rc.append(ref["route_completion"])
                continue
            r = _rr(f)
            h = harm.first_harm(r)
            miss.append(h is not None and not (r.triggered and r.trigger_step < h))
            stop.append(bool(r.triggered) and ref_harm is None)
            ind.append(any(not c["at_fault"] for c in f["collisions"]) and not ref_nonfault)
            rc.append(f["route_completion"])
        miss.append(ref_harm is not None); stop.append(False); ind.append(False)
        rc.append(ref["route_completion"])
        self.miss = np.array(miss, float)
        self.stop = np.array(stop, float)
        self.induced = np.array(ind, float)
        self.rc = np.array(rc, float)
        self.ref_harmful = ref_harm is not None

    def tick(self, key: str, lams, lat: int = 0) -> np.ndarray:
        """Firing tick index for each lam (n_ticks = never)."""
        i = np.searchsorted(self.cummax[key], np.atleast_1d(lams), side="right") + lat
        return np.minimum(i, self.n_ticks)


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        # part_*.jsonl hold scenario rows; skip_*.jsonl hold static-ego / error markers
        for f in sorted(Path(p).glob("part_*.jsonl")) if Path(p).is_dir() else [Path(p)]:
            with open(f) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                 # half-written last line from an interrupted run
                    if "ref" in r:
                        rows.append(r)
    seen, out = set(), []
    for r in rows:                                 # a resumed campaign can repeat a seed
        if r["seed"] not in seen:
            seen.add(r["seed"]); out.append(r)
    return out


# ------------------------------------------------------------------------ core tables
def matrices(scns, key, grid, lat=0):
    """(n, G) miss / stop / induced / triggered / route-completion for every lam in grid."""
    T = np.stack([s.tick(key, grid, lat) for s in scns])
    M = np.stack([s.miss[t] for s, t in zip(scns, T)])
    S = np.stack([s.stop[t] for s, t in zip(scns, T)])
    I = np.stack([s.induced[t] for s, t in zip(scns, T)])
    RC = np.stack([s.rc[t] for s, t in zip(scns, T)])
    F = (T < np.array([s.n_ticks for s in scns])[:, None]).astype(float)
    return M, S, I, F, RC


def lam_grid(scns, key, size=400):
    v = np.concatenate([s.cummax[key] for s in scns])
    q = np.unique(np.quantile(v, np.linspace(0, 1, size)))
    return np.concatenate([[q[0] - 1e-6], q, [np.inf]])


def summarize(scns, key, lam, lat=0) -> dict:
    M, S, I, F, RC = matrices(scns, key, np.array([lam]), lat)
    return _summ(M[:, 0], S[:, 0], I[:, 0], F[:, 0], RC[:, 0])


def _summ(m, s, i, f, rc) -> dict:
    n = len(m)
    def ci(x):
        k = x.sum(); p = k / n; z = 1.96; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [max(0.0, c - h), min(1.0, c + h)]
    return dict(n=n, miss=float(m.mean()), miss_ci=ci(m), unnecessary_stop=float(s.mean()),
                stop_ci=ci(s), induced=float(i.mean()), fired=float(f.mean()),
                autonomy=float(1 - f.mean()), route_completion=float(rc.mean()))


# --------------------------------------------------------------------------- methods
def tuned(cal, key, alpha, grid=None):
    grid = lam_grid(cal, key) if grid is None else grid
    M = matrices(cal, key, grid)[0]
    ok = np.flatnonzero(M.mean(0) <= alpha)
    return float(grid[ok[-1]]) if len(ok) else -np.inf


def ltt(cal, key, alpha, delta, grid=None):
    grid = lam_grid(cal, key) if grid is None else grid
    M = matrices(cal, key, grid)[0]
    r = learn_then_test(grid, M, alpha, delta)
    return -np.inf if r["lam_hat"] is None else float(r["lam_hat"])


def crc(cal, key, alpha, grid=None):
    grid = lam_grid(cal, key) if grid is None else grid
    M = matrices(cal, key, grid)[0]
    n = len(cal)
    ok = (n * M.mean(0) + 1) / (n + 1) <= alpha
    first_fail = int(np.argmin(ok)) if not ok.all() else len(grid)
    return float(grid[first_fail - 1]) if first_fail > 0 else -np.inf


def ltt_weighted(cal, key, alpha, delta, w, grid=None):
    """
    Density-ratio-weighted LTT (covariate-shift repair). Risk is the w-weighted mean miss;
    the Hoeffding-Bentkus p-value uses the Kish effective sample size n_eff = (sum w)^2 /
    sum w^2, so certification visibly weakens -- and eventually refuses -- as the weights
    concentrate, instead of silently claiming the unweighted n.
    """
    grid = lam_grid(cal, key) if grid is None else grid
    M = matrices(cal, key, grid)[0]
    w = np.asarray(w, float) / np.mean(w)
    n_eff = int(np.floor(w.sum() ** 2 / (w ** 2).sum()))
    risk = (w[:, None] * M).sum(0) / w.sum()
    lam = -np.inf
    for j in range(len(grid)):
        if hb_pvalue(float(risk[j]), max(n_eff, 1), alpha) > delta:
            break
        lam = float(grid[j])
    return lam, n_eff


def cp_open_lambda(pred_err_scores, alpha):
    """T3: lam = -q, q = split-conformal radius of the predictor's open-loop error."""
    from conformal import split_conformal_quantile
    return -split_conformal_quantile(np.asarray(pred_err_scores, float), alpha)


def online(test, key, lam0, eta, alpha, order, lat=0):
    """CDT, episode-level: lam_{t+1} = lam_t + eta (alpha - L_t). Smaller lam = fire earlier."""
    lam = lam0
    m, s, i, f, rc = [], [], [], [], []
    for j in order:
        sc = test[j]
        t = sc.tick(key, [lam], lat)[0]
        m.append(sc.miss[t]); s.append(sc.stop[t]); i.append(sc.induced[t])
        f.append(float(t < sc.n_ticks)); rc.append(sc.rc[t])
        lam = lam + eta * (alpha - sc.miss[t])
    return _summ(*map(np.array, (m, s, i, f, rc)))


def aci(test, pred_err_scores, alpha, gamma, order):
    """ACI on cp_open's miscoverage level: a_{t+1} = a_t + gamma (alpha - err_t)."""
    a = alpha
    m, s, i, f, rc = [], [], [], [], []
    for j in order:
        sc = test[j]
        lam = cp_open_lambda(pred_err_scores, float(np.clip(a, 1e-3, 0.999)))
        t = sc.tick("gap", [lam])[0]
        m.append(sc.miss[t]); s.append(sc.stop[t]); i.append(sc.induced[t])
        f.append(float(t < sc.n_ticks)); rc.append(sc.rc[t])
        a = a + gamma * (alpha - sc.miss[t])
    return _summ(*map(np.array, (m, s, i, f, rc)))


def oracle(test, key, alpha):
    grid = lam_grid(test, key)
    M, S = matrices(test, key, grid)[:2]
    ok = np.flatnonzero(M.mean(0) <= alpha)
    if not len(ok):
        return -np.inf
    j = ok[np.argmin(S.mean(0)[ok])]
    return float(grid[j])


# ------------------------------------------------------------------------ experiments
def headline(cal, test, alpha, delta, pred_err=None, seed=0, etas=(0.01, 0.05, 0.1, 0.5)):
    """Every method, calibrated on `cal`, evaluated on `test`. Online methods: mean over
    20 random test orders, best step size per method (favourable to the baselines)."""
    res = {}
    res["never"] = summarize(test, "geom", np.inf)
    res["always"] = summarize(test, "geom", -np.inf)
    for key, name in (("conf", "T1 tuned confidence"), ("ens", "T2 tuned ensemble (MC dropout)"),
                      ("geom", "tuned geometric (ours, no guarantee)")):
        if key not in test[0].traces:
            continue
        res[name] = dict(lam=tuned(cal, key, alpha), **summarize(test, key, tuned(cal, key, alpha)))
    if pred_err is not None:
        lam = cp_open_lambda(pred_err, alpha)
        res["T3 open-loop conformal"] = dict(lam=lam, **summarize(test, "gap", lam))
    for key in ("geom",):
        lam = ltt(cal, key, alpha, delta)
        res["T4 LTT (ours)"] = dict(lam=lam, **summarize(test, key, lam))
        lam = crc(cal, key, alpha)
        res["T4 CRC (ours)"] = dict(lam=lam, **summarize(test, key, lam))
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(test)) for _ in range(20)]
    lam0 = tuned(cal, "geom", alpha)
    best = None
    for eta in etas:
        runs = [online(test, "geom", lam0 if np.isfinite(lam0) else 0.0, eta, alpha, o) for o in orders]
        avg = {k: float(np.mean([r[k] for r in runs])) for k in ("miss", "unnecessary_stop", "induced", "autonomy", "route_completion")}
        if best is None or (avg["miss"] <= alpha and avg["unnecessary_stop"] < best[1]["unnecessary_stop"]) \
                or (best[1]["miss"] > alpha and avg["miss"] < best[1]["miss"]):
            best = (eta, avg)
    res["CDT (Lekeufack+ 2024)"] = dict(eta=best[0], **best[1])
    if pred_err is not None:
        best = None
        for g in etas:
            runs = [aci(test, pred_err, alpha, g, o) for o in orders]
            avg = {k: float(np.mean([r[k] for r in runs])) for k in ("miss", "unnecessary_stop", "induced", "autonomy", "route_completion")}
            if best is None or abs(avg["miss"] - alpha) < abs(best[1]["miss"] - alpha):
                best = (g, avg)
        res["ACI (Gibbs+Candes 2021)"] = dict(gamma=best[0], **best[1])
    lam = oracle(test, "geom", alpha)
    res["oracle (hindsight, not deployable)"] = dict(lam=lam, **summarize(test, "geom", lam))
    return res


def validity(scns, alpha, delta, reps=200, seed=0, pred_err=None):
    """
    Guarantee validity: `reps` random half/half resplits of exchangeable scenarios. For each
    method, the distribution of realized TEST miss rates and the violation frequency
    P(test miss > alpha). A valid high-probability certificate keeps violation <= delta.
    """
    rng = np.random.default_rng(seed)
    methods = {
        "T1 tuned confidence": lambda c: ("conf", tuned(c, "conf", alpha)),
        "tuned geometric": lambda c: ("geom", tuned(c, "geom", alpha)),
        "T4 LTT (ours)": lambda c: ("geom", ltt(c, "geom", alpha, delta)),
        "T4 CRC (ours)": lambda c: ("geom", crc(c, "geom", alpha)),
    }
    if pred_err is not None:
        methods["T3 open-loop conformal"] = lambda c: ("gap", cp_open_lambda(pred_err, alpha))
    out = {k: [] for k in methods}
    n = len(scns)
    for _ in range(reps):
        p = rng.permutation(n)
        cal = [scns[i] for i in p[: n // 2]]
        test = [scns[i] for i in p[n // 2:]]
        for name, f in methods.items():
            key, lam = f(cal)
            s = summarize(test, key, lam)
            out[name].append((s["miss"], s["unnecessary_stop"]))
    return {k: dict(violation_freq=float(np.mean([m > alpha for m, _ in v])),
                    miss_mean=float(np.mean([m for m, _ in v])),
                    miss_q95=float(np.quantile([m for m, _ in v], 0.95)),
                    stop_mean=float(np.mean([s for _, s in v])))
            for k, v in out.items()}


def ablations(cal, test, alpha, delta, lats=(0, 1, 2)):
    """Each design choice switched off in turn, all certified by LTT on the same scenarios."""
    res = {}
    for key, name in (("geom", "full (kinematic corridor + length-aware box)"),
                      ("geom_route", "- route-following corridor instead of kinematic"),
                      ("geom_iso", "- width-only isotropic radius instead of box"),
                      ("conf", "- confidence score instead of geometry"),
                      ("ens", "- ensemble spread instead of geometry")):
        if key not in test[0].traces:
            continue
        lam = ltt(cal, key, alpha, delta)
        res[f"score: {name}"] = dict(lam=lam, **summarize(test, key, lam))
    for name, fn in (("LTT", lambda: ltt(cal, "geom", alpha, delta)),
                     ("CRC", lambda: crc(cal, "geom", alpha)),
                     ("empirical tuning (no certificate)", lambda: tuned(cal, "geom", alpha))):
        lam = fn()
        res[f"calibrator: {name}"] = dict(lam=lam, **summarize(test, "geom", lam))
    for lat in lats:
        lam = ltt(cal, "geom", alpha, delta)
        res[f"latency: +{lat} tick(s)"] = dict(lam=lam, **summarize(test, "geom", lam, lat))
    return res


def h1_rankings(cal, test, alpha, delta, keys=("conf", "ens", "gap", "geom", "geom_route", "geom_iso")):
    """
    H1: does open-loop evaluation rank uncertainty scores the same way closed-loop does?
      open-loop  = AUROC of the scenario's peak score for predicting harm in the no-fallback
                   run (the failure-detection metric open-loop work reports, e.g. Farid et al.)
      closed-loop = unnecessary-stop rate on test at the LTT-certified threshold (lower = better)
    Reports both rankings and Kendall's tau between them.
    """
    from scipy.stats import kendalltau
    keys = [k for k in keys if k in test[0].traces]
    y = np.array([s.ref_harmful for s in test], float)
    auroc, stops = {}, {}
    for k in keys:
        peak = np.array([s.cummax[k][-1] for s in test])
        pos, neg = peak[y == 1], peak[y == 0]
        auroc[k] = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")
        lam = ltt(cal, k, alpha, delta)
        stops[k] = summarize(test, k, lam)["unnecessary_stop"]
    open_rank = sorted(keys, key=lambda k: -auroc[k])
    closed_rank = sorted(keys, key=lambda k: stops[k])
    tau = kendalltau([open_rank.index(k) for k in keys], [closed_rank.index(k) for k in keys]).statistic
    return dict(auroc=auroc, closed_loop_unnecessary_stop=stops, open_loop_rank=open_rank,
                closed_loop_rank=closed_rank, kendall_tau=float(tau),
                same_top=open_rank[0] == closed_rank[0])


def shift(cal, target, alpha, delta):
    """H3 + repair: certify on source, deploy on target; unweighted vs density-ratio-weighted."""
    from conformal import domain_classifier_weights
    lam = ltt(cal, "geom", alpha, delta)
    w = domain_classifier_weights(np.stack([s.feats for s in cal]), np.stack([s.feats for s in target]))
    lam_w, n_eff = ltt_weighted(cal, "geom", alpha, delta, w)
    return {"unweighted LTT": dict(lam=lam, **summarize(target, "geom", lam)),
            "weighted LTT (repair)": dict(lam=lam_w, n_eff=n_eff, **summarize(target, "geom", lam_w)),
            "oracle on target": dict(lam=oracle(target, "geom", alpha),
                                     **summarize(target, "geom", oracle(target, "geom", alpha)))}


def pred_error_scores(npz_path, horizon=30):
    """Open-loop nonconformity (min-mode worst-step L2 error over `horizon` steps) on held-out logs."""
    from conformal import nonconformity_scores
    z = np.load(npz_path)
    return nonconformity_scores(z["pred_trajs"][:, :, :horizon], z["gt"][:, :horizon],
                                z["gt_mask"][:, :horizon].astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--shift", nargs="+", default=None, help="target-domain rows for H3")
    ap.add_argument("--pred_err", default=None, help="npz of open-loop predictions on held-out logs (T3)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()

    cal = [Scenario(r) for r in load_rows(a.cal)]
    test = [Scenario(r) for r in load_rows(a.test)]
    pe = pred_error_scores(a.pred_err) if a.pred_err else None
    out = dict(alpha=a.alpha, delta=a.delta, n_cal=len(cal), n_test=len(test))
    out["headline"] = headline(cal, test, a.alpha, a.delta, pe)
    out["ablations"] = ablations(cal, test, a.alpha, a.delta)
    out["validity"] = validity(cal + test, a.alpha, a.delta, a.reps, pred_err=pe)
    out["h1"] = h1_rankings(cal, test, a.alpha, a.delta)
    if a.shift:
        out["shift"] = shift(cal, [Scenario(r) for r in load_rows(a.shift)], a.alpha, a.delta)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for sec in ("headline", "ablations", "shift"):
        if sec not in out:
            continue
        print(f"\n== {sec} (alpha={a.alpha}, n_cal={len(cal)}, n_test={len(test)}) ==")
        print(f"{'method':<52} {'miss':>6} {'unnec.stop':>10} {'induced':>8} {'autonomy':>8}")
        for k, v in out[sec].items():
            print(f"{k:<52} {v['miss']:>6.3f} {v['unnecessary_stop']:>10.3f} {v['induced']:>8.3f} {v['autonomy']:>8.3f}")
    print(f"\n== validity over {a.reps} resplits: P(test miss > alpha) must be <= delta={a.delta} for a certificate ==")
    for k, v in out["validity"].items():
        print(f"{k:<52} violation={v['violation_freq']:.3f}  miss mean={v['miss_mean']:.3f} q95={v['miss_q95']:.3f}")
    h = out["h1"]
    print(f"\n== H1: open-loop (AUROC) vs closed-loop (unnecessary stops at certified lam) ==")
    for k in h["auroc"]:
        print(f"  {k:<12} AUROC={h['auroc'][k]:.3f}  closed-loop unnec.stop={h['closed_loop_unnecessary_stop'][k]:.3f}")
    print(f"  open-loop rank {h['open_loop_rank']}\n  closed-loop rank {h['closed_loop_rank']}\n"
          f"  Kendall tau={h['kendall_tau']:.2f}  same top={h['same_top']}  "
          f"(pre-registered: H1 refuted if tau >= 0.8 AND same top)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
