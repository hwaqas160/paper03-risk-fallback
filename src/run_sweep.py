"""
Closed-loop lam sweep over N scenarios -> LTT / CRC -> risk-coverage, in parallel.

Phases
  1. probe   : a held-out block of scenarios at lam = inf; the lam grid is put on quantiles
               of their per-scenario peak score (a fixed grid lands on one side of the
               decision boundary). Probe scenarios are never reused for calibration.
  2. sweep   : every calibration scenario is run under each lam plus the lam = inf
               counterfactual, sharded over W workers (commit-guarded, torch-free).
  3. analyse : losses under the headline harm definition -> LTT, CRC, risk-coverage; and,
               because every rollout keeps its TTC trace, the same rollouts are RESCORED
               under every candidate harm definition — so the author can choose the
               definition from data without re-simulating.

Traffic is IDM-REACTIVE by default. Pure log replay makes non-reactive agents drive into an
ego that slows down: on 1 000 dev scenarios it inflated collisions 11.1 % -> vs 3.7 %
reactive (results/harm_traces_av2_dev*.summary.json). For a paper about the cost of
stopping, that artifact would be fatal. `--replay` exists only to reproduce the contrast.

    python src/run_sweep.py --db av2_dev --n 300 --workers 8 --out results/sweep_dev.npz
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import simenv
from risk_control import conformal_risk_control, learn_then_test, risk_coverage_curve

STATIC_FRAC = 0.30  # ~27 % of AV2 scenarios have a static ego; over-allocate seed blocks


def _worker(job):
    """job: mode 'probe' -> list of peak scores; mode 'sweep' -> list of sweep rows."""
    os.environ["OMP_NUM_THREADS"] = "1"
    import simenv as se
    if job["predictor"] == "cv":
        se.block_torch()                      # sim-only worker: keep torch out (commit)
    from rollout import ConstantVelocityPredictor, make_fallback_policy_cls, rollout, sweep_scenario

    db, start, count, want = job["db"], job["start"], job["count"], job["want"]
    if job["predictor"] == "autobot":
        from autobot_predictor import AutoBotPredictor
        predictor = AutoBotPredictor(threads=1)
    else:
        predictor = ConstantVelocityPredictor(spread=job["spread"])
    env = se.make_env(db, start, count, policy=make_fallback_policy_cls(), **job["overrides"])
    kw = dict(decide_every=job["decide_every"], latency_steps=job["latency_steps"], score_key=job["score_key"])
    out, seed = [], start
    try:
        while len(out) < want and seed < start + count:
            try:
                env.reset(seed=seed)
                if not se.ego_is_static(env):
                    if job["mode"] == "probe":
                        r = rollout(env, seed, float("inf"), predictor, **kw)
                        out.append(max(r.scores) if r.scores else 0.0)
                    else:
                        out.append(sweep_scenario(env, seed, job["lams"], predictor, **kw))
            except Exception as e:  # noqa: BLE001
                print(f"  [w{start}] seed {seed} failed: {e!r}", flush=True)
                env.close()
                env = se.make_env(db, start, count, policy=make_fallback_policy_cls(), **job["overrides"])
            seed += 1
    finally:
        env.close()
    return out


def _pool_run(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_worker, jobs))


def _blocks(start, end, workers, want_total, base):
    """Split [start, end) into `workers` contiguous blocks, each asked for its share."""
    per = math.ceil(want_total / workers)
    size = (end - start) // workers
    return [dict(base, start=start + i * size, count=size, want=per) for i in range(workers)]


def rescore(rows, lams, harm):
    """Loss / stop / triggered matrices under any HarmDef, from stored rollouts (no re-sim)."""
    from rollout import RolloutResult, harmful, missed
    L, S, T = [], [], []
    for row in rows:
        cf = RolloutResult(**row["counterfactual"])
        rs = [RolloutResult(**d) for d in row["rollouts"]]
        L.append([float(missed(r, harm)) for r in rs])
        S.append([float(r.triggered and not harmful(cf, harm)) for r in rs])
        T.append([float(r.triggered) for r in rs])
    return np.array(L), np.array(S), np.array(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="av2_dev", choices=list(simenv.DBS))  # dev by default; test only for pre-registered evals
    ap.add_argument("--n", type=int, default=60, help="calibration scenarios (non-static)")
    ap.add_argument("--n_probe", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lams", type=float, nargs="+", default=None)
    ap.add_argument("--n_lams", type=int, default=9)
    ap.add_argument("--predictor", choices=["cv", "autobot"], default="cv",
                    help="cv = constant-velocity stub; autobot = Paper 01 checkpoint, online")
    ap.add_argument("--score", choices=list(__import__("rollout").SCORE_FNS), default="geom",
                    help="geom = T3/T4 plan-intersection family; conf = T1 fixed-confidence baseline")
    ap.add_argument("--spread", type=float, default=3.0, help="cv stub only")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--decide_every", type=int, default=1)
    ap.add_argument("--latency_steps", type=int, default=0)
    ap.add_argument("--replay", action="store_true", help="pure log replay (NOT recommended)")
    ap.add_argument("--harm_ttc", type=float, default=0.95, help="headline harm: TTC threshold (s); nuPlan = 0.95")
    ap.add_argument("--harm_sustain", type=int, default=1, help="headline harm: consecutive steps; nuPlan = 1")
    ap.add_argument("--out", default="results/sweep.npz")
    args = ap.parse_args()

    from bench_throughput import affordable_workers
    from harm_base_rate import CANDIDATES
    from rollout import HarmDef
    from scenarionet.common_utils import read_dataset_summary

    w = affordable_workers(args.workers, 2.8 if args.predictor == "autobot" else 1.8)
    if w < 1:
        raise SystemExit("not enough free commit for even one worker")
    total = len(read_dataset_summary(simenv.DBS[args.db])[1])
    base = dict(db=args.db, predictor=args.predictor, spread=args.spread, decide_every=args.decide_every,
                latency_steps=args.latency_steps, score_key=args.score,
                overrides=dict(reactive_traffic=not args.replay))
    t0 = time.time()

    # ---- phase 1: probe block, held out --------------------------------------------------
    probe_span = math.ceil(args.n_probe / (1 - STATIC_FRAC) * 1.3)
    if args.lams is None:
        pw = min(w, max(1, args.n_probe // 5))
        peaks = [p for part in _pool_run(_blocks(0, probe_span, pw, args.n_probe,
                                                 dict(base, mode="probe", lams=None)), pw) for p in part]
        lams = np.unique(np.round(np.quantile(peaks, np.linspace(0.05, 0.95, args.n_lams)), 3))
        print(f"probe: {len(peaks)} scenarios, peak score min={min(peaks):.2f} "
              f"median={np.median(peaks):.2f} max={max(peaks):.2f}")
    else:
        lams = np.asarray(sorted(args.lams), float)
    print(f"lam grid: {lams.tolist()}  ({'REPLAY' if args.replay else 'reactive'} traffic, {w} workers)", flush=True)

    # ---- phase 2: calibration sweep --------------------------------------------------------
    span = math.ceil(args.n / (1 - STATIC_FRAC) * 1.3)
    stop_at = min(total, probe_span + span)
    parts = _pool_run(_blocks(probe_span, stop_at, w, args.n, dict(base, mode="sweep", lams=lams.tolist())), w)
    rows = [r for part in parts for r in part][: args.n]
    el = time.time() - t0
    print(f"swept {len(rows)} scenarios x {len(lams) + 1} rollouts in {el / 60:.1f} min "
          f"({el / max(len(rows), 1):.1f} s/scenario wall)", flush=True)
    n_div = int(sum(sum(r.get("diverged", [])) for r in rows))
    print(f"score-trace reuse: {n_div} diverged rollouts (must be 0)", flush=True)

    out = Path(args.out)
    with open(out.with_suffix(".rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- phase 3: analysis -----------------------------------------------------------------
    headline = HarmDef(ttc_s=args.harm_ttc, sustain=args.harm_sustain)
    L, S, T = rescore(rows, lams, headline)
    np.savez(out, lams=lams, seeds=np.array([r["seed"] for r in rows]), loss=L, stop=S, triggered=T,
             induced=np.array([r["induced"] for r in rows]),
             harm=np.array([headline.label()]), reactive=np.array([not args.replay]))

    print(f"\n=== {len(L)} scenarios | harm = {headline.label()} | alpha={args.alpha} delta={args.delta} "
          f"| predictor={args.predictor}" + (" (STUB — plumbing, not a paper result)" if args.predictor == "cv" else "") + " ===")
    rc = risk_coverage_curve(L, T, S)
    I = np.array([r["induced"] for r in rows])
    print(f"{'lam':>6} {'miss rate':>10} {'autonomy':>9} {'unnec.stop':>11} {'induced coll.':>14}")
    for j, lam in enumerate(lams):
        print(f"{lam:>6.2f} {rc['risk'][j]:>10.3f} {rc['autonomy'][j]:>9.3f} "
              f"{rc['unnecessary_stop'][j]:>11.3f} {I.mean(0)[j]:>14.3f}")
    ltt = learn_then_test(lams, L, args.alpha, args.delta)
    crc = conformal_risk_control(lams, L, args.alpha)
    print(f"LTT lam_hat={ltt['lam_hat']}   CRC lam_hat={crc['lam_hat']}   "
          f"non-monotone scenarios: {crc['monotone_violations']:.3f}")

    # T3 (notes/design.md Sec. 2): the SAME score family as T4, but lam chosen by a plain
    # split-conformal quantile of the calibration peak scores (Vovk et al. 2005) -- "hand-
    # tuned via conformal calibration", NOT a closed-loop decision guarantee. Peak scores
    # come from the already-computed lam=inf rollouts of THIS sweep, so no extra simulation.
    from conformal import split_conformal_quantile
    peaks = np.array([max(r["counterfactual"]["scores"]) if r["counterfactual"]["scores"] else 0.0
                      for r in rows])
    lam_t3 = split_conformal_quantile(peaks, args.alpha)
    j_t3 = int(np.argmin(np.abs(lams - lam_t3))) if np.isfinite(lam_t3) else len(lams) - 1
    print(f"T3 (plain conformal, alpha={args.alpha}) lam_hat={lam_t3:.3f}  "
          f"(nearest grid point lam={lams[j_t3]:.3f}: miss={rc['risk'][j_t3]:.3f}, "
          f"autonomy={rc['autonomy'][j_t3]:.3f}, unnec.stop={rc['unnecessary_stop'][j_t3]:.3f})")

    # every candidate harm definition, from the same rollouts
    print(f"\n{'harm definition':<34} {'no-trigger':>10} {'min miss':>9} {'miss@max-auton':>15}  LTT lam_hat @alpha={args.alpha}")
    summary = []
    for c in CANDIDATES:
        hd = HarmDef(**c)
        Lc, Sc, Tc = rescore(rows, lams, hd)
        base_rate = float(np.mean([hd.first_harm(__import__("rollout").RolloutResult(**r["counterfactual"])) is not None
                                   for r in rows]))
        lt = learn_then_test(lams, Lc, args.alpha, args.delta)
        summary.append(dict(harm=hd.label(), base_rate=base_rate, min_miss=float(Lc.mean(0).min()),
                            miss_at_max_autonomy=float(Lc.mean(0)[-1]), ltt_lam=lt["lam_hat"],
                            risk=Lc.mean(0).tolist(), unnecessary_stop=Sc.mean(0).tolist(),
                            autonomy=(1 - Tc.mean(0)).tolist()))
        print(f"{hd.label():<34} {base_rate:>10.3f} {Lc.mean(0).min():>9.3f} {Lc.mean(0)[-1]:>15.3f}  {lt['lam_hat']}")
    out.with_suffix(".summary.json").write_text(json.dumps(dict(
        db=args.db, n=len(rows), lams=lams.tolist(), reactive=not args.replay, alpha=args.alpha,
        delta=args.delta, headline=headline.label(), by_harm=summary), indent=2))
    print(f"\nsaved {out}, {out.with_suffix('.rows.jsonl').name}, {out.with_suffix('.summary.json').name}")


if __name__ == "__main__":
    main()
