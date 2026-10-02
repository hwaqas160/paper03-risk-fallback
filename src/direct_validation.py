"""
Amendment 7 (notes/falsification.md), V1: does executing a trigger directly reproduce the lookup?

For a random subset of stored av2_test scenarios, run each trigger configuration LIVE (predictor
scoring at every decision tick; no score replay, no forced-fire table) and compare firing step and
outcomes with the exact lookup in evaluate.Scenario.

    python src/direct_validation.py --n 150 --out results/final/direct_validation.json
Resumable: finished (seed, config) pairs are read back from <out>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CFG = Path("results/campaign/av2_test/config.json")
CONFIGS = [("ltt_lat0", "ltt", 0), ("ltt_lat1", "ltt", 1), ("peakq25_lat0", "q25", 0), ("peakq75_lat0", "q75", 0)]
ALPHA, DELTA = 0.05, 0.10


def thresholds():
    from evaluate import Scenario, load_rows, ltt
    cal = [Scenario(r) for r in load_rows(["results/campaign/av2_cal"])]
    peaks = np.array([s.cummax["geom"][-1] for s in cal])
    # q25 / q75 of the calibration peak scores: thresholds at which roughly 75 % / 25 % of scenarios fire
    return {"ltt": ltt(cal, "geom", ALPHA, DELTA), "q25": float(np.quantile(peaks, 0.25)),
            "q75": float(np.quantile(peaks, 0.75))}


def lookup(sc, lam, lat):
    t = int(sc.tick("geom", lam, lat)[0])
    return dict(tick=t, fired=bool(t < sc.n_ticks), miss=float(sc.miss[t]), stop=float(sc.stop[t]),
                induced=float(sc.induced[t]), harm_run=float(sc.harm_run[t]), rc=float(sc.rc[t]))


def direct(res, ref_row, harm):
    from evaluate import _rr
    from rollout import induced_collision
    cf = _rr(ref_row["ref"])
    h = harm.first_harm(res)
    ref_harm = harm.first_harm(cf)
    return dict(step=res.trigger_step, fired=bool(res.triggered),
                miss=float(h is not None and not (res.triggered and res.trigger_step < h)),
                stop=float(bool(res.triggered) and ref_harm is None),
                induced=float(induced_collision(res, cf)), harm_run=float(h is not None),
                rc=float(res.route_completion))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", type=int, default=0, help="this process handles scenarios with position %% parts == part")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", default="results/final/direct_validation.json")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"

    import simenv as se
    from autobot_predictor import AutoBotPredictor
    from evaluate import NUPLAN_HARM, Scenario, load_rows
    from rollout import make_fallback_policy_cls, rollout

    cfg = json.loads(CFG.read_text())
    de, size = cfg["decide_every"], cfg["span"] // 8
    rows = load_rows(["results/campaign/av2_test"])
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(rows), size=min(a.n, len(rows)), replace=False).tolist())[a.part::a.parts]
    lams = thresholds()
    print("thresholds", {str(k): v for k, v in lams.items()}, flush=True)

    out_f = Path(a.out).with_suffix(f".part{a.part}.jsonl" if a.parts > 1 else ".jsonl")
    out_f.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_f.exists():
        for line in out_f.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add((d["seed"], d["config"]))
            except (json.JSONDecodeError, KeyError):
                pass

    predictor = AutoBotPredictor(threads=1, mc=cfg["mc"], ckpt=cfg["ckpt"])
    policy = make_fallback_policy_cls(cfg["mrm_decel"])
    envs = {}
    t0 = time.time()
    for n, i in enumerate(pick):
        row = rows[i]
        seed = row["seed"]
        wid = min(seed // size, 7)
        if wid not in envs:
            for e in envs.values():
                e.close()
            envs = {wid: se.make_env(cfg["db"], wid * size, size, policy=policy, reactive_traffic=not cfg["replay"])}
        env = envs[wid]
        sc = Scenario(row)
        for name, key, lat in CONFIGS:
            if (seed, name) in done:
                continue
            lam = lams[key]
            rec = dict(seed=seed, config=name, lam=float(lam), lat=lat, lookup=lookup(sc, lam, lat))
            try:
                res = rollout(env, seed, float(lam), predictor, decide_every=de, latency_steps=lat * de)
                rec["direct"] = direct(res, row, NUPLAN_HARM)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:300]
                env.close()
                envs[wid] = env = se.make_env(cfg["db"], wid * size, size, policy=policy,
                                              reactive_traffic=not cfg["replay"])
            with open(out_f, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
        print(f"  {n + 1}/{len(pick)} seed {seed}  {(time.time() - t0) / 60:.1f} min", flush=True)
    for e in envs.values():
        e.close()


if __name__ == "__main__":
    main()
