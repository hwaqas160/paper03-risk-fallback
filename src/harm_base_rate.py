"""
How often is a replayed scenario "harmful" — under each candidate harm definition?

Runs the no-fallback closed loop (IDM ego, lam = inf, no trigger scoring) on the DEV split,
records per-step TTC traces once, then scores every candidate HarmDef post hoc. Output is
the table the author needs to fix the harm definition and alpha in the falsification note:
a definition whose base rate exceeds alpha makes alpha unreachable by construction, and
harm that starts in the first second can't be prevented by any trigger.

    python src/harm_base_rate.py --db av2_dev --n 1000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from simenv import DBS

EARLY_STEPS = 10  # harm beginning in the first 1.0 s: no trigger + brake can plausibly prevent it

CANDIDATES = [
    # headline: nuPlan closed-loop benchmark (at-fault collision or TTC < 0.95 s)
    dict(),
    # sensitivity variants
    dict(use_ttc=False),                          # nuPlan at-fault collisions only
    dict(sustain=3),                              # nuPlan TTC, sustained 0.3 s
    dict(ttc_s=1.5),                              # common T-ITS "critical TTC" 1.5 s
    dict(at_fault_only=False),                    # any collision (incl. rear-ended) or TTC
    dict(use_ttc=False, at_fault_only=False),     # any collision
]


def _worker(args):
    db, start, count, want, overrides = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import simenv
    simenv.block_torch()
    from rollout import make_fallback_policy_cls, rollout

    env = simenv.make_env(db, start, count, policy=make_fallback_policy_cls(), **overrides)
    out, seed = [], start
    try:
        while len(out) < want and seed < start + count:
            try:
                env.reset(seed=seed)
                if not simenv.ego_is_static(env):
                    r = rollout(env, seed, float("inf"), None)
                    d = asdict(r)
                    d.pop("scores")
                    d.pop("ego_trace")
                    out.append(d)
            except Exception as e:  # noqa: BLE001 — one bad map must not kill the study
                out.append(dict(seed=seed, error=repr(e)))
                env.close()
                env = simenv.make_env(db, start, count, policy=make_fallback_policy_cls(), **overrides)
            seed += 1
    finally:
        env.close()
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="av2_dev", choices=list(DBS))
    ap.add_argument("--n", type=int, default=1000, help="non-static scenarios in total")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reactive", action="store_true",
                    help="IDM-reactive traffic instead of pure log replay")
    args = ap.parse_args()

    from bench_throughput import affordable_workers
    from scenarionet.common_utils import read_dataset_summary
    w = affordable_workers(args.workers)
    if w < 1:
        raise SystemExit("not enough free commit for even one worker")
    total = len(read_dataset_summary(DBS[args.db])[1])
    shard = total // w
    want = -(-args.n // w)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
        res = [r for part in ex.map(_worker, [(args.db, i * shard, shard, want, dict(reactive_traffic=args.reactive)) for i in range(w)]) for r in part]
    errors = [r for r in res if "error" in r]
    res = [r for r in res if "error" not in r]
    print(f"{len(res)} rollouts ({len(errors)} errors) on {args.db} in {time.time() - t0:.0f}s with {w} workers\n")

    out = Path(args.out or Path(__file__).resolve().parents[1] / "results" / f"harm_traces_{args.db}{'_reactive' if args.reactive else ''}.jsonl")
    with open(out, "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")

    from rollout import HarmDef, RolloutResult
    rolls = [RolloutResult(**r) for r in res]
    n = len(rolls)
    print(f"{'definition':<34} {'base rate':>10} {'95% CI':>15} {'starts <1s':>11} {'median start':>13}")
    table = []
    for c in CANDIDATES:
        hd = HarmDef(**c)
        starts = [hd.first_harm(r) for r in rolls]
        hit = [s for s in starts if s is not None]
        k = len(hit)
        lo, hi = wilson(k, n)
        early = sum(s < EARLY_STEPS for s in hit)
        med = float(np.median(hit)) if hit else float("nan")
        table.append(dict(definition=hd.label(), **c, base_rate=k / n, ci=(lo, hi),
                          early_frac_of_harm=early / max(k, 1), unpreventable_rate=early / n,
                          median_start_step=med))
        print(f"{hd.label():<34} {k / n:>10.3f} {f'[{lo:.3f},{hi:.3f}]':>15} "
              f"{early / max(k, 1):>11.2f} {med:>13.0f}")
    coll = sum(r.collision for r in rolls)
    print(f"\ncollisions: {coll}/{n} = {coll / n:.3f}   (IDM ego, {'reactive' if args.reactive else 'log-replayed'} traffic, no fallback)")
    print("'starts <1s' = share of harmful scenarios whose harm begins in the first second —")
    print("a floor on the miss rate that no trigger can remove. alpha must sit above it.")
    (out.with_suffix(".summary.json")).write_text(json.dumps(dict(db=args.db, reactive=args.reactive, n=n, errors=len(errors),
                                                                  collisions=coll, table=table), indent=2))
    print(f"\nwrote {out} and {out.with_suffix('.summary.json').name}")


if __name__ == "__main__":
    main()
