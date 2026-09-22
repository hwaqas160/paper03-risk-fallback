"""
Week-2 GATE: headless closed-loop rollout throughput across W worker processes.

Each worker owns a disjoint shard of the scenario database and runs IDM closed-loop
rollouts (the cheapest realistic ego policy) until its wall budget is spent. Engine
start-up is timed separately and excluded from the steady-state rate. A scenario that
raises is recorded and skipped rather than killing the worker.

    python src/bench_throughput.py --db av2_test --workers 1 4 8 16 --budget 120
Appends to results/throughput_<db>.json; summarised in notes/sim_throughput.md.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

from simenv import DBS, free_commit_gb

# Measured 2026-09-15 on this machine: a sim worker commits ~1.8 GB with torch blocked
# (~2.3 GB without). Keep a reserve so we never repeat the commit exhaustion that killed
# Paper 01's training run.
COMMIT_PER_WORKER_GB = 1.8
COMMIT_RESERVE_GB = 6.0


def _worker(args):
    wid, db, start, count, budget = args
    os.environ["OMP_NUM_THREADS"] = "1"
    from simenv import block_torch, ego_is_static, make_env

    block_torch()
    t0 = time.perf_counter()
    env = make_env(db, start, count)
    init_s = time.perf_counter() - t0

    eps, skipped, errors, seed = [], 0, [], start
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < budget and seed < start + count:
        tr = time.perf_counter()
        try:
            env.reset(seed=seed)
            reset_s = time.perf_counter() - tr
            if ego_is_static(env):
                skipped += 1
                continue
            steps, ts, crashed = 0, time.perf_counter(), False
            while True:
                _, _, tm, tc, info = env.step([0.0, 0.0])
                steps += 1
                crashed = crashed or bool(info.get("crash"))  # any step, not just the last
                if tm or tc:
                    break
            eps.append(dict(seed=seed, reset_s=reset_s, steps=steps, step_s=time.perf_counter() - ts,
                            crash=crashed, arrive=bool(info.get("arrive_dest"))))
        except Exception as e:  # noqa: BLE001 — record and move on; a benchmark must not die on one map
            errors.append(dict(seed=seed, error=repr(e), tb=traceback.format_exc(limit=3)))
            env.close()
            env = make_env(db, start, count)
        finally:
            seed += 1
    wall = time.perf_counter() - t_start
    env.close()
    return dict(wid=wid, init_s=init_s, wall_s=wall, skipped=skipped, errors=errors, episodes=eps)


def affordable_workers(workers: int, per_worker_gb: float = COMMIT_PER_WORKER_GB) -> int:
    """Clip the worker count to what the machine's free commit can actually hold.
    per_worker_gb: 1.8 for torch-free sim workers, ~2.8 for AutoBot workers (measured)."""
    free = free_commit_gb()
    cap = max(int((free - COMMIT_RESERVE_GB) // per_worker_gb), 0)
    if workers > cap:
        print(f"  [guard] free commit {free:.1f} GB -> at most {cap} workers "
              f"(reserve {COMMIT_RESERVE_GB} GB, {per_worker_gb} GB/worker); "
              f"requested {workers}", flush=True)
    return min(workers, cap)


def run(db: str, workers: int, budget: float) -> dict:
    from scenarionet.common_utils import read_dataset_summary
    total = len(read_dataset_summary(DBS[db])[1])
    shard = total // workers
    jobs = [(w, db, w * shard, shard, budget) for w in range(workers)]
    # ProcessPoolExecutor, not Pool: a worker that dies raises BrokenProcessPool instead of
    # being respawned forever (which is what flooded the log when commit ran out).
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_worker, jobs))

    eps = [e for r in res for e in r["episodes"]]
    errors = [e for r in res for e in r["errors"]]
    skipped = sum(r["skipped"] for r in res)
    wall = sum(r["wall_s"] for r in res) / workers
    steps = sum(e["steps"] for e in eps)
    n = max(len(eps), 1)
    return dict(
        db=db, workers=workers, budget_s=budget, cpu_logical=os.cpu_count(),
        init_s_mean=sum(r["init_s"] for r in res) / workers,
        episodes=len(eps), skipped_static=skipped, errors=len(errors),
        error_samples=errors[:3],
        scenarios_per_hour=3600.0 * (len(eps) + skipped + len(errors)) / wall,
        rollouts_per_hour=3600.0 * len(eps) / wall,
        steps_per_s=steps / wall,
        mean_reset_s=sum(e["reset_s"] for e in eps) / n,
        mean_steps=steps / n,
        mean_step_fps=steps / max(sum(e["step_s"] for e in eps), 1e-9),
        idm_crash_rate=sum(e["crash"] for e in eps) / n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="av2_test", choices=list(DBS))
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--budget", type=float, default=120.0, help="steady-state seconds per worker")
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / "results" / f"throughput_{args.db}.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    for w_req in args.workers:
        w = affordable_workers(w_req)
        if w < w_req:
            print(f"W={w_req:>2}  SKIPPED — not enough free commit for {w_req} workers", flush=True)
            continue
        r = run(args.db, w, args.budget)
        r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows.append(r)
        out.write_text(json.dumps(rows, indent=2))
        print(f"W={w:>2}  rollouts/h={r['rollouts_per_hour']:>8.0f}  steps/s={r['steps_per_s']:>6.0f}  "
              f"reset={r['mean_reset_s']:.2f}s  fps/proc={r['mean_step_fps']:.0f}  eps={r['episodes']} "
              f"static={r['skipped_static']} errors={r['errors']} idm_crash={r['idm_crash_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
