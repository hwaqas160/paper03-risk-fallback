"""
Resumable, parallel data-collection campaign: one tick-sweep row per scenario
(rollout.sweep_scenario_ticks), from which src/evaluate.py computes every method exactly.

Crash-safe by construction: each worker owns a contiguous seed block and appends one JSON
line per finished scenario (flushed + fsynced) to its own file. On restart, finished and
skipped seeds are read back and not re-run, so an interruption costs at most the scenario
in flight — important for a many-hour run on a machine shared with other projects.

    python src/campaign.py --db av2_cal --n 2000 --workers 12 --out results/campaign/av2_cal
    (re-run the same command to resume)
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

import simenv

STATIC_FRAC = 0.30


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["seed"])
                except (json.JSONDecodeError, KeyError):
                    pass                     # a half-written last line from a crash: redo it
    return out


def _append(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _rescore_worker(job):
    """
    Re-run ONLY the reference (fallback-disabled) run for scenarios whose stored rows lack
    per-agent tables (collected before 2026-09-24), and write agents_<wid>.jsonl sidecars.
    The forced-fire rollouts -- the physics every outcome comes from -- are NOT redone: they
    depend only on the firing tick, not on any score. Each sidecar also stores the fresh run's
    'geom' trace and tick count so evaluate.py / the caller can verify the re-run matches the
    original (same tick count, same geometric score) before trusting it.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    import simenv as se
    from rollout import make_fallback_policy_cls, rollout
    from autobot_predictor import AutoBotPredictor

    out_dir = Path(job["out"])
    side_f = out_dir / f"agents_{job['wid']:02d}.jsonl"
    todo = []
    src = out_dir / f"part_{job['wid']:02d}.jsonl"
    have = _done(side_f)
    if src.exists():
        with open(src) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ref" in r and not r["ref"].get("agent_tables") and r["seed"] not in have:
                    todo.append(r["seed"])
    if not todo:
        return dict(wid=job["wid"], rows=0, minutes=0.0)
    predictor = AutoBotPredictor(threads=1, mc=0, ckpt=job["ckpt"])       # MC dropout not needed here
    env = se.make_env(job["db"], job["start"], job["count"], policy=make_fallback_policy_cls(job["mrm_decel"]),
                      reactive_traffic=not job["replay"])
    t0, n = time.time(), 0
    try:
        for seed in todo:
            try:
                env.reset(seed=seed)
                ref = rollout(env, seed, float("inf"), predictor, decide_every=job["decide_every"], record_all=True)
                _append(side_f, dict(seed=seed, n_ticks=len(ref.agent_tables), agent_tables=ref.agent_tables,
                                     geom=ref.score_traces.get("geom", []), steps=ref.steps))
                n += 1
            except Exception as e:  # noqa: BLE001
                _append(out_dir / f"skip_{job['wid']:02d}.jsonl", dict(seed=seed, reason="rescore_error", error=repr(e)[:300]))
                env.close()
                env = se.make_env(job["db"], job["start"], job["count"], policy=make_fallback_policy_cls(job["mrm_decel"]),
                                  reactive_traffic=not job["replay"])
    finally:
        env.close()
    return dict(wid=job["wid"], rows=n, minutes=(time.time() - t0) / 60)


def _worker(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    import simenv as se
    from rollout import make_fallback_policy_cls, sweep_scenario_ticks
    from autobot_predictor import AutoBotPredictor

    out_dir = Path(job["out"])
    rows_f = out_dir / f"part_{job['wid']:02d}.jsonl"
    skip_f = out_dir / f"skip_{job['wid']:02d}.jsonl"
    done = _done(rows_f) | _done(skip_f)
    have = len(_done(rows_f))
    predictor = AutoBotPredictor(threads=1, mc=job["mc"], ckpt=job["ckpt"], method=job.get("method", "autobot"))
    policy = make_fallback_policy_cls(job["mrm_decel"])
    env = se.make_env(job["db"], job["start"], job["count"], policy=policy,
                      reactive_traffic=not job["replay"])
    t0 = time.time()
    seed = job["start"]
    try:
        while have < job["want"] and seed < job["start"] + job["count"]:
            if seed in done:
                seed += 1
                continue
            try:
                env.reset(seed=seed)
                if se.ego_is_static(env):
                    _append(skip_f, dict(seed=seed, reason="static_ego"))
                else:
                    row = sweep_scenario_ticks(env, seed, predictor, decide_every=job["decide_every"])
                    row.update(db=job["db"], ckpt=str(job["ckpt"]), replay=job["replay"],
                               mrm_decel=job["mrm_decel"], mc=job["mc"])
                    _append(rows_f, row)
                    have += 1
            except Exception as e:  # noqa: BLE001 — log, skip, keep the block going
                _append(skip_f, dict(seed=seed, reason="error", error=repr(e)[:300]))
                env.close()
                env = se.make_env(job["db"], job["start"], job["count"], policy=policy,
                                  reactive_traffic=not job["replay"])
            seed += 1
    finally:
        env.close()
    return dict(wid=job["wid"], rows=have, minutes=(time.time() - t0) / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, choices=list(simenv.DBS))
    ap.add_argument("--n", type=int, required=True, help="non-static scenarios wanted (total)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--decide_every", type=int, default=5)
    ap.add_argument("--mc", type=int, default=5, help="MC-dropout passes for T2 (0 = off)")
    ap.add_argument("--mrm_decel", type=float, default=4.0)
    ap.add_argument("--replay", action="store_true", help="non-reactive log-replay traffic (ablation)")
    ap.add_argument("--ckpt", default=None, help="AutoBot checkpoint (default: autobot_predictor.DEFAULT_CKPT)")
    ap.add_argument("--method", default="autobot", choices=["autobot", "wayformer"], help="UniTraj model family of --ckpt")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run only the reference run for stored rows lacking per-agent tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from autobot_predictor import DEFAULT_CKPT
    from bench_throughput import affordable_workers
    from scenarionet.common_utils import read_dataset_summary

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = affordable_workers(a.workers, 2.8)
    if w < 1:
        raise SystemExit("not enough free commit for even one worker")
    if w < a.workers:
        print(f"  [guard] running {w} workers instead of {a.workers}", flush=True)
    total = len(read_dataset_summary(simenv.DBS[a.db])[1])
    # Blocks are fixed by --workers (not the guarded count), so a resume with a different
    # free-memory situation still maps every seed to the same file.
    span = min(total - a.start, math.ceil(a.n / (1 - STATIC_FRAC) * 1.3))
    size = span // a.workers
    per = math.ceil(a.n / a.workers)
    jobs = [dict(wid=i, db=a.db, start=a.start + i * size, count=size, want=per, out=str(out),
                 decide_every=a.decide_every, mc=a.mc, mrm_decel=a.mrm_decel, replay=a.replay,
                 ckpt=str(a.ckpt or DEFAULT_CKPT), method=a.method) for i in range(a.workers)]
    (out / "config.json").write_text(json.dumps(dict(vars(a), ckpt=str(a.ckpt or DEFAULT_CKPT),
                                                     total_in_db=total, span=span), indent=2))
    t0 = time.time()
    fn = _rescore_worker if a.rescore else _worker
    with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(fn, jobs):
            print(f"  worker {r['wid']:02d}: {r['rows']} rows, {r['minutes']:.1f} min", flush=True)
    pat = "agents_*.jsonl" if a.rescore else "part_*.jsonl"
    n = sum(len(_done(p)) for p in out.glob(pat))
    print(f"{a.db}: {n} {'rescored' if a.rescore else ''} rows in {out} ({(time.time() - t0) / 3600:.2f} h this run)", flush=True)


if __name__ == "__main__":
    main()
