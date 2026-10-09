"""
Amendment 12, A12-6: 10 Hz decision arm. For each non-static scenario of an existing arm, repeat the reference run WITHOUT the predictor recording per-step
ego speed, in-lane lead gap/speed and TTC, then one forced-fire rollout at EVERY simulator step. Resumable (one JSON line per scenario).

    python src/campaign10.py --arm results/campaign/av2_cal5 --out results/campaign/av2_cal5_10hz --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUTCOME_KEYS = ("steps", "collision", "collision_step", "at_fault_step", "collisions", "offroad", "min_ttc", "ttc_violations", "triggered",
                "trigger_step", "diverged", "stopped_steps", "route_completion", "arrive", "peak_decel", "ttc_trace")


def _done(path):
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                out.add(json.loads(line)["seed"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def _worker(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import simenv as se
    from rollout import make_fallback_policy_cls, rollout

    out_dir, cfg, seeds, wid = Path(job["out"]), job["cfg"], job["seeds"], job["wid"]
    out_f = out_dir / f"part_{wid:02d}.jsonl"
    todo = [s for s in seeds if s not in _done(out_f)]
    policy = make_fallback_policy_cls(cfg["mrm_decel"])
    size = cfg["span"] // cfg["workers"]
    env, cur, t0, n = None, None, time.time(), 0
    for seed in sorted(todo):
        blk = min((seed - cfg["start"]) // size, cfg["workers"] - 1)
        if blk != cur:
            if env is not None:
                env.close()
            env = se.make_env(cfg["db"], cfg["start"] + blk * size, size, policy=policy, reactive_traffic=not cfg["replay"])
            cur = blk
        try:
            kin = []
            ref = rollout(env, seed, float("inf"), None, decide_every=1, kin_log=kin)
            fired = {}
            for k in range(0, ref.steps):
                r = rollout(env, seed, float("inf"), None, decide_every=1, fire_step=k, ref_ego=ref.ego_trace)
                fired[str(k)] = {key: getattr(r, key) for key in OUTCOME_KEYS}
            row = dict(seed=seed, steps=ref.steps, kin=kin, ref={key: getattr(ref, key) for key in OUTCOME_KEYS}, fired=fired,
                       n_diverged=sum(int(v["diverged"]) for v in fired.values()))
        except Exception as e:  # noqa: BLE001
            row = dict(seed=seed, error=repr(e)[:200])
            if env is not None:
                env.close()
            env, cur = None, None
        with open(out_f, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
        n += 1
        if n % 5 == 0:
            print(f"  w{wid}: {n}/{len(todo)} {(time.time() - t0) / 60:.1f} min", flush=True)
    if env is not None:
        env.close()
    return wid, n, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_throughput import affordable_workers
    from evaluate import load_rows
    arm, out = Path(a.arm), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((arm / "config.json").read_text())
    (out / "config.json").write_text(json.dumps(dict(cfg, source_arm=str(arm), decisions="every_step"), indent=2))
    seeds = sorted(r["seed"] for r in load_rows([str(arm)]))
    w = min(a.workers, max(1, affordable_workers(a.workers, 2.0)))
    jobs = [dict(out=str(out), cfg=cfg, seeds=seeds[i::w], wid=i) for i in range(w)]
    with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
        for wid, n, mins in ex.map(_worker, jobs):
            print(f"worker {wid}: {n} scenarios, {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
