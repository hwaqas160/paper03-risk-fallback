"""
Amendment 9, E1 (notes/falsification.md): add the MC-dropout ensemble-spread trace (T2) to an already-collected arm by
re-running ONLY its reference run with `mc` dropout passes. Forced-fire outcomes are reused (they do not depend on any
score). Each sidecar row stores the fresh run's tick count and geometric trace so the caller can verify that the re-run
reproduced the stored reference (evaluate-side check in src/ens_eval.py).

    python src/ens_sidecar.py --arm results/campaign/av2_cal5_gpu --workers 4
Resumable: finished seeds are read back from ens_<wid>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


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
    import simenv as se
    from autobot_predictor import AutoBotPredictor
    from rollout import make_fallback_policy_cls, rollout

    arm, cfg, seeds, wid, mc = Path(job["arm"]), job["cfg"], job["seeds"], job["wid"], job["mc"]
    out_f = arm / f"ens_{wid:02d}.jsonl"
    todo = [s for s in seeds if s not in _done(out_f)]
    if not todo:
        return wid, 0, 0.0
    predictor = AutoBotPredictor(threads=1, mc=mc, ckpt=cfg["ckpt"])
    policy = make_fallback_policy_cls(cfg["mrm_decel"])
    size = cfg["span"] // cfg["workers"]
    env, cur, n, t0 = None, None, 0, time.time()
    for seed in sorted(todo):
        blk = min((seed - cfg["start"]) // size, cfg["workers"] - 1)
        if blk != cur:
            if env is not None:
                env.close()
            env = se.make_env(cfg["db"], cfg["start"] + blk * size, size, policy=policy, reactive_traffic=not cfg["replay"])
            cur = blk
        try:
            ref = rollout(env, seed, float("inf"), predictor, decide_every=cfg["decide_every"], record_all=True)
            rec = dict(seed=seed, n_ticks=len(ref.score_traces.get("geom", [])), steps=ref.steps,
                       geom=ref.score_traces.get("geom", []), ens=ref.score_traces.get("ens", []))
        except Exception as e:  # noqa: BLE001
            rec = dict(seed=seed, error=repr(e)[:200])
            env.close(); env = None; cur = None
        with open(out_f, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        n += 1
        if n % 10 == 0:
            print(f"  w{wid}: {n}/{len(todo)} {(time.time() - t0) / 60:.1f} min", flush=True)
    if env is not None:
        env.close()
    return wid, n, (time.time() - t0) / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mc", type=int, default=5)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import load_rows
    arm = Path(a.arm)
    cfg = json.loads((arm / "config.json").read_text())
    seeds = sorted(r["seed"] for r in load_rows([str(arm)]))
    jobs = [dict(arm=str(arm), cfg=cfg, seeds=seeds[i::a.workers], wid=i, mc=a.mc) for i in range(a.workers)]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for wid, n, mins in ex.map(_worker, jobs):
            print(f"worker {wid}: {n} rows, {mins:.1f} min", flush=True)


if __name__ == "__main__":
    main()
