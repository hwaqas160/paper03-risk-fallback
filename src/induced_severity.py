"""
Amendment 9, C1 (notes/falsification.md): re-run the single forced rollout of every scenario with an induced collision at
the certified threshold (av2_test) and record ego speed, partner speed, speed difference and the follower at firing.

    python src/induced_severity.py --out results/final/induced_severity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    import simenv as se
    from evaluate import Scenario, load_rows, ltt
    from rollout import make_fallback_policy_cls, rollout

    cfg = json.loads(Path("results/campaign/av2_test/config.json").read_text())
    de, size = cfg["decide_every"], cfg["span"] // 8
    cal = [s for s in (Scenario(r) for r in load_rows(["results/campaign/av2_cal"])) if "conf_conflict" in s.traces]
    lam = ltt(cal, "geom", 0.05, 0.10)
    todo = []
    for r in load_rows(["results/campaign/av2_test"]):
        sc = Scenario(r)
        if "conf_conflict" not in sc.traces:
            continue
        t = int(sc.tick("geom", lam)[0])
        if t < sc.n_ticks and sc.induced[t]:
            f = r["fired"][str(t * de)]
            todo.append((r["seed"], t * de, [c["step"] for c in f["collisions"] if not c["at_fault"]]))
    policy = make_fallback_policy_cls(cfg["mrm_decel"])
    envs, res = {}, []
    for seed, fire, stored in sorted(todo):
        wid = min(seed // size, 7)
        if wid not in envs:
            for e in envs.values():
                e.close()
            envs = {wid: se.make_env(cfg["db"], wid * size, size, policy=policy, reactive_traffic=not cfg["replay"])}
        log = []
        try:
            r = rollout(envs[wid], seed, float("inf"), None, decide_every=de, fire_step=fire, contact_log=log)
        except Exception as e:  # noqa: BLE001
            res.append(dict(seed=seed, error=repr(e)[:200])); continue
        got = [c["step"] for c in r.collisions if not c["at_fault"]]
        res.append(dict(seed=seed, fire_step=fire, stored_steps=stored, rerun_steps=got,
                        reproduced=bool(got == stored), log=log))
        print(seed, "reproduced" if got == stored else f"MISMATCH {stored} vs {got}", flush=True)
    for e in envs.values():
        e.close()
    Path(a.out).write_text(json.dumps(dict(lam=lam, n_scenarios=len(todo), results=res), indent=1, default=float))


if __name__ == "__main__":
    main()
