"""
Amendment 9, E1 (notes/falsification.md): second-predictor ensemble baseline from the ens_*.jsonl sidecars.

    python src/ens_eval.py --out results/final/ens_gpu.json
Scenarios whose re-run does not reproduce the stored reference (tick count or geometric trace) are dropped and counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, ltt, matrices, summarize, tuned  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def load_arm(path):
    side = {}
    for f in sorted(Path(path).glob("ens_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in d:
                side[d["seed"]] = d
    out, dropped, total = [], 0, 0
    for r in load_rows([path]):
        s = Scenario(r)
        if "conf_conflict" not in s.traces:
            continue
        total += 1
        d = side.get(r["seed"])
        if d is None:
            continue
        g = np.asarray(d["geom"], float)
        if d["n_ticks"] != s.n_ticks or len(g) != len(s.traces["geom"]) or np.abs(g - s.traces["geom"]).max() > 1e-3:
            dropped += 1
            continue
        s.traces["ens"] = np.asarray(d["ens"], float)
        s.cummax["ens"] = np.maximum.accumulate(s.traces["ens"])
        out.append(s)
    return out, dict(total=total, with_sidecar=len(side), reproduced=len(out), not_reproduced=dropped)


def paired(a, b, B=10000, seed=0):
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return dict(diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="results/campaign/av2_cal5_gpu")
    ap.add_argument("--test", default="results/campaign/av2_test5_gpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal, cs = load_arm(a.cal)
    test, ts = load_arm(a.test)
    thr = {"LTT geometric": ("geom", ltt(cal, "geom", ALPHA, DELTA)),
           "tuned geometric": ("geom", tuned(cal, "geom", ALPHA)),
           "tuned T1 confidence": ("conf_conflict", tuned(cal, "conf_conflict", ALPHA)),
           "tuned T2 ensemble": ("ens", tuned(cal, "ens", ALPHA))}
    out = dict(cal=cs, test=ts, n_cal=len(cal), n_test=len(test), methods={})
    stops = {}
    for k, (key, lam) in thr.items():
        out["methods"][k] = dict(lam=lam, **summarize(test, key, lam))
        stops[k] = matrices(test, key, np.array([lam]))[1][:, 0]
    meet = [k for k in ("tuned T1 confidence", "tuned T2 ensemble") if out["methods"][k]["miss"] <= ALPHA]
    out["verdict_E1"] = dict(ltt_miss=out["methods"]["LTT geometric"]["miss"], baselines_meeting_target=meet)
    for k in ("tuned T1 confidence", "tuned T2 ensemble"):
        out["verdict_E1"][f"stop_diff_ltt_minus_{k}"] = paired(stops["LTT geometric"], stops[k])
    t2 = out["verdict_E1"]["stop_diff_ltt_minus_tuned T2 ensemble"]
    out["verdict_E1"]["advantage_over_T2_survives"] = bool("tuned T2 ensemble" in meet and t2["ci"][1] < 0)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(cs, ts)
    for k, v in out["methods"].items():
        print(f"{k:22s} miss {v['miss']:.3f} [{v['miss_ci'][0]:.3f},{v['miss_ci'][1]:.3f}] stop {v['unnecessary_stop']:.3f}")
    print(out["verdict_E1"])


if __name__ == "__main__":
    main()
