"""
Amendment 10, W1 (notes/falsification.md): Wayformer arm. Validity of LTT over 200 resplits, held-out miss, and the paired
stop-rate difference against tuned T1 (Wayformer confidence).

    python src/way_eval.py --out results/final/way_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, ltt, matrices, summarize, tuned, validity  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def load(path):
    out = []
    for r in load_rows([path]):
        s = Scenario(r)
        if "conf_conflict" not in s.traces:
            continue
        s.traces.pop("ens", None); s.cummax.pop("ens", None)
        out.append(s)
    return out


def paired(a, b, B=10000, seed=0):
    d = a - b
    bs = d[np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))].mean(1)
    return dict(diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="results/campaign/av2_cal5_way")
    ap.add_argument("--test", default="results/campaign/av2_test5_way")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    cal, test = load(a.cal), load(a.test)
    thr = {"LTT geometric": ("geom", ltt(cal, "geom", ALPHA, DELTA)),
           "CRC geometric": ("geom", None),
           "tuned geometric": ("geom", tuned(cal, "geom", ALPHA)),
           "tuned T1 confidence": ("conf_conflict", tuned(cal, "conf_conflict", ALPHA))}
    out = dict(n_cal=len(cal), n_test=len(test), methods={})
    stops = {}
    for k, (key, lam) in thr.items():
        if lam is None:
            continue
        out["methods"][k] = dict(lam=lam, **summarize(test, key, lam))
        stops[k] = matrices(test, key, np.array([lam]))[1][:, 0]
    v = validity(cal + test, ALPHA, DELTA, reps=a.reps)
    out["validity"] = v
    t1 = out["methods"]["tuned T1 confidence"]
    ltt_m = out["methods"]["LTT geometric"]
    out["stop_diff_ltt_minus_t1"] = paired(stops["LTT geometric"], stops["tuned T1 confidence"])
    ok_a = v["T4 LTT (ours)"]["violation_freq"] <= DELTA and ltt_m["miss"] <= ALPHA
    ok_b = t1["miss"] <= ALPHA and out["stop_diff_ltt_minus_t1"]["ci"][1] < 0
    out["verdict_W1"] = dict(a_valid=bool(ok_a), ltt_violation=v["T4 LTT (ours)"]["violation_freq"], ltt_miss=ltt_m["miss"],
                             t1_meets_target=bool(t1["miss"] <= ALPHA), b_advantage=bool(ok_b), refuted=bool(not (ok_a and ok_b)))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for k, m in out["methods"].items():
        print(f"{k:22s} miss {m['miss']:.3f} [{m['miss_ci'][0]:.3f},{m['miss_ci'][1]:.3f}] stop {m['unnecessary_stop']:.3f}")
    print({k: (round(x["violation_freq"], 3), round(x["stop_mean"], 3)) for k, x in v.items()})
    print(out["stop_diff_ltt_minus_t1"], out["verdict_W1"])


if __name__ == "__main__":
    main()
