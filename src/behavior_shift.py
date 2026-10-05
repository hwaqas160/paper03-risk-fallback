"""
Amendment 9, B1 (notes/falsification.md): thresholds from reactive-traffic calibration, deployed on the non-reactive
log-replay arm. Reads stored rows only.

    python src/behavior_shift.py --out results/final/behavior_shift.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ld = lambda p: [s for s in (Scenario(r) for r in load_rows([p])) if "conf_conflict" in s.traces]
    cal, rep, react = ld("results/campaign/av2_cal"), ld("results/campaign/av2_test_replay"), ld("results/campaign/av2_test")
    seeds = {s.seed for s in rep}
    react_p = [s for s in react if s.seed in seeds]            # same scenarios, reactive traffic (paired)
    rep = [s for s in rep if s.seed in {x.seed for x in react_p}]
    react_p = sorted(react_p, key=lambda s: s.seed); rep = sorted(rep, key=lambda s: s.seed)
    thr = {"LTT geometric": ("geom", ltt(cal, "geom", ALPHA, DELTA)),
           "tuned geometric": ("geom", tuned(cal, "geom", ALPHA)),
           "tuned T1 confidence": ("conf_conflict", tuned(cal, "conf_conflict", ALPHA))}
    out = dict(n_paired=len(rep), harm_rate_reactive=float(np.mean([s.ref_harmful for s in react_p])),
               harm_rate_replay=float(np.mean([s.ref_harmful for s in rep])), methods={})
    per = {}
    for name, (key, lam) in thr.items():
        sr, sp = summarize(react_p, key, lam), summarize(rep, key, lam)
        per[name] = (key, lam)
        out["methods"][name] = dict(lam=lam, reactive=sr, replay=sp)
    # paired bootstrap of the stop-rate difference LTT-geom vs T1 on replay traffic
    def stops(scn, key, lam):
        return matrices(scn, key, np.array([lam]))[1][:, 0]
    d = stops(rep, *per["LTT geometric"]) - stops(rep, *per["tuned T1 confidence"])
    rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    out["stop_diff_ltt_minus_t1_replay"] = dict(diff=float(d.mean()), ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])
    m = out["methods"]["LTT geometric"]["replay"]
    out["verdict_B1"] = dict(ltt_miss_replay=m["miss"], ltt_miss_ci=m["miss_ci"],
                             refuted=bool(m["miss"] > ALPHA and m["miss_ci"][0] > ALPHA),
                             t1_meets_target=bool(out["methods"]["tuned T1 confidence"]["replay"]["miss"] <= ALPHA))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print("n paired", len(rep), "harm reactive", round(out["harm_rate_reactive"], 3), "replay", round(out["harm_rate_replay"], 3))
    for k, v in out["methods"].items():
        print(f"{k:22s} reactive miss {v['reactive']['miss']:.3f} stop {v['reactive']['unnecessary_stop']:.3f} | "
              f"replay miss {v['replay']['miss']:.3f} [{v['replay']['miss_ci'][0]:.3f},{v['replay']['miss_ci'][1]:.3f}] stop {v['replay']['unnecessary_stop']:.3f}")
    print(out["stop_diff_ltt_minus_t1_replay"], out["verdict_B1"])


if __name__ == "__main__":
    main()
