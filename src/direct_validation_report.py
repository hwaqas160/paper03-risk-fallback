"""
Amendment 7 (notes/falsification.md), V1: aggregate src/direct_validation.py output.

Only scenarios for which ALL configurations finished without error are used, so every configuration is
evaluated on the same scenarios. Reading rule (fixed in the note): the lookup is refuted as a substitute
for direct execution if, for any quantity, the disagreement rate over scenario x configuration pairs
exceeds 5 %.

    python src/direct_validation_report.py --out results/final/direct_validation.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DE = 5                      # decision interval in simulator steps (campaign config: decide_every = 5)
QUANTITIES = ("fired", "step", "miss", "stop", "induced", "harm_run")
THRESH = 0.05


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/final/direct_validation*.jsonl")
    ap.add_argument("--out", default="results/final/direct_validation.json")
    a = ap.parse_args()

    recs = []
    for f in sorted(Path().glob(a.glob)):
        for line in f.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    by_seed = defaultdict(dict)
    n_err = 0
    for r in recs:
        if "error" in r:
            n_err += 1
        by_seed[r["seed"]][r["config"]] = r
    configs = sorted({c for d in by_seed.values() for c in d})
    complete = [s for s, d in by_seed.items() if len(d) == len(configs) and all("direct" in v for v in d.values())]

    rows, dis = [], []
    for s in sorted(complete):
        for c in configs:
            r = by_seed[s][c]
            lk, dr = r["lookup"], r["direct"]
            lk_step = lk["tick"] * DE if lk["fired"] else None
            row = dict(seed=s, config=c,
                       fired=(lk["fired"], dr["fired"]),
                       step=(lk_step, dr["step"]),
                       miss=(lk["miss"], dr["miss"]), stop=(lk["stop"], dr["stop"]),
                       induced=(lk["induced"], dr["induced"]), harm_run=(lk["harm_run"], dr["harm_run"]),
                       rc=(lk["rc"], dr["rc"]))
            rows.append(row)
            bad = [q for q in QUANTITIES if row[q][0] != row[q][1]]
            if bad:
                dis.append(dict(seed=s, config=c, quantities=bad, **{q: row[q] for q in bad}))

    n = len(rows)
    per_q = {}
    for q in QUANTITIES:
        k = sum(r[q][0] != r[q][1] for r in rows)
        per_q[q] = dict(disagree=k, rate=k / n, wilson_ci=wilson(k, n), refutes=bool(k / n > THRESH))
    rc_abs = np.array([abs(r["rc"][0] - r["rc"][1]) for r in rows])
    per_cfg = {}
    for c in configs:
        rr = [r for r in rows if r["config"] == c]
        per_cfg[c] = dict(
            n=len(rr), lam=by_seed[complete[0]][c]["lam"], lat=by_seed[complete[0]][c]["lat"],
            **{f"{q}_disagree": int(sum(r[q][0] != r[q][1] for r in rr)) for q in QUANTITIES},
            lookup_miss=float(np.mean([r["miss"][0] for r in rr])), direct_miss=float(np.mean([r["miss"][1] for r in rr])),
            lookup_stop=float(np.mean([r["stop"][0] for r in rr])), direct_stop=float(np.mean([r["stop"][1] for r in rr])),
            lookup_fired=float(np.mean([r["fired"][0] for r in rr])), direct_fired=float(np.mean([r["fired"][1] for r in rr])))
    out = dict(n_scenarios=len(complete), n_pairs=n, n_errors=n_err, configs=configs, per_quantity=per_q,
               route_completion=dict(mean_abs_diff=float(rc_abs.mean()), max_abs_diff=float(rc_abs.max()),
                                     n_exactly_equal=int((rc_abs == 0).sum())),
               per_config=per_cfg, disagreements=dis,
               verdict_V1_lookup_reproduces_direct=bool(not any(v["refutes"] for v in per_q.values())))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("n_scenarios", "n_pairs", "n_errors", "per_quantity", "route_completion",
                                          "verdict_V1_lookup_reproduces_direct")}, indent=1, default=float))
    print(f"{len(dis)} disagreeing pairs")
    for d in dis[:15]:
        print("  ", d)


if __name__ == "__main__":
    main()
