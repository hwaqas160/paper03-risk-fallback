"""
Amendment 8 (notes/falsification.md), L1: LTT calibrated under the deployed actuation latency.

    python src/latency_cert.py --out results/final/latency_cert.json
Reads stored rows only; nothing is re-simulated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, lam_grid, load_rows, matrices, summarize  # noqa: E402
from risk_control import learn_then_test  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def ltt_lat(cal, lat, grid):
    M = matrices(cal, "geom", grid, lat)[0]
    r = learn_then_test(grid, M, ALPHA, DELTA)
    return -np.inf if r["lam_hat"] is None else float(r["lam_hat"])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    cal = [Scenario(r) for r in load_rows(["results/campaign/av2_cal"])]
    test = [Scenario(r) for r in load_rows(["results/campaign/av2_test"])]
    cal = [s for s in cal if "conf_conflict" in s.traces]
    test = [s for s in test if "conf_conflict" in s.traces]
    grid = lam_grid(cal, "geom")
    lam0 = ltt_lat(cal, 0, grid)
    out = dict(n_cal=len(cal), n_test=len(test), alpha=ALPHA, delta=DELTA, headline={}, validity={})
    for lat in (0, 1, 2):
        lam = ltt_lat(cal, lat, grid)
        out["headline"][str(lat)] = dict(
            certified=bool(np.isfinite(lam)), lam_lat_aware=lam,
            lat_aware=summarize(test, "geom", lam, lat) if np.isfinite(lam) else None,
            lam_zero_latency=lam0, zero_latency_cert=summarize(test, "geom", lam0, lat))
    pool = cal + test
    rng = np.random.default_rng(0)
    viol = {f"{k}{l}": [] for l in (0, 1, 2) for k in ("aware", "naive")}
    refused = {l: 0 for l in (0, 1, 2)}
    stops = {l: [] for l in (0, 1, 2)}
    for _ in range(a.reps):
        p = rng.permutation(len(pool)); c = [pool[i] for i in p[:len(pool) // 2]]; t = [pool[i] for i in p[len(pool) // 2:]]
        g = lam_grid(c, "geom")
        l0 = ltt_lat(c, 0, g)
        for lat in (0, 1, 2):
            la = ltt_lat(c, lat, g)
            if not np.isfinite(la):
                refused[lat] += 1
            else:
                s = summarize(t, "geom", la, lat)
                viol[f"aware{lat}"].append(s["miss"] > ALPHA); stops[lat].append(s["unnecessary_stop"])
            viol[f"naive{lat}"].append(summarize(t, "geom", l0, lat)["miss"] > ALPHA if np.isfinite(l0) else False)
    for lat in (0, 1, 2):
        va, vn = viol[f"aware{lat}"], viol[f"naive{lat}"]
        out["validity"][str(lat)] = dict(
            aware_violation=float(np.mean(va)) if va else None, aware_n=len(va),
            aware_ci=wilson(int(sum(va)), len(va)) if va else None, aware_refused=refused[lat],
            aware_stop_mean=float(np.mean(stops[lat])) if stops[lat] else None,
            naive_violation=float(np.mean(vn)), naive_ci=wilson(int(sum(vn)), len(vn)))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for lat in ("0", "1", "2"):
        h, v = out["headline"][lat], out["validity"][lat]
        la = h["lat_aware"]
        print(f"lat {lat}: aware lam={h['lam_lat_aware']:.3f} miss={la['miss']:.3f} stop={la['unnecessary_stop']:.3f} "
              f"viol={v['aware_violation']} (refused {v['aware_refused']}) | naive miss={h['zero_latency_cert']['miss']:.3f} "
              f"viol={v['naive_violation']:.3f}")


if __name__ == "__main__":
    main()
