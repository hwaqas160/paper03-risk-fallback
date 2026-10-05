"""
Amendment 9, A1 (notes/falsification.md): leave-one-city-out certification inside Argoverse 2.
Reads stored rows only. Needs results/final/av2_city_labels.json (src/city_labels.py).

    python src/city_shift.py --out results/final/city_shift.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, ltt, summarize, tuned  # noqa: E402

ALPHA, DELTA, MIN_N = 0.05, 0.10, 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    pool, city = [], []
    for db in ("av2_cal", "av2_test"):
        for r in load_rows([f"results/campaign/{db}"]):
            s = Scenario(r)
            if "conf_conflict" not in s.traces:
                continue
            pool.append(s)
            city.append(lab[db][str(r["seed"])])
    city = np.array(city)
    cities = [c for c in sorted(set(city)) if (city == c).sum() >= MIN_N]
    lam_all = ltt(pool, "geom", ALPHA, DELTA)
    out = dict(n=len(pool), cities=cities, rows={}, lam_all_cities=lam_all)
    for c in cities:
        held = [s for s, k in zip(pool, city) if k == c]
        rest = [s for s, k in zip(pool, city) if k != c]
        lam, lam_t = ltt(rest, "geom", ALPHA, DELTA), tuned(rest, "geom", ALPHA)
        out["rows"][c] = dict(
            n_heldout=len(held), n_rest=len(rest),
            harm_rate=float(np.mean([s.ref_harmful for s in held])),
            ltt=dict(lam=lam, **summarize(held, "geom", lam)),
            tuned=dict(lam=lam_t, **summarize(held, "geom", lam_t)),
            in_pool_threshold=summarize(held, "geom", lam_all))
    ok = [c for c in cities if out["rows"][c]["ltt"]["miss"] <= ALPHA]
    ok_t = [c for c in cities if out["rows"][c]["tuned"]["miss"] <= ALPHA]
    out["verdict_A1"] = dict(ltt_cities_meeting_alpha=ok, tuned_cities_meeting_alpha=ok_t,
                             share_ltt=len(ok) / len(cities), share_tuned=len(ok_t) / len(cities),
                             transfers=bool(len(ok) / len(cities) >= 0.8))
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"pool n={len(pool)} cities={cities}")
    for c in cities:
        r = out["rows"][c]
        print(f"{c:14s} n={r['n_heldout']:4d} harm={r['harm_rate']:.3f} | LTT miss {r['ltt']['miss']:.3f} "
              f"[{r['ltt']['miss_ci'][0]:.3f},{r['ltt']['miss_ci'][1]:.3f}] stop {r['ltt']['unnecessary_stop']:.3f} | "
              f"tuned miss {r['tuned']['miss']:.3f} | in-pool miss {r['in_pool_threshold']['miss']:.3f}")
    print(out["verdict_A1"])


if __name__ == "__main__":
    main()
