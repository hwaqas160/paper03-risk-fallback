"""
Sensitivity arms (notes/falsification.md Amendment 1.9 / 3.8): log-replay traffic, 2.0 m/s^2
comfort MRM, 10 Hz decisions. Each arm re-simulates a prefix of the SAME av2_test scenarios, so
every comparison is PAIRED by seed against the main reactive / 4 m/s^2 / 2 Hz run. The threshold is
the one certified on av2_cal under the main setting and is NOT re-tuned (the question is whether
the certificate transfers across simulator choices).

    python src/sensitivity.py --cal results/campaign/av2_cal --main results/campaign/av2_test \
        --arm replay=results/campaign/av2_test_replay --arm mrm2=results/campaign/av2_test_mrm2 \
        --arm hz10=results/campaign/av2_test_10hz --out results/final/sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from costs_and_luo import breakdown  # noqa: E402
from evaluate import Scenario, load_rows, ltt, matrices, summarize  # noqa: E402

KEY = "geom"
ALPHA, DELTA = 0.05, 0.10


def load(paths):
    return {r["seed"]: Scenario(r) for r in load_rows(paths)}


def per_scenario(scns, lam):
    M, S, I = matrices(scns, KEY, np.array([lam]))[:3]
    return M[:, 0], S[:, 0], I[:, 0]


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    b = d[idx].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", nargs="+", required=True)
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cal = [s for s in load(a.cal).values() if "conf_conflict" in s.traces]
    lam = ltt(cal, KEY, ALPHA, DELTA)
    main_rows = load(a.main)
    out = dict(lam=lam, arms={})
    print(f"AV2-certified LTT lambda (main setting) = {lam:.4f}; main test rows = {len(main_rows)}")
    print(f"{'arm':<8}{'n':>5}{'base harm':>10}{'miss':>7}{'CI':>17}{'unnec':>7}{'induced':>8}{'resid':>7} |"
          f"{'d miss':>8}{'95% CI':>17}{'d stop':>8}{'95% CI':>17}")
    for spec in a.arm:
        name, _, path = spec.partition("=")
        arm = load([path])
        seeds = sorted(set(arm) & set(main_rows))
        A = [arm[s] for s in seeds]
        B = [main_rows[s] for s in seeds]
        if not A:
            print(f"{name}: no overlapping seeds with the main run")
            continue
        sa = summarize(A, KEY, lam)
        bd = breakdown(A, lam)
        ma, sta, ia = per_scenario(A, lam)
        mb, stb, ib = per_scenario(B, lam)
        dm, ds = ma - mb, sta - stb
        out["arms"][name] = dict(
            n=len(A), base_harm_rate=float(np.mean([s.ref_harmful for s in A])),
            base_harm_rate_main=float(np.mean([s.ref_harmful for s in B])),
            miss=sa["miss"], miss_ci=sa["miss_ci"], unnecessary_stop=sa["unnecessary_stop"], induced=sa["induced"],
            residual_harm=bd["residual_harm"], main_miss=float(mb.mean()), main_stop=float(stb.mean()),
            main_induced=float(ib.mean()),
            paired_miss_diff=float(dm.mean()), paired_miss_diff_ci=boot_ci(dm),
            paired_stop_diff=float(ds.mean()), paired_stop_diff_ci=boot_ci(ds),
            certificate_transfers=bool(sa["miss_ci"][0] <= ALPHA or sa["miss"] <= ALPHA))
        o = out["arms"][name]
        print(f"{name:<8}{o['n']:>5}{o['base_harm_rate']:>10.3f}{o['miss']:>7.3f}"
              f"  [{o['miss_ci'][0]:.3f},{o['miss_ci'][1]:.3f}]{o['unnecessary_stop']:>7.3f}{o['induced']:>8.3f}"
              f"{o['residual_harm']:>7.3f} |{o['paired_miss_diff']:>+8.3f}"
              f"  [{o['paired_miss_diff_ci'][0]:+.3f},{o['paired_miss_diff_ci'][1]:+.3f}]"
              f"{o['paired_stop_diff']:>+8.3f}  [{o['paired_stop_diff_ci'][0]:+.3f},{o['paired_stop_diff_ci'][1]:+.3f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
