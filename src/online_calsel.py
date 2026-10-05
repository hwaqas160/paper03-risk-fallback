"""
Amendment 8, C1 (notes/falsification.md): CDT and ACI with the step size chosen on calibration data only.

    python src/online_calsel.py --out results/final/online_calsel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, aci, load_rows, online, pred_error_scores, tuned  # noqa: E402

ALPHA = 0.05
ETAS = (0.01, 0.05, 0.1, 0.5)
KEYS = ("miss", "unnecessary_stop", "induced", "autonomy", "route_completion")


def avg(runs):
    return {k: float(np.mean([r[k] for r in runs])) for k in KEYS}


def pick_cdt(avgs):
    best = None
    for eta, a in avgs.items():
        if best is None or (a["miss"] <= ALPHA and a["unnecessary_stop"] < best[1]["unnecessary_stop"]) \
                or (best[1]["miss"] > ALPHA and a["miss"] < best[1]["miss"]):
            best = (eta, a)
    return best[0]


def pick_aci(avgs):
    return min(avgs, key=lambda g: abs(avgs[g]["miss"] - ALPHA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = [s for s in (Scenario(r) for r in load_rows(["results/campaign/av2_cal"])) if "conf_conflict" in s.traces]
    test = [s for s in (Scenario(r) for r in load_rows(["results/campaign/av2_test"])) if "conf_conflict" in s.traces]
    pe = pred_error_scores("results/preds/av2cal_from_av2_cpu_v2.npz")
    rng = np.random.default_rng(0)
    oc = [rng.permutation(len(cal)) for _ in range(20)]
    ot = [rng.permutation(len(test)) for _ in range(20)]
    lam0 = tuned(cal, "geom", ALPHA)
    lam0 = lam0 if np.isfinite(lam0) else 0.0
    out = {}
    for name, fn_cal, fn_test, pick in (
            ("CDT", lambda e, o: online(cal, "geom", lam0, e, ALPHA, o), lambda e, o: online(test, "geom", lam0, e, ALPHA, o), pick_cdt),
            ("ACI", lambda g, o: aci(cal, pe, ALPHA, g, o), lambda g, o: aci(test, pe, ALPHA, g, o), pick_aci)):
        cal_avgs = {e: avg([fn_cal(e, o) for o in oc]) for e in ETAS}
        test_avgs = {e: avg([fn_test(e, o) for o in ot]) for e in ETAS}
        e_cal, e_test = pick(cal_avgs), pick(test_avgs)
        out[name] = dict(step_selected_on_calibration=e_cal, step_selected_on_test=e_test,
                         deployed_calsel=test_avgs[e_cal], deployed_testsel=test_avgs[e_test],
                         calibration_curve=cal_avgs, test_curve=test_avgs)
        c, t = test_avgs[e_cal], test_avgs[e_test]
        print(f"{name}: cal-selected step {e_cal}: miss {c['miss']:.3f} stop {c['unnecessary_stop']:.3f} | "
              f"test-selected step {e_test}: miss {t['miss']:.3f} stop {t['unnecessary_stop']:.3f}")
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
