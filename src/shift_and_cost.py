"""
Descriptive additions prompted by an external review (notes/falsification.md, Amendment 7, V2/V3).
Reads stored rows only; nothing is re-simulated, and no hypothesis is tested.

  V2  Shift severity: cross-validated AUROC of the domain classifier used for the density ratio
      (av2_cal vs each target; av2_cal vs av2_test is the null control), Kish n_eff of the weights,
      harm base rate, first-tick floor, and the miss / stops of the AV2-certified threshold.
  V3  Cost of the intervention at the certified threshold on av2_test: induced-collision rate by
      initial ego speed tercile and by firing time, collision types, stopped time.

    python src/shift_and_cost.py --out results/final/shift_and_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, ltt, ltt_weighted, summarize  # noqa: E402
from rollout import FEATURE_NAMES  # noqa: E402

ALPHA, DELTA = 0.05, 0.10
ARMS = dict(av2_cal="results/campaign/av2_cal", av2_test="results/campaign/av2_test",
            nuscenes="results/campaign/ns_val", waymo="results/campaign/waymo_val")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def domain_auroc(src, tgt):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([np.stack([s.feats for s in src]), np.stack([s.feats for s in tgt])])
    y = np.r_[np.zeros(len(src)), np.ones(len(tgt))]
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y,
                          cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from conformal import domain_classifier_weights
    rows = {k: load_rows([p]) for k, p in ARMS.items()}
    scn = {k: [Scenario(r) for r in v] for k, v in rows.items()}
    # same inclusion rule as the main analyses (per-agent records matched: conf_conflict trace present)
    scn = {k: [s for s in v if s.feats.size and "conf_conflict" in s.traces] for k, v in scn.items()}
    cal = scn["av2_cal"]
    lam = ltt(cal, "geom", ALPHA, DELTA)
    out = dict(lam_ltt=lam, features=FEATURE_NAMES, shift={}, cost={})

    for k in ("av2_test", "nuscenes", "waymo"):
        t = scn[k]
        s = summarize(t, "geom", lam)
        w = domain_classifier_weights(np.stack([x.feats for x in cal]), np.stack([x.feats for x in t]))
        _, n_eff = ltt_weighted(cal, "geom", ALPHA, DELTA, w)
        kish_all = float(w.sum() ** 2 / (w ** 2).sum())
        floor = float(np.mean([x.miss[0] for x in t]))   # harm no trigger can pre-empt: missed even if fired at tick 0
        out["shift"][k] = dict(
            n=len(t), domain_auroc=domain_auroc(cal, t), kish_n_eff_all_weights=kish_all, n_eff_ltt=n_eff,
            harm_rate=float(np.mean([x.ref_harmful for x in t])), first_tick_floor=floor,
            miss=s["miss"], miss_ci=s["miss_ci"], unnecessary_stop=s["unnecessary_stop"])

    # ---- V2b: scene-level dependence on nuScenes (the scene name is in the scenario id). The cluster
    # bootstrap resamples whole scenes; if scenarios within a scene were strongly dependent its
    # interval is much wider than the Wilson one, which treats scenarios as independent. ns_val is the
    # main arm; ns_val5 is a second, largely disjoint set of scenes (Amendment 5 collection).
    import re
    from scenarionet.common_utils import read_dataset_summary
    import simenv
    _, lst, _ = read_dataset_summary(simenv.DBS["ns_val"])
    out["nuscenes_scene_cluster"] = {}
    for arm, path in (("ns_val", ARMS["nuscenes"]), ("ns_val5", "results/campaign/ns_val5")):
        tn = [Scenario(r) for r in load_rows([path])]
        scene = np.array([re.search(r"scene-\d+", lst[x.seed]).group(0) for x in tn])
        tk = [int(x.tick("geom", lam)[0]) for x in tn]
        miss = np.array([x.miss[k] for x, k in zip(tn, tk)])
        floor = float(np.mean([x.miss[0] for x in tn]))
        uniq, inv = np.unique(scene, return_inverse=True)
        sums = np.bincount(inv, weights=miss); cnt = np.bincount(inv).astype(float)
        idx = np.random.default_rng(0).integers(0, len(uniq), size=(10000, len(uniq)))
        boots = sums[idx].sum(1) / cnt[idx].sum(1)
        lo, hi = (float(q) for q in np.quantile(boots, [.025, .975]))
        out["nuscenes_scene_cluster"][arm] = dict(
            n_scenarios=len(tn), n_scenes=int(len(uniq)), miss=float(miss.mean()), first_tick_floor=floor,
            wilson_ci=wilson(int(miss.sum()), len(miss)), cluster_bootstrap_ci=[lo, hi], cluster_ci_excludes_alpha=bool(lo > ALPHA),
            scenes=sorted(set(uniq.tolist())))
    a_, b_ = (set(out["nuscenes_scene_cluster"][k].pop("scenes")) for k in ("ns_val", "ns_val5"))
    out["nuscenes_scene_cluster"]["scenes_shared"] = len(a_ & b_)

    # ---- V3: where do the induced collisions come from? (av2_test, certified threshold)
    t = scn["av2_test"]
    rws = {r["seed"]: r for r in rows["av2_test"]}
    spd = np.array([x.feats[FEATURE_NAMES.index("ego_speed0")] for x in t])
    tick = np.array([int(x.tick("geom", lam)[0]) for x in t])
    fired = tick < np.array([x.n_ticks for x in t])
    ind = np.array([x.induced[tk] for x, tk in zip(t, tick)])
    cuts = np.quantile(spd, [1 / 3, 2 / 3])
    bins = np.digitize(spd, cuts)
    by_speed = []
    for b, name in enumerate(["low", "mid", "high"]):
        m = bins == b
        by_speed.append(dict(tercile=name, speed_range=[float(spd[m].min()), float(spd[m].max())], n=int(m.sum()),
                             fired=float(fired[m].mean()), induced=float(ind[m].mean()),
                             induced_ci=wilson(int(ind[m].sum()), int(m.sum()))))
    med = float(np.median(tick[fired]))
    by_time = []
    for name, m in (("early (tick <= median)", fired & (tick <= med)), ("late (tick > median)", fired & (tick > med))):
        by_time.append(dict(group=name, n=int(m.sum()), induced=float(ind[m].mean()),
                            induced_ci=wilson(int(ind[m].sum()), int(m.sum()))))
    types = Counter()
    stopped, rc_fired = [], []
    for x, tk, f in zip(t, tick, fired):
        if not f:
            continue
        rec = rws[x.seed]["fired"].get(str(tk * rws[x.seed]["decide_every"]))
        if rec is None:
            continue
        stopped.append(rec["stopped_steps"] * 0.1)
        rc_fired.append(rec["route_completion"])
        if x.induced[tk]:
            for c in rec["collisions"]:
                if not c["at_fault"]:
                    types[c["type"]] += 1
    out["cost"] = dict(n=len(t), fired=float(fired.mean()), induced=float(ind.mean()),
                       induced_ci=wilson(int(ind.sum()), len(ind)), by_speed=by_speed, by_firing_time=by_time,
                       median_firing_tick=med, induced_collision_types=dict(types),
                       mean_stopped_s_when_fired=float(np.mean(stopped)),
                       mean_route_completion_when_fired=float(np.mean(rc_fired)),
                       mean_route_completion_never_fired=float(np.mean(
                           [x.rc[-1] for x, f in zip(t, fired) if not f])))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
