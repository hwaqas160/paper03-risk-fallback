"""
Amendment 10, G1 (notes/falsification.md): spatial-group dependence. Groups = single-linkage components of the ego start
points within EPS metres (per city for Argoverse 2). Reads stored rows and the ScenarioNet pickles only.

    python src/spatial_groups.py --out results/final/spatial_groups.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simenv  # noqa: E402
from evaluate import Scenario, load_rows, ltt, summarize  # noqa: E402
from scenarionet.common_utils import read_dataset_summary  # noqa: E402

ALPHA, DELTA = 0.05, 0.10


def ego_xy(db, seeds):
    _, lst, mapping = read_dataset_summary(simenv.DBS[db])
    out = {}
    for sd in seeds:
        f = lst[sd]
        path = Path(simenv.DBS[db]) / (mapping[f] if mapping else "") / f
        d = pickle.load(open(path, "rb"))
        sdc = d["metadata"]["sdc_id"]
        out[sd] = np.asarray(d["tracks"][sdc]["state"]["position"][0][:2], float)
    return out


def components(xy, eps):
    tree = cKDTree(xy)
    pairs = np.array(list(tree.query_pairs(eps)))
    n = len(xy)
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def group_resplits(pool, grp, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(grp)
    idx = {g: np.flatnonzero(grp == g) for g in ids}
    viol, miss, n_cal = [], [], []
    for _ in range(reps):
        order = rng.permutation(ids)
        cal_i, tot = [], 0
        for g in order:
            if tot >= len(pool) // 2:
                break
            cal_i += idx[g].tolist(); tot += len(idx[g])
        mask = np.zeros(len(pool), bool); mask[cal_i] = True
        cal = [s for s, m in zip(pool, mask) if m]; test = [s for s, m in zip(pool, mask) if not m]
        lam = ltt(cal, "geom", ALPHA, DELTA)
        m = summarize(test, "geom", lam)["miss"] if np.isfinite(lam) else np.nan
        miss.append(m); n_cal.append(len(cal))
        viol.append(bool(m > ALPHA) if np.isfinite(m) else False)
    return dict(violation=float(np.mean(viol)), miss_mean=float(np.nanmean(miss)), cal_size_mean=float(np.mean(n_cal)))


def cluster_boot(miss, grp, B=10000, seed=0):
    ids, inv = np.unique(grp, return_inverse=True)
    s, c = np.bincount(inv, weights=miss), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(ids), size=(B, len(ids)))
    b = s[k].sum(1) / c[k].sum(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[150.0, 50.0])
    a = ap.parse_args()
    lab = json.loads(Path("results/final/av2_city_labels.json").read_text())
    ld = lambda p: [s for s in (Scenario(r) for r in load_rows([p])) if "conf_conflict" in s.traces]
    pools = {"av2_cal": ld("results/campaign/av2_cal"), "av2_test": ld("results/campaign/av2_test"),
             "waymo": ld("results/campaign/waymo_val")}
    xy = {"av2_cal": ego_xy("av2_cal", [s.seed for s in pools["av2_cal"]]),
          "av2_test": ego_xy("av2_test", [s.seed for s in pools["av2_test"]]),
          "waymo": ego_xy("waymo_val", [s.seed for s in pools["waymo"]])}
    out = {}
    for eps in a.eps:
        res = {}
        # --- Argoverse 2: groups per city on the pooled cal + test scenarios
        pool = pools["av2_cal"] + pools["av2_test"]
        pts = np.stack([xy["av2_cal"][s.seed] for s in pools["av2_cal"]] + [xy["av2_test"][s.seed] for s in pools["av2_test"]])
        city = np.array([lab["av2_cal"][str(s.seed)] for s in pools["av2_cal"]] + [lab["av2_test"][str(s.seed)] for s in pools["av2_test"]])
        grp = np.zeros(len(pool), int); off = 0
        for c in sorted(set(city)):
            m = city == c
            g = components(pts[m], eps); grp[m] = g + off; off += g.max() + 1
        sizes = np.bincount(grp)
        res["av2"] = dict(n=len(pool), n_groups=int(len(sizes)), largest=int(sizes.max()), median=float(np.median(sizes)),
                          share_largest=float(sizes.max() / len(pool)),
                          singletons=int((sizes == 1).sum()))
        if sizes.max() / len(pool) <= 0.25:
            res["av2"]["group_resplits"] = group_resplits(pool, grp)
        else:
            res["av2"]["group_resplits"] = None
        # --- held-out test miss of the certified threshold, group-bootstrap CI
        cal = pools["av2_cal"]; test = pools["av2_test"]
        lam = ltt(cal, "geom", ALPHA, DELTA)
        gt = grp[len(cal):]
        miss_t = np.array([s.miss[int(s.tick("geom", lam)[0])] for s in test])
        res["av2_test_miss"] = dict(miss=float(miss_t.mean()), wilson=summarize(test, "geom", lam)["miss_ci"],
                                    group_bootstrap_ci=cluster_boot(miss_t, gt), n_groups=int(len(np.unique(gt))))
        # --- Waymo
        w = pools["waymo"]
        pw = np.stack([xy["waymo"][s.seed] for s in w]); gw = components(pw, eps)
        sw = np.bincount(gw)
        miss_w = np.array([s.miss[int(s.tick("geom", lam)[0])] for s in w])
        res["waymo"] = dict(n=len(w), n_groups=int(len(sw)), largest=int(sw.max()), median=float(np.median(sw)),
                            miss=float(miss_w.mean()), wilson=summarize(w, "geom", lam)["miss_ci"],
                            group_bootstrap_ci=cluster_boot(miss_w, gw))
        out[str(eps)] = res
        print(eps, json.dumps(res, default=float)[:1500], flush=True)
    Path(a.out).write_text(json.dumps(dict(alpha=ALPHA, delta=DELTA, results=out), indent=2, default=float))


if __name__ == "__main__":
    main()
