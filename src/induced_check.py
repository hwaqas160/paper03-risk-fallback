"""Amendment 8, I1: when do induced collisions happen relative to the fallback firing and the ego stopping?"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import Scenario, load_rows, ltt

cal = [s for s in (Scenario(r) for r in load_rows(["results/campaign/av2_cal"])) if "conf_conflict" in s.traces]
lam = ltt(cal, "geom", 0.05, 0.10)
rows = [r for r in load_rows(["results/campaign/av2_test"])]
DT = 0.1
ev = []
for r in rows:
    sc = Scenario(r)
    if "conf_conflict" not in sc.traces:
        continue
    t = int(sc.tick("geom", lam)[0])
    if t >= sc.n_ticks or not sc.induced[t]:
        continue
    f = r["fired"][str(t * r["decide_every"])]
    ref_nonfault = any(not c["at_fault"] for c in r["ref"]["collisions"])
    for c in f["collisions"]:
        if c["at_fault"]:
            continue
        ev.append(dict(seed=r["seed"], type=c["type"], dt_s=(c["step"] - f["trigger_step"]) * DT,
                       stopped_s_total=f["stopped_steps"] * DT, speed0=float(sc.feats[3]),
                       fire_s=f["trigger_step"] * DT, ref_has_nonfault=ref_nonfault))
dt = np.array([e["dt_s"] for e in ev])
out = dict(lam=lam, n_events=len(ev), n_scenarios=len({e["seed"] for e in ev}),
           types={k: sum(e["type"] == k for e in ev) for k in {e["type"] for e in ev}},
           dt_after_fire_quantiles=dict(zip(["min", "q10", "q25", "median", "q75", "q90", "max"],
                                            np.quantile(dt, [0, .1, .25, .5, .75, .9, 1]).tolist())),
           share_within_1s=float((dt <= 1).mean()), share_within_2s=float((dt <= 2).mean()),
           share_after_5s=float((dt > 5).mean()),
           share_fired_at_standstill=float(np.mean([e["speed0"] < 0.1 for e in ev])),
           median_speed0=float(np.median([e["speed0"] for e in ev])))
Path("results/final/induced_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
