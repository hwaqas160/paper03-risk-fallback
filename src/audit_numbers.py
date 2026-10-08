"""Check headline numbers quoted in paper/main.tex against the result files. Prints mismatches; exit code 1 if any."""
import json, re, sys
from pathlib import Path
R = Path("results/final")
tex = Path("paper/main.tex").read_text(encoding="utf-8")
J = lambda f: json.loads((R / f).read_text())
pct = lambda x: f"{100 * x:.1f}"
ind, lat, p2, p5, nc, shift = J("r11_cswc_indist.json"), J("r11_latency_outcome.json"), J("r11_p2_validity.json"), J("r11_p5_ttc.json"), J("r11_nc_audit.json"), J("shift_and_cost.json")
checks = []
def chk(name, val, must_contain=None):
    s = pct(val) if isinstance(val, float) else str(val)
    ok = (must_contain or s) in tex
    checks.append((name, s, ok))
m = ind["methods"]
chk("CSWC zero-latency O_res", m["CSWC"]["res_target"]["test"]["o_res"])
chk("CSWC zero-latency stops", m["CSWC"]["res_target"]["test"]["stop"])
chk("geometric zero-latency stops", m["geometric"]["res_target"]["test"]["stop"])
chk("TTC zero-latency stops", m["TTC (T0a)"]["res_target"]["test"]["stop"])
for l in ("1", "2"):
    for k in ("CSWC", "geometric", "TTC (T0a)"):
        v = lat[l]["methods"][k]["test"]["stop"]
        chk(f"latency {l} {k} stops", v)
chk("TTC LTT stops", p5["T0a TTC, LTT"]["unnecessary_stop"])
chk("TTC LTT miss", p5["T0a TTC, LTT"]["miss"])
chk("TTC lat1 miss", p5["T0a TTC, LTT"]["latency"]["1"]["miss"])
chk("TTC lat2 miss", p5["T0a TTC, LTT"]["latency"]["2"]["miss"])
chk("audit UB residual", nc["verdict_NC"]["ub_res"])
chk("nuScenes TTC miss", p5["shift"]["nuscenes"]["miss"])
chk("nuScenes domain AUROC", round(shift["shift"]["nuscenes"]["domain_auroc"], 2), "0.97")
chk("LTT corrected violation", p2["random_resplits_av2_pool"]["LTT"]["corrected"], "9.5")
bad = [c for c in checks if not c[2]]
for n, s, ok in checks:
    print(("ok   " if ok else "MISS ") + f"{n}: {s}")
sys.exit(1 if bad else 0)
