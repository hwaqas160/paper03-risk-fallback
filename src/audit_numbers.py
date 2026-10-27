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

chk("TTC LTT stops", p5["T0a TTC, LTT"]["unnecessary_stop"])
chk("TTC LTT miss", p5["T0a TTC, LTT"]["miss"])
chk("audit UB residual", nc["verdict_NC"]["ub_res"])
chk("nuScenes domain AUROC", round(shift["shift"]["nuscenes"]["domain_auroc"], 2), "0.97")
chk("LTT corrected violation", p2["random_resplits_av2_pool"]["LTT"]["corrected"], "9.5")

hz = J("r12_hz10.json"); hzl = J("r12_hz10_posthoc_long.json")
res = {**hz["results"], **hzl["results"]}
for lat, lab in (("0", "0 s"), ("2", "0.2 s"), ("3", "0.3 s"), ("5", "0.5 s"), ("8", "0.8 s"), ("10", "1.0 s")):
    m = res[lat]["methods"]
    for k in ("TTC (10 Hz)", "RSS (10 Hz)", "learned (counterfactual labels)"):
        if m[k].get("stop") is not None:
            chk(f"10 Hz {lab} {k} stops", m[k]["stop"])
sup = J("r12_sup.json")
chk("supervised l=1 stops", sup["1"]["methods"]["supervised time-to-harm"]["test"]["stop"])
# Amendment 13: certifiability map (Table II, right block and the resplit sentence) and collision-only descriptive numbers
fr = J("r13_frontier.json"); co = J("r13_collonly.json")
for lat, lab in (("0", "0 s"), ("2", "0.2 s"), ("3", "0.3 s"), ("5", "0.5 s"), ("8", "0.8 s"), ("10", "1.0 s")):
    for n in ("TTC", "RSS", "geometric", "learned"):
        f_ = fr["frontier"][lat][n]
        row_ok = f_ is not None
        chk(f"frontier {lab} {n}", f"{100 * f_:g}" if row_ok else "none", f"{100 * f_:g}" if row_ok else "")
pu = fr["prob_useful_at_0.075_over_50_splits"]
chk("TTC certifies usefully at 0.5 s across resplits (%)", f"{100 * pu['5']['TTC']:.0f}", f"{100 * pu['5']['TTC']:.0f}" + "\,\% of them")
chk("geometric across resplits at 0.5 s (%)", f"{100 * pu['5']['geometric']:.0f}", f"{100 * pu['5']['geometric']:.0f}" + "\,\% for the geometric")
chk("RSS across resplits at 0.5 s (%)", f"{100 * pu['5']['RSS']:.0f}", f"{100 * pu['5']['RSS']:.0f}" + "\,\% for the RSS")
c3 = co["targets"]["0.03"]["0"]
chk("collision-only 3% TTC stops", c3["methods"]["TTC"]["test"]["stop"])
chk("collision-only 3% geometric stops", c3["methods"]["geometric"]["test"]["stop"])
# Amendment 14: outcome certificate under random and spatial-group resplits and without drifted scenarios
ov = J("r14_outvalid.json")
for k, nm in (("learned", "learned"), ("geometric", "geometric"), ("TTC", "TTC")):
    chk(f"A14 random-resplit violation {nm} (%)", f"{100 * ov['all'][k]['random_resplits']['uncorrected']:.1f}")
    chk(f"A14 group-resplit violation {nm} (%)", f"{100 * ov['all'][k]['group_resplits']['uncorrected']:.1f}")
chk("A14 drift-free max realized harm (%)", f"{100 * max(ov['drift_free'][k]['realized']['o_res'] for k in ov['drift_free']):.1f}")
chk("A14 drift-free learned stops (%)", f"{100 * ov['drift_free']['learned']['realized']['stop']:.1f}")
chk("A14 drift-free TTC stops (%)", f"{100 * ov['drift_free']['TTC']['realized']['stop']:.1f}")
# Amendment 15: cluster-aware LTT
cg = J("r15_cluster_ltt.json")["results"]["50.0"]; g15 = J("r15_groups.json")
chk("A15 plain group violation at 50 m (%)", f"{100 * cg['plain']['uncorrected']:.0f}", r"(0\,\% against " + f"{100 * cg['plain']['uncorrected']:.0f}" + r"\,\%)")
chk("A15 kish group violation at 50 m (%)", f"{100 * cg['kish']['uncorrected']:.0f}", r"(0\,\% against 14\,\%)")
chk("A15 stop cost (points, rounded)", f"{100 * cg['paired_stop_kish_minus_plain']['mean']:.0f}", "at 10 points more unnecessary stops")
chk("A15 n_eff at 150 m (about)", f"{g15['150.0']['n_eff_calibration_half']['mean']:.0f}", "about 10 scenarios")
chk("A15 largest group share at 150 m (%)", f"{100 * 649 / 3990:.0f}", r"holds 16\,\% of the pool")
sz_all, sz_df = ov["all_sizes"], ov["drift_free_sizes"]
chk("A14 drift-flagged share dropped (%)", f"{100 * (sum(sz_all) - sum(sz_df)) / sum(sz_all):.1f}", f"{100 * (sum(sz_all) - sum(sz_df)) / sum(sz_all):.1f}" + "\,\% of scenarios with flagged simulator")
bad = [c for c in checks if not c[2]]
for n, s, ok in checks:
    print(("ok   " if ok else "MISS ") + f"{n}: {s}")
sys.exit(1 if bad else 0)
