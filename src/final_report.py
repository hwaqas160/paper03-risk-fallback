"""
Builds results/final/FINAL_SUMMARY.md from whatever analysis outputs exist, applying the
PRE-REGISTERED verdict rules mechanically (notes/falsification.md: H1-H3, Amendment 3 H3-W and the
generality rule, Amendment 4B R1-R4) so nobody has to re-interpret results after the fact.
Safe to run on partial data: missing pieces are reported as "pending".

    python src/final_report.py            # writes results/final/FINAL_SUMMARY.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
ALPHA = 0.05


def _load(name):
    p = FINAL / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def h1_verdict(ev):
    settings = {"AV2": ev.get("h1"), "nuScenes": ev.get("h1_shift"), "Waymo": ev.get("h1_waymo")}
    rows, refuted_everywhere, seen = [], True, 0
    for k, h in settings.items():
        if not h:
            rows.append(f"| {k} | pending | | |")
            continue
        seen += 1
        ref = h["kendall_tau"] >= 0.8 and h["same_top"]
        refuted_everywhere &= ref
        rows.append(f"| {k} | {h['kendall_tau']:.2f} | {h['same_top']} | {'rule met (refuting)' if ref else 'not refuted here'} |")
    verdict = ("pending" if not seen else ("REFUTED (all settings)" if refuted_everywhere else "NOT refuted (weak: tau < 0.8 or top differs in >= 1 setting)"))
    return verdict, rows


def h2_verdict(ev):
    h2 = ev.get("h2")
    if not h2:
        return "pending", []
    rows, refuted = [], False
    for name, v in h2.items():
        not_worse_3pp = v["ci"][1] < 0.03          # baseline not shown >= 3pp worse than T4
        refuted |= not_worse_3pp
        rows.append(f"| {name} | {v['diff']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'within/better (refutes H2)' if not_worse_3pp else 'worse by >= 3pp'} |")
    t4 = ev["headline"]["T4 LTT (ours)"]
    if t4["miss_ci"][0] > ALPHA:
        refuted = True
    return ("REFUTED" if refuted else "SUPPORTED"), rows


def h3_verdict(ev, key):
    sh = ev.get(key)
    if not sh:
        return "pending", None
    u = sh["unweighted LTT"]
    lo, hi = u["miss_ci"]
    return ("SUPPORTED (CI excludes alpha)" if not (lo <= ALPHA <= hi) else "REFUTED (CI includes alpha)"), u


def build() -> str:
    ev = _load("eval_all.json") or _load("eval_main.json")
    cl = _load("costs_luo_all.json") or _load("costs_luo_main.json")
    rw = _load("recal_waymo.json")
    rn = _load("recal_nuscenes_exploratory.json")
    se = _load("sensitivity.json")
    L = [f"# FINAL SUMMARY (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
         "Verdicts are computed mechanically from the pre-registered rules in notes/falsification.md. "
         "Anything marked pending needs data that has not finished collecting.", ""]
    if not ev:
        L.append("No evaluation output yet (results/final/eval_*.json missing).")
        return "\n".join(L) + "\n"
    L += [f"Scenarios: cal n={ev['n_cal']}, test n={ev['n_test']}, nuScenes n={ev.get('n_shift', 'n/a')}, "
          f"Waymo n={ev.get('n_waymo', 'pending')}", ""]
    hd = ev["headline"]
    L += ["## Headline (AV2 test, alpha = 0.05)", "", "| Method | Miss | Unnecessary stops | Induced |", "|---|---|---|---|"]
    for k, v in hd.items():
        L.append(f"| {k} | {_pct(v['miss'])} | {_pct(v['unnecessary_stop'])} | {_pct(v['induced'])} |")
    val = ev.get("validity", {})
    if val:
        L += ["", "Validity over resplits (violation frequency, valid if <= 10%): " +
              "; ".join(f"{k}: {100 * v['violation_freq']:.1f}%" for k, v in val.items())]
    v1, r1 = h1_verdict(ev)
    v2, r2 = h2_verdict(ev)
    v3, u3 = h3_verdict(ev, "shift")
    v3w, u3w = h3_verdict(ev, "shift_waymo")
    L += ["", "## Pre-registered verdicts", "",
          f"- **H1** (open-loop vs closed-loop ranking): {v1}",
          f"- **H2** (certified beats best tuned baseline): {v2}",
          f"- **H3** (AV2 certificate fails on nuScenes): {v3}" + ("" if not u3 else f" — miss {_pct(u3['miss'])} CI [{_pct(u3['miss_ci'][0])}, {_pct(u3['miss_ci'][1])}]"),
          f"- **H3-W** (fails on Waymo): {v3w}" + ("" if not u3w else f" — miss {_pct(u3w['miss'])} CI [{_pct(u3w['miss_ci'][0])}, {_pct(u3w['miss_ci'][1])}]")]
    sup = [v3.startswith("SUPPORTED"), v3w.startswith("SUPPORTED")]
    if "pending" in (v3, v3w):
        gen = "pending (needs both targets)"
    elif all(sup):
        gen = "CLAIMABLE: the guarantee fails under real cross-dataset shift (both targets)"
    elif any(sup):
        gen = "DATASET-DEPENDENT: fails on exactly one target; abstract must say so"
    else:
        gen = "NOT SUPPORTED on either target"
    L += [f"- **Generality rule (Amendment 3.3):** {gen}", "", "### H1 detail", "", "| Setting | Kendall tau | Same top | Note |", "|---|---|---|---|"] + r1
    L += ["", "### H2 detail (paired bootstrap: baseline stops - T4 stops)", "", "| Baseline | Diff | 95% CI | Reading |", "|---|---|---|---|"] + r2
    if rw:
        L += ["", "## Waymo recalibration study (pre-registered R1-R4, primary = all scenarios)", ""]
        for k, v in rw["verdicts_primary_all_scenarios"].items():
            L.append(f"- {k}: **{'HOLDS' if v['holds'] else 'REFUTED'}** ({json.dumps({a: b for a, b in v.items() if a != 'holds'}, default=float)})")
    else:
        L += ["", "## Waymo recalibration study: pending"]
    if rn:
        L += ["", "(nuScenes recalibration replicate is exploratory, in results/final/recal_nuscenes_exploratory.json)"]
    if cl:
        L += ["", "## Consequences of the intervention (post hoc)", "", "| Data | Method | Marg. miss | Cond. FNR | Unnec. | Residual harm | New harm | Induced |", "|---|---|---|---|---|---|---|---|"]
        for ds, rows in cl.items():
            if not isinstance(rows, list):
                continue
            for r in rows:
                L.append(f"| {ds} | {r['method']} | {_pct(r['miss'])} | {_pct(r['cond_fnr'])} | {_pct(r['unnecessary_stop'])} | {_pct(r['b_residual_harm'])} | {_pct(r['b_new_harm'])} | {_pct(r['b_induced_any'])} |")
    if se:
        L += ["", "## Sensitivity arms (AV2-certified threshold, paired by seed vs main run)", "",
              "| Arm | n | Base harm | Miss [CI] | Unnec. | Induced | dMiss [CI] | dStop [CI] |", "|---|---|---|---|---|---|---|---|"]
        for nm, o in se["arms"].items():
            L.append(f"| {nm} | {o['n']} | {_pct(o['base_harm_rate'])} | {_pct(o['miss'])} [{_pct(o['miss_ci'][0])}, {_pct(o['miss_ci'][1])}] | "
                     f"{_pct(o['unnecessary_stop'])} | {_pct(o['induced'])} | {o['paired_miss_diff']:+.3f} [{o['paired_miss_diff_ci'][0]:+.3f}, {o['paired_miss_diff_ci'][1]:+.3f}] | "
                     f"{o['paired_stop_diff']:+.3f} [{o['paired_stop_diff_ci'][0]:+.3f}, {o['paired_stop_diff_ci'][1]:+.3f}] |")
    else:
        L += ["", "## Sensitivity arms: pending"]
    L += ["", "## Next step for a human / Claude session",
          "Paste these numbers into paper/main.tex (Waymo subsection, sensitivity subsection, abstract/conclusion `[pending]` markers), "
          "recompile, and record outcomes in the Outcome log of notes/falsification.md."]
    return "\n".join(L) + "\n"


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "FINAL_SUMMARY.md").write_text(build(), encoding="utf-8")
    print(f"wrote {FINAL / 'FINAL_SUMMARY.md'}")


if __name__ == "__main__":
    sys.exit(main())
