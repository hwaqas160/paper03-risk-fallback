"""
Progress of every campaign arm, and a completion test for run/campaign.cmd.

    python src/campaign_status.py          # table
    python src/campaign_status.py --quiet  # no output; exit code 0 iff EVERY arm is complete

An arm is complete when it has its target number of scenario rows AND every row carries
per-agent tables (inline, or via an agents_*.jsonl rescoring sidecar with a matching tick
count) -- rows collected before 2026-09-24 lack them and need the rescoring pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "campaign"
ARMS = {                      # name -> target scenario rows
    "av2_cal": 2000, "av2_test": 2000, "ns_val": 1500,
    "av2_test_replay": 500, "av2_test_mrm2": 500, "av2_test_10hz": 100, "waymo_val": 1000,
    # Amendment 5 (outcome-level certification, N1-N5): fresh, disjoint scenario ranges, verified
    # against the arms above (max touched index 3594/3583/2623) before being fixed in the amendment.
    "av2_cal5": 900, "av2_test5": 900,                      # N1-N3, primary predictor, d=4.0
    "av2_cal5_d2": 900, "av2_test5_d2": 900,                # N4, same seeds, d=2.0
    "ns_val5": 1300, "waymo_val5": 1000,                    # extension pools (not required by N1-N4)
    "av2_cal5_gpu": 900, "av2_test5_gpu": 900,              # N5, second predictor, same seeds as av2_*5
}


def arm_state(name: str, target: int) -> dict:
    d = ROOT / name
    rows, need_tables, errors = {}, 0, 0
    for f in sorted(d.glob("part_*.jsonl")) if d.exists() else []:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ref" in r and r["seed"] not in rows:
                    rows[r["seed"]] = (len(next(iter(r["ref"]["score_traces"].values()))),
                                       bool(r["ref"].get("agent_tables")))
    side = {}
    for f in sorted(d.glob("agents_*.jsonl")) if d.exists() else []:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                side[r["seed"]] = r["n_ticks"]
    for f in sorted(d.glob("skip_*.jsonl")) if d.exists() else []:
        with open(f) as fh:
            errors += sum('"error"' in line for line in fh)
    unmatched = 0
    for seed, (n_ticks, has) in rows.items():
        if has:
            continue
        if seed not in side:
            need_tables += 1          # never rescored yet -> work remains
        elif side[seed] != n_ticks:
            unmatched += 1            # rescored but the re-run differs in length: dropped (Amendment 2)
    return dict(name=name, target=target, rows=len(rows), need_tables=need_tables, errors=errors,
                unmatched=unmatched, complete=len(rows) >= target and need_tables == 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    states = [arm_state(n, t) for n, t in ARMS.items()]
    if not a.quiet:
        print(f"{'arm':<18}{'rows':>6}/{'target':<6}{'need tables':>13}{'unmatched':>10}{'errors':>8}  state")
        for s in states:
            print(f"{s['name']:<18}{s['rows']:>6}/{s['target']:<6}{s['need_tables']:>13}{s['unmatched']:>10}{s['errors']:>8}  "
                  f"{'COMPLETE' if s['complete'] else 'in progress'}")
    sys.exit(0 if all(s["complete"] for s in states) else 1)


if __name__ == "__main__":
    main()
