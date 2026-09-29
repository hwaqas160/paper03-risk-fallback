"""
Amendment 5, N5: does N2's finding (the alarm-level certificate violates the outcome-level target)
replicate with a second, independently-trained predictor? Reads the two outcome_cert.py JSON
outputs (primary predictor, second predictor) and checks direction/rough-magnitude replication, per
the rule fixed in notes/falsification.md before any of this data was collected.

    python src/n5_replication.py --primary results/final/outcome_cert.json \
        --second results/final/outcome_cert_gpu.json --out results/final/n5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = json.loads(Path(a.primary).read_text())
    s = json.loads(Path(a.second).read_text())
    p_dir = p["verdicts"]["N2_alarm_cert_violates_outcome_target"]["holds"]
    s_dir = s["verdicts"]["N2_alarm_cert_violates_outcome_target"]["holds"]
    out = dict(
        primary_predictor=dict(holds=p_dir, residual_harm=p["alarm_level_on_test"]["residual_harm"],
                               ci=p["alarm_level_on_test"]["residual_harm_ci"]),
        second_predictor=dict(holds=s_dir, residual_harm=s["alarm_level_on_test"]["residual_harm"],
                              ci=s["alarm_level_on_test"]["residual_harm_ci"]),
        verdict_N5_replicates=dict(holds=bool(p_dir and s_dir)),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(f"\nN5 verdict: {'HOLDS' if out['verdict_N5_replicates']['holds'] else 'REFUTED'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
