"""
Paper 03's OWN measurement of conformal coverage under the AV2 -> nuScenes shift.

Why this file exists: H3 (notes/falsification.md) claims a threshold certified on AV2
loses its guarantee under dataset shift. That claim must not rest on citing Paper 01's
(unpublished) result — a reviewer, or a desk-reject of Paper 01, would leave H3 without a
citation. This script recomputes the coverage numbers from scratch, using only:
  (a) src/conformal.py — copied (not imported across projects) from Paper 01, since split
      conformal prediction is a standard method (Vovk, Gammerman & Shafer 2005), not a
      Paper 01 contribution — citing the METHOD needs no citation to Paper 01 at all;
  (b) raw model prediction dumps (pred_trajs / gt / gt_mask arrays) — the same kind of
      artifact as a downloaded pretrained-model output, not a "finding" of Paper 01's.

AutoBot itself (Kim et al., ICLR 2022) is cited as the predictor architecture; the specific
checkpoint is described in Paper 03's own methods (data, training set, minADE) exactly as a
paper would describe any pretrained backbone it uses. Paper 03 depends on Paper 01's
INFRASTRUCTURE (converted data, a trained checkpoint) but asserts no claim that requires
Paper 01 to be published, reviewed, or even cited as a paper.

    python src/coverage_selfcheck.py --preds F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\results\\preds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from conformal import SplitConformal, nonconformity_scores


def load(preds_dir: Path, tag: str):
    z = np.load(preds_dir / f"{tag}.npz")
    return z["pred_trajs"], z["gt"], z["gt_mask"].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="directory of raw *_from_<ckpt>.npz prediction dumps")
    ap.add_argument("--cal_tag", default="av2cal_from_av2_cpu_v1")
    ap.add_argument("--test_tag", default="av2test_from_av2_cpu_v1")
    ap.add_argument("--shift_tag", default="ns_from_av2_cpu_v1")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = ap.parse_args()

    preds_dir = Path(args.preds)
    pt_cal, gt_cal, m_cal = load(preds_dir, args.cal_tag)
    pt_in, gt_in, m_in = load(preds_dir, args.test_tag)
    pt_sh, gt_sh, m_sh = load(preds_dir, args.shift_tag)

    s_cal = nonconformity_scores(pt_cal, gt_cal, m_cal)
    s_in = nonconformity_scores(pt_in, gt_in, m_in)
    s_sh = nonconformity_scores(pt_sh, gt_sh, m_sh)
    print(f"calibration n={len(s_cal)} ({args.cal_tag})")
    print(f"in-domain   n={len(s_in)}  ({args.test_tag})")
    print(f"shifted     n={len(s_sh)}  ({args.shift_tag})\n")

    print(f"{'alpha':>6} {'nominal':>8} {'in-domain cov':>14} {'shifted cov':>12} {'drop (pp)':>10}")
    rows = []
    for a in args.alphas:
        sc = SplitConformal(alpha=a).calibrate(s_cal)
        r_in = sc.evaluate(s_in)
        r_sh = sc.evaluate(s_sh)
        drop = 100 * (r_in["coverage"] - r_sh["coverage"])
        rows.append(dict(alpha=a, nominal=1 - a, q_hat=sc.q_hat_, cov_in=r_in["coverage"],
                         cov_shift=r_sh["coverage"], drop_pp=drop))
        print(f"{a:>6.2f} {1 - a:>8.2f} {r_in['coverage']:>14.4f} {r_sh['coverage']:>12.4f} {drop:>10.2f}")

    out = Path(__file__).resolve().parents[1] / "results" / "coverage_selfcheck.json"
    import json
    out.write_text(json.dumps(dict(cal_tag=args.cal_tag, test_tag=args.test_tag, shift_tag=args.shift_tag,
                                   n_cal=len(s_cal), n_in=len(s_in), n_shift=len(s_sh), rows=rows), indent=2))
    print(f"\nwrote {out}")
    print("\nThis is Paper 03's OWN measurement (src/conformal.py + raw prediction dumps),")
    print("not a citation to Paper 01's derived results file.")


if __name__ == "__main__":
    main()
