#!/bin/bash
# One command per paper result (run from the repository root with the Paper 03 Python environment; no simulation needed).
set -e
PY=${PY:-python}
$PY src/evaluate.py --cal results/campaign/av2_cal --test results/campaign/av2_test --out results/final/eval_main.json        # Table I base rows
$PY src/review_fixes.py p1 --out results/final/r11_p1_physics.json                                                              # TTC / headway baselines
$PY src/review_fixes.py p2 --out results/final/r11_p2_validity.json                                                             # corrected validity
$PY src/review_fixes.py p5 --out results/final/r11_p5_ttc.json                                                                  # TTC row, shift
$PY src/cswc.py indist --out results/final/r11_cswc_indist.json                                                                 # outcome certification (2 Hz arm)
$PY src/cswc.py nc --out results/final/r11_nc_audit.json                                                                         # outcome audit
$PY src/cswc2.py sup --out results/final/r12_sup.json                                                                            # supervised time-to-harm baseline
$PY src/cswc2.py sweep --out results/final/r12_sweep.json                                                                        # target sweep
$PY src/hz10.py --out results/final/r12_hz10.json                                                                                # 10 Hz decision arm (Table II)
$PY src/hz10.py --lats 8 10 --out results/final/r12_hz10_posthoc_long.json                                                       # 0.8 s and 1.0 s (post hoc)
$PY src/city_shift.py --out results/final/city_shift.json                                                                        # held-out cities
$PY src/spatial_groups.py --out results/final/spatial_groups.json                                                                # spatial-group resplits
$PY src/audit_numbers.py                                                                                                         # checks headline numbers against result files
