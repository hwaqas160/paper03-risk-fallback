# FINAL SUMMARY (auto-generated 2026-09-30 12:25)

Verdicts are computed mechanically from the pre-registered rules in notes/falsification.md. Anything marked pending needs data that has not finished collecting.

Scenarios: cal n=1994, test n=1996, nuScenes n=1504, Waymo n=1000

## Headline (AV2 test, alpha = 0.05)

| Method | Miss | Unnecessary stops | Induced |
|---|---|---|---|
| never | 15.1% | 0.0% | 0.0% |
| always | 0.0% | 84.9% | 6.5% |
| T1 tuned confidence (most conflicting agent) | 4.6% | 68.3% | 4.0% |
| T1 variant: nearest agent ahead | 4.2% | 75.5% | 4.6% |
| T2 tuned ensemble (MC dropout) | 4.0% | 70.8% | 4.9% |
| tuned geometric (ours, no guarantee) | 4.3% | 26.0% | 2.3% |
| T3 open-loop conformal | 0.0% | 73.8% | 5.5% |
| T4 LTT (ours) | 3.7% | 28.5% | 2.6% |
| T4 CRC (ours) | 4.3% | 26.0% | 2.3% |
| CDT (Lekeufack+ 2024) | 4.9% | 24.1% | 2.2% |
| ACI (Gibbs+Candes 2021) | 1.3% | 45.0% | 3.5% |
| oracle (hindsight, not deployable) | 5.0% | 23.6% | 2.2% |

Validity over resplits (violation frequency, valid if <= 10%): tuned geometric: 43.5%; T4 LTT (ours): 9.5%; T4 CRC (ours): 41.0%; T1 tuned confidence: 50.5%; T3 open-loop conformal: 0.0%

## Pre-registered verdicts

- **H1** (open-loop vs closed-loop ranking): NOT refuted (weak: tau < 0.8 or top differs in >= 1 setting)
- **H2** (certified beats best tuned baseline): REFUTED
- **H3** (AV2 certificate fails on nuScenes): SUPPORTED (CI entirely above alpha: guarantee fails) — miss 13.2% CI [11.6%, 15.0%]
- **H3-W** (fails on Waymo): REFUTED (CI includes or is below alpha: no detectable loss) — miss 2.7% CI [1.9%, 3.9%]
- **Generality rule (Amendment 3.3):** DATASET-DEPENDENT: fails on exactly one target (nuScenes); abstract must say so, not generalise

### H1 detail

| Setting | Kendall tau | Same top | Note |
|---|---|---|---|
| AV2 | 0.60 | True | not refuted here |
| nuScenes | 0.47 | False | not refuted here |
| Waymo | 0.73 | True | not refuted here |

### H2 detail (paired bootstrap: baseline stops - T4 stops)

| Baseline | Diff | 95% CI | Reading |
|---|---|---|---|
| T1 tuned confidence (most conflicting agent) | +0.398 | [+0.372, +0.424] | worse by >= 3pp |
| T1 variant: nearest agent ahead | +0.470 | [+0.445, +0.494] | worse by >= 3pp |
| T2 tuned ensemble (MC dropout) | +0.423 | [+0.398, +0.448] | worse by >= 3pp |
| tuned geometric (ours, no guarantee) | -0.025 | [-0.032, -0.018] | within/better (refutes H2) |
| T3 open-loop conformal | +0.453 | [+0.432, +0.476] | worse by >= 3pp |
| T4 CRC (ours) | -0.025 | [-0.032, -0.018] | within/better (refutes H2) |

## Waymo recalibration study (pre-registered R1-R4, primary = all scenarios)

- R1_M2_violates_ge_50pct_both_k: **HOLDS** ({"values": [0.695, 0.59]})
- R2_M4_k60_violation_le_delta: **HOLDS** ({"value": 0.0})
- R3_M4_k60_stops_ge_10pts_above_in_domain: **HOLDS** ({"value": 0.21293618087238303, "in_domain_stop": 0.2845691382765531})
- R4_M1_target_miss_CI_excludes_alpha: **REFUTED** ({"ci": [0.018621256649289826, 0.03899898941343562]})

(nuScenes recalibration replicate is exploratory, in results/final/recal_nuscenes_exploratory.json)

## Consequences of the intervention (post hoc)

| Data | Method | Marg. miss | Cond. FNR | Unnec. | Residual harm | New harm | Induced |
|---|---|---|---|---|---|---|---|
| av2_test | never fire | 15.1% | 100.0% | 0.0% | 15.1% | 0.0% | 0.0% |
| av2_test | LTT (marginal, ours) | 3.7% | 24.2% | 28.5% | 6.5% | 1.1% | 2.6% |
| av2_test | tuned geometric (no cert.) | 4.3% | 28.5% | 26.0% | 6.9% | 0.9% | 2.3% |
| av2_test | Luo-style conditional eps=0.05 | 0.0% | 0.0% | 84.9% | 4.1% | 3.1% | 6.5% |
| av2_test | Luo-style conditional eps=0.32 (matched) | 4.3% | 28.5% | 26.0% | 6.9% | 0.9% | 2.3% |
| av2_test | always fire | 0.0% | 0.0% | 84.9% | 4.1% | 3.1% | 6.5% |
| nuScenes | never fire | 22.1% | 100.0% | 0.0% | 22.1% | 0.0% | 0.0% |
| nuScenes | LTT (marginal, ours) | 13.2% | 58.1% | 26.1% | 15.6% | 1.7% | 2.4% |
| nuScenes | tuned geometric (no cert.) | 14.9% | 66.0% | 21.3% | 17.0% | 1.5% | 2.3% |
| nuScenes | Luo-style conditional eps=0.05 | 5.0% | 19.6% | 77.9% | 9.7% | 3.8% | 6.4% |
| nuScenes | Luo-style conditional eps=0.32 (matched) | 14.9% | 66.0% | 21.3% | 17.0% | 1.5% | 2.3% |
| nuScenes | always fire | 5.0% | 19.6% | 77.9% | 9.7% | 3.8% | 6.4% |
| waymo | never fire | 15.4% | 100.0% | 0.0% | 15.4% | 0.0% | 0.0% |
| waymo | LTT (marginal, ours) | 2.7% | 17.5% | 45.1% | 4.3% | 0.7% | 5.1% |
| waymo | tuned geometric (no cert.) | 3.1% | 20.1% | 42.2% | 4.6% | 0.7% | 5.0% |
| waymo | Luo-style conditional eps=0.05 | 0.1% | 0.6% | 84.6% | 3.7% | 2.6% | 7.9% |
| waymo | Luo-style conditional eps=0.32 (matched) | 3.1% | 20.1% | 42.2% | 4.6% | 0.7% | 5.0% |
| waymo | always fire | 0.1% | 0.6% | 84.6% | 3.7% | 2.6% | 7.9% |

## Sensitivity arms (AV2-certified threshold, paired by seed vs main run)

| Arm | n | Base harm | Miss [CI] | Unnec. | Induced | dMiss [CI] | dStop [CI] |
|---|---|---|---|---|---|---|---|
| replay | 378 | 17.7% | 3.7% [2.2%, 6.1%] | 35.4% | 18.5% | -0.005 [-0.013, +0.000] | +0.079 [+0.053, +0.108] |
| mrm2 | 378 | 16.7% | 4.2% [2.6%, 6.8%] | 27.5% | 1.6% | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| hz10 | 103 | 17.5% | 2.9% [1.0%, 8.2%] | 29.1% | 2.9% | -0.019 [-0.049, +0.000] | +0.029 [+0.000, +0.068] |

## Next step for a human / Claude session
Paste these numbers into paper/main.tex (Waymo subsection, sensitivity subsection, abstract/conclusion `[pending]` markers), recompile, and record outcomes in the Outcome log of notes/falsification.md.
