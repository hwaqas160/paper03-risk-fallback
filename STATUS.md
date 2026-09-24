# Paper 03 — Status

_Updated 2026-09-24 (late morning)_

## One-line hook (the paper's single claim)

**The first finite-sample guarantee on *when to trigger* a minimum-risk manoeuvre that is
stated on the closed-loop consequence of the decision — and that counts the harm the
fallback itself causes.**

Closest prior work is Conformal Decision Theory (Lekeufack et al., ICRA 2024): it also
calibrates decisions, but bounds a long-run *average* risk at rate O(1/t) for a continuously
acting planner parameter (pedestrian navigation). An MRM trigger is a one-shot, irreversible
decision with two error currencies (missed interventions *and* the harm of stopping) that
needs its certificate *before* deployment. CDT and ACI are both implemented as baselines.

## Data campaign (Task Scheduler `P03_Campaign`, resumable, self-healing)

**What happened:** `av2_cal` finished overnight (2 000 rows, 0 errors, 9.76 h at 8 workers).
`av2_test` reached 1 139/2 000 and then **stopped at 10:50** (a stray `^C` in the log; the task
uses "Interactive only" logon mode, most likely killed with the session — cause unconfirmed).
The wrapper kept reporting "Running", so nothing flagged it for 16 minutes. No data was lost
(one fsynced line per scenario). Then, before restarting, two pipeline bugs were found in the
T1 baseline (below), so the restart runs the corrected pipeline.

**Now:** the task re-launches `run\campaign.cmd` every 30 min (idempotent; ignores a new
instance while one runs), so an interruption heals itself, and it deletes itself only when
`src/campaign_status.py` says every arm is complete. Order: finish `av2_test` → rescore old rows
(reference run only) → `ns_val` → sensitivity arms. Progress: `python src/campaign_status.py`.

| Arm | Target | Rows | Needs per-agent tables |
|---|---|---|---|
| `av2_cal` | 2 000 | 2 000 ✔ | 2 000 (rescore pass) |
| `av2_test` | 2 000 | 1 139 | 1 139 (rescore pass); the remaining 861 record them inline |
| `ns_val` (H3 shift) | 1 500 | 0 | AutoBot-in-loop verified on nuScenes scenes |
| `av2_test_replay` / `_mrm2` / `_10hz` | 500 / 500 / 300 | 0 | sensitivity arms |

Estimated remaining compute: ~13 h collection + ~5 h rescoring at 8 workers. Runs at
below-normal priority; stop with `schtasks /delete /tn P03_Campaign /f`.

## What is built (all committed; 19 tests across 4 suites)

- **Exact per-tick evaluation** (`rollout.sweep_scenario_ticks`, `src/evaluate.py`): a rollout's
  outcome depends only on *when* the fallback fires. One reference run records every trigger's
  score; one rollout is forced to fire at each decision tick. Every trigger/threshold/online
  method/latency is then an exact lookup on identical physics. Measured at scale (33 761
  forced-fire rollouts): 98.4 % agree with the reference to <1 cm pre-fire; the pre-fire harm
  status differed in 1 rollout (0.003 %); results are also reported after dropping the 3.6 % of
  scenarios with any flagged drift.
- **Methods (ours):** LTT and CRC on the closed-loop miss indicator; density-ratio-weighted LTT
  with Kish effective sample size (refuses to certify when data can't support it).
- **Baselines (7):** T1 tuned confidence; T2 tuned MC-dropout ensemble; T3 open-loop conformal
  (radius from the predictor's own held-out error); tuned geometric (isolates the calibrator);
  CDT; ACI; oracle upper bound.
- **Ablations:** route-following vs kinematic corridor; width-only isotropic vs length-aware box;
  confidence/ensemble vs geometric score; LTT vs CRC vs empirical tuning; latency 0/1/2 ticks.
- **Validity study:** 200 random resplits, violation frequency P(test miss > α) vs δ.
- **H1 / H3 / harm-sensitivity** outputs in `evaluate.py`.

## Honest check against the eight publication goals

| Goal | Status | What's still needed |
|---|---|---|
| Clear novelty hook | ✔ stated above; positioned against CDT/ACI in Related Work | keep every section pointing at it |
| Strong baselines (3–5 recent SOTA) | **Partial.** Genuinely recent decision-level SOTA: CDT (2024), ACI (2021). Others are deployed practice or open-loop conformal. | Add Farid et al. (CoRL 2022) task-relevant failure detector (QAD) as a baseline — already read, implementable from the K predicted modes; be explicit about the approximation. A stronger predictor arm (Wayformer/MTR via UniTraj) would show results aren't AutoBot-specific. |
| Ablation study | ✔ implemented (score components, calibrator, latency) | run on campaign data |
| Multiple datasets/settings | **2 datasets, one shift direction.** Plus 3 traffic/MRM/rate sensitivity settings. | Free upgrade: also report the *reverse* shift (nuScenes-calibrated → AV2) by splitting `ns_val` rows — no new simulation. Waymo/nuPlan need licences only you can accept. |
| Framed limitations | ✔ "Scope and Future Work" section rewritten as scope choices | fill numbers once results exist |
| Reproducibility | ✔ pre-registration with dated amendment, per-scenario records, resumable campaign, configs | release repo + archive at submission; state hyper-parameters in an appendix table |
| Practical/theoretical value | ✔ certificate + priced trade-off + refuse-to-certify diagnostic | quantify in the abstract once results exist |
| One consistent story | ✔ title/abstract/intro/method/baselines all aimed at the hook | re-read for consistency after results land |

## Paper (`paper/main.tex`, IEEE T-IV template, compiles clean, 6 pages)

Title, abstract, four contributions, Related Work (incl. new CDT/ACI subsection), Method
(+ shift-weighted certification, + exact evaluation), Setup (data, baselines, metrics,
pre-registration), Results skeleton (one slot per table/figure the campaign produces), Scope
and Future Work, 15 references. **Corrected a wrong citation:** AutoBot is Girgis et al.
(ICLR 2022), not "Kim et al." as written in earlier notes.

## Findings so far (dev split, before pre-registered runs — reported, not headline)

- Decision rate: 2 Hz vs 10 Hz gave identical miss rates on 35 paired scenarios.
- A trigger scored against a route-following corridor missed near-misses that a route-agnostic
  TTC metric flags (divergence up to 30 m at 3 s); aligning the two collapsed the miss rate.
- **RETRACTED (2026-09-24):** an earlier version of this file said AutoBot's confidence score
  was constant across scenarios and that T1 may carry no risk information. That was a
  **pipeline bug, not a property of AutoBot**: the adapter gave agents the network did not
  predict (pedestrians, cyclists, vehicles beyond 60 m) constant-velocity forecasts with
  fabricated uniform mode probabilities, pinning `1 - max prob` at 1 - 1/6 = 0.833. Paper 01's
  independent inference path shows real variation (median max-prob 0.25, 462 distinct values,
  AUROC 0.63 for high error). Fixed (network-predicted agents flagged; T1 = confidence of the
  network-predicted agent whose modes come closest to the ego); T1 is re-measured, not assumed.
- Fallback braking must be UN R157-compliant (≤ 4.0 m/s²); an earlier ~10 m/s² stop produced
  rear-end collisions that were an artifact of that choice.

## Needs from you

1. **Nothing blocking the campaign.** It re-launches itself every 30 min while you are logged
   in ("Interactive only": logging off stops it; it resumes next time).
2. **Optional, high-value:** accept the Waymo Open Motion and nuPlan licences if you want a 3rd/4th
   dataset — only you can. Everything downstream (ScenarioNet conversion) is standard.
3. **Decision:** train a second/third predictor (Wayformer via UniTraj) as a robustness arm?
   Costs CPU-days on a shared machine; I'd do it after the campaign finishes.

## Next steps

1. Finish the campaign (`av2_test` → `ns_val` → sensitivity arms).
2. Run `src/evaluate.py` → every table; check the pre-registered refutation rules honestly.
3. Add the QAD baseline and the reverse-shift analysis (both need no new simulation beyond
   what's queued).
4. Fill Results/Discussion/Conclusion, generate figures, finalize abstract numbers.
