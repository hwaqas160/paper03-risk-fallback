# Paper 03 — Status

_Updated 2026-09-24 (morning)_

## One-line hook (the paper's single claim)

**The first finite-sample guarantee on *when to trigger* a minimum-risk manoeuvre that is
stated on the closed-loop consequence of the decision — and that counts the harm the
fallback itself causes.**

Closest prior work is Conformal Decision Theory (Lekeufack et al., ICRA 2024): it also
calibrates decisions, but bounds a long-run *average* risk at rate O(1/t) for a continuously
acting planner parameter (pedestrian navigation). An MRM trigger is a one-shot, irreversible
decision with two error currencies (missed interventions *and* the harm of stopping) that
needs its certificate *before* deployment. CDT and ACI are both implemented as baselines.

## Live: data campaign (Task Scheduler `P03_Campaign`, resumable)

| Arm | Target | Done | Notes |
|---|---|---|---|
| `av2_cal` | 2 000 | **2 000 ✔** | 0 errors, 9.76 h at 8 workers, unattended overnight |
| `av2_test` | 2 000 | ~1 140 (running) | auto-throttled 8→6 workers when free commit dropped to 24 GB |
| `ns_val` (H3 shift) | 1 500 | queued | |
| `av2_test_replay` (log-replay traffic) | 500 | queued | sensitivity |
| `av2_test_mrm2` (2.0 m/s² MRM) | 500 | queued | sensitivity |
| `av2_test_10hz` | 300 | queued | decision-rate robustness |

Crash-safe: one fsynced JSON line per scenario; re-running `run\campaign.cmd` resumes.
Runs below-normal priority so Paper 01 / AI2 jobs on this machine keep priority.
Stop: `schtasks /end /tn P03_Campaign`. Log: `results\campaign\campaign.log`.

## What is built (all committed; 19 tests across 4 suites)

- **Exact per-tick evaluation** (`rollout.sweep_scenario_ticks`, `src/evaluate.py`): a rollout's
  outcome depends only on *when* the fallback fires. One reference run records every trigger's
  score; one rollout is forced to fire at each decision tick. Every trigger/threshold/online
  method/latency is then an exact lookup on identical physics. Verified: 0 divergences.
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
- AutoBot's confidence score was a constant across 15 probe scenarios (T1 may carry little
  risk information) — to be confirmed at scale; the ablation table will show it either way.
- Fallback braking must be UN R157-compliant (≤ 4.0 m/s²); an earlier ~10 m/s² stop produced
  rear-end collisions that were an artifact of that choice.

## Needs from you

1. **Nothing blocking the campaign.** It is running unattended.
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
