# Falsification Note

> Fill this in BEFORE running any experiment. Commit it. Do not edit after results arrive.
> Copy the hypotheses from PLAN.md section 3 and make them numeric.

**Date written:** 2026-09-22
**Git history:** first committed at `e53c5c4` (repo init, before this file existed in its
current form). This addendum reframing H3's premise (dependency note, self-computed coverage
numbers) was added in a later commit — see git log for the exact SHA and diff. Nothing in the
H1/H2/H3 claims, metrics, or refutation thresholds themselves changed after `e53c5c4`; only
how the H3 premise is sourced (self-computed vs. cited) was clarified.
**Written by:** Claude, on the author's explicit instruction ("take the best harm definition
used in top journal papers and keep working"). The author may amend it freely before the
first `av2_test` or nuScenes evaluation is run.

## What had been seen when this was written (full disclosure)

- Harm base rates on 1 000 **dev** scenarios (AV2 val/train, no fallback, reactive traffic):
  nuPlan headline 12.9 % [11.0, 15.1]; at-fault collisions 3.1 %.
- Two pilots with a **stub constant-velocity predictor** (no AutoBot, no trigger comparison):
  plumbing checks only. The earlier of them touched `av2_test` (300 scenarios); no
  hypothesis below was evaluated on it.
- Open-loop coverage, AV2 → nuScenes: 95 → 92.2 %, 90 → 86.7 %, 80 → 77.2 % (drop 2.7–3.7 pp).
  **This paper computes that number itself** (`src/coverage_selfcheck.py`, `src/conformal.py`
  — split conformal prediction, Vovk/Gammerman/Shafer 2005) from raw model prediction dumps.
  It is not a citation to any external paper's derived result.
- **Not seen:** any closed-loop result with a real predictor; any trigger comparison; any
  nuScenes closed-loop run.

## A note on dependencies (why this claim doesn't need Paper 01 to exist)

This work uses a trained AutoBot checkpoint and converted scenario data as **infrastructure**
— the same relationship any paper has to a pretrained backbone or a public dataset. It makes
no claim that requires a companion paper to be published, accepted, or even cited:
- AutoBot is a published, citable architecture (Kim et al., ICLR 2022); the checkpoint used
  here is described directly in this paper's own methods (training data, minADE), not by
  reference to another paper's evaluation of it.
- The AV2 → nuScenes coverage-collapse premise behind H3 is **computed by this paper's own
  code** (above), not asserted on another paper's authority.
- Split conformal prediction and Learn-then-Test are cited to their original sources
  (Vovk et al. 2005; Angelopoulos, Bates, Candès, Jordan, Lei 2021), not to any unpublished
  work.

If a "Paper 01" is ever cited in the manuscript, it should be for the checkpoint/data
artifact only ("we use a publicly-releasable checkpoint trained on..."), never as the
authority for a scientific claim this paper needs to stand on its own.

## Fixed definitions

- **Harm (headline)** = nuPlan closed-loop benchmark (Karnchanachari et al., ICRA 2024;
  nuplan-devkit metrics): an ego **at-fault collision** (stopped-track, active-front, or
  active-lateral with ego out of lane) **or TTC < 0.95 s** (3 s horizon, 0.1 s step; not
  evaluated while ego is stopped; tracks behind ego ignored). Implementation:
  `src/nuplan_metrics.py`, deviations from nuPlan listed in its docstring.
- **Miss** = harm occurs and the fallback did not fire strictly before its first step.
- **Unnecessary stop** = fallback fired and the no-fallback counterfactual of the same
  scenario is harm-free.
- **Induced collision** = the fallback run has a not-at-fault collision (rear-ended /
  stopped) that the counterfactual does not. Reported as a cost; never counted as a miss.
- **Risk level** α = 0.05, δ = 0.10 (LTT with Hoeffding–Bentkus p-values, fixed-sequence
  testing). α is below the 12.9 % no-trigger base rate, so "never trigger" does not satisfy it.
- **Traffic**: IDM-reactive. **Decision rate**: every step (10 Hz). **Latency**: 0 steps for
  headline numbers (Exp 7 sweeps it).
- **Splits**: design on `av2_dev`; λ calibrated on `av2_cal`; evaluated on `av2_test`
  (AV2 in-domain) and nuScenes full logs (shift). λ grid from quantiles of a probe block
  that is excluded from calibration.
- **n**: 2 000 calibration scenarios, 2 000 AV2 test scenarios, all available nuScenes
  full-log scenarios with a moving ego. 95 % CIs are Wilson intervals; paired differences
  use a scenario-level bootstrap (10 000 resamples).

## H1 — open-loop metrics mislead
- **Claim:** ranking the triggers by open-loop quality differs from ranking them by
  closed-loop safety outcome.
- **Metric:** Kendall's τ between (a) the ranking by open-loop conformal efficiency at matched
  coverage and (b) the ranking by closed-loop unnecessary-stop rate at matched miss rate ≤ α.
- **Datasets / arms:** T1 fixed confidence, T2 ensemble variance, T3 conformal region size,
  T4 risk-calibrated; AV2→AV2 and AV2→nuScenes.
- **Refuted if:** τ ≥ 0.8 **and** the top-ranked trigger is the same under both metrics, in
  both dataset settings.

## H2 — calibrated triggering wins
- **Claim:** T4 gives a better safety / over-conservatism trade-off than the best
  hand-tuned fixed threshold.
- **Metric:** unnecessary-stop rate on `av2_test` at realised miss rate ≤ α; each baseline's
  threshold tuned on `av2_cal` to its own best operating point with miss ≤ α.
- **Refuted if:** the best of T1–T3 has an unnecessary-stop rate within **3 pp** of T4
  (paired bootstrap 95 % CI of the difference includes −3 pp), **or** T4's realised miss rate
  on `av2_test` exceeds α with its 95 % CI entirely above α.

## H3 — the arc closes
- **Claim:** under the AV2 → nuScenes shift, the threshold certified on AV2 loses its
  guarantee in closed loop.
- **Metric:** realised miss rate on nuScenes full-log scenarios at the λ̂ certified on
  `av2_cal`.
- **Refuted if:** the 95 % CI of that miss rate includes α = 0.05 (no detectable loss of the
  guarantee). The open-loop ~4 pp coverage drop was already known when this was written, so
  H3 is stated on the closed-loop outcome only.

---

## Outcome log (fill in AFTER experiments)

| Hypothesis | Result | Supported? | Notes |
|---|---|---|---|
| H1 | | | |
| H2 | | | |
| H3 | | | |

---

**Rule:** if a hypothesis is refuted, report it. A clearly reported negative result is
publishable and is worth more than a positive result nobody can reproduce.

---

## Amendment 1 — 2026-09-23, BEFORE any calibration- or test-split data was collected with the final pipeline

Every change below was made on the development split or on engineering grounds. The
refutation thresholds for H1–H3 are **unchanged**; only how they are operationalized is
made concrete.

1. **Decision rate: 2 Hz (every 5 steps), not 10 Hz.** A paired dev-split test (35 identical
   scenarios, identical λ grid) gave bit-for-bit identical miss rates at 2 Hz and 10 Hz
   (notes/design.md §14). A 10 Hz replication on a 300-scenario subset of `av2_test` is
   reported as a robustness check.
2. **Predictor: `av2_cpu_v2/epoch03` (minADE₆ 1.092)** instead of `av2_cpu_v1/epoch08`
   (1.349). Same architecture; verified drop-in.
3. **Evaluation is exact from per-tick rollouts.** Each scenario stores one reference run
   with every trigger's score trace, plus one rollout forced to fire at each decision tick.
   Every method is a lookup on identical physics (`src/evaluate.py`,
   `test_fire_step_reproduces_lambda_trigger`).
4. **Baseline set for H2**, each evaluated on the same test scenarios:
   T1 tuned confidence threshold; T2 tuned MC-dropout ensemble spread (5 passes; a deep
   ensemble is an additional arm when trained); T3 open-loop conformal — inflate predicted
   modes by the split-conformal radius of the predictor's own 3 s open-loop error on held-out
   AV2 logs, fire on contact with the ego corridor (same α); tuned geometric score (isolates
   the calibrator); CDT (Lekeufack et al. 2024) and ACI (Gibbs & Candès 2021), online, best
   step size per method; oracle in hindsight (upper bound, not deployable). **H2 is refuted
   if any deployable baseline is within 3 pp of T4's unnecessary-stop rate at realized miss
   ≤ α**, as pre-registered.
5. **H1 operationalized:** open-loop quality = AUROC of each score's scenario peak for
   predicting harm in the no-fallback run (the metric open-loop failure-detection work
   reports); closed-loop quality = unnecessary-stop rate at that score's LTT-certified λ. Scores
   ranked: conf, ens, gap, geom, geom_route, geom_iso. Refutation rule unchanged (τ ≥ 0.8 and
   same top-ranked score).
6. **Guarantee validity:** 200 random half/half resplits of the pooled AV2 cal+test scenarios;
   LTT's violation frequency P(test miss > α) must be ≤ δ.
7. **H3 target data:** nuScenes prediction-challenge val scenarios (8.1 s each, 9041 total,
   merged at `data/ns_val_merged`), not full ~20 s logs. The earlier note that these were
   "~2.5 s, too short" was wrong. H3's refutation rule is unchanged. **Added arm:** density-ratio-
   weighted LTT as the shift repair, weights from a domain classifier on pre-deployment scenario
   covariates only (`rollout.FEATURE_NAMES`).
8. **n:** 2 000 `av2_cal`, 2 000 `av2_test`, 1 500 `ns_val` (non-static scenarios).
9. **Sensitivity arms (reported, not used for H1–H3):** log-replay traffic and a 2.0 m/s²
   comfort MRM, each on 500 `av2_test` scenarios.
