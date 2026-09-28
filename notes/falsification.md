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
- AutoBot is a published, citable architecture (Girgis et al., ICLR 2022 -- citation corrected 2026-09-23 from an earlier wrong "Kim et al."); the checkpoint used
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
| H1 | AV2: τ = 0.60, same top (geom_iso). nuScenes: τ = 0.47, different top (geom vs geom_iso, 0.002 apart). Drop-diverged: τ = 0.73 / 0.60, same top in both. | Not refuted (τ < 0.8 everywhere) — **weakly** | Disagreement is among mid/low-ranked scores; open-loop AUROC picked the closed-loop best (or a statistical tie) in every setting. Must be reported as "rankings diverge below the top", not "open-loop picks the wrong trigger". |
| H2 | Unnec. stop, T4 LTT 0.285 (miss 0.037, CI [0.029, 0.046]). Tuned geometric 0.260, diff −0.025 [−0.032, −0.018]; CRC same; CDT 0.241. T1/T2/T3 +0.40 to +0.45 worse. | **Refuted** | Same-score tuned threshold, CRC and CDT beat LTT on stops. Heuristic triggers (confidence, ensemble, open-loop conformal) are 40–45 pp worse. Validity: LTT violation freq 0.095 ≤ δ; tuned 0.435, CRC 0.41, T1 0.505. |
| H3 | Miss on nuScenes at AV2-certified λ: 0.132, Wilson 95 % CI [0.116, 0.150] (drop-diverged 0.152). Weighted LTT repair: miss 0.070 at unnec. stop 0.524. | **Supported** | Guarantee breaks under shift (2.6× α); the weighting repair halves the excess but does not restore α and doubles stops. |

Evaluated 2026-09-28 on 1 994 cal / 1 996 test / 1 504 nuScenes scenarios (10 AV2 scenarios dropped
for unmatched per-agent tables, per Amendment 2). Outputs: `results/eval.txt`, `results/eval_dropdiv.txt`.
Before evaluation, `evaluate.py` gained the pre-registered statistics it lacked (H2 paired bootstrap,
H1 on the nuScenes setting, H3 Wilson CI printout); no threshold or rule changed.

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


## Amendment 2 — 2026-09-24, BEFORE any evaluation of calibration- or test-split outcomes

Found by inspecting the *score distribution* of the finished calibration campaign
(descriptive; no outcomes, no test data, no hypothesis evaluated). H1–H3 claims, metrics and
refutation thresholds are **unchanged**.

1. **T1 (confidence threshold) was mis-implemented, and its earlier "finding" is retracted.**
   The stored confidence score was pinned at 1 − 1/6 in every scenario because the adapter gave
   agents the network did not predict (pedestrians, cyclists, vehicles beyond 60 m) fabricated
   uniform mode probabilities (44 % of agent-tick rows in a real-scenario check), and because the
   code took the least-confident of any agent, not the design's "most conflicting agent".
   **T1 is now: 1 − max mode probability of the NETWORK-PREDICTED agent whose predicted modes come
   closest to the ego** (`conf_conflict`), with the nearest-agent-ahead variant (`conf_ahead`)
   reported alongside. The corrected definition is the strongest reasonable version of a
   deployed confidence trigger, not a weakened one. See `notes/design.md` §16.
2. **Exactness at scale (measured, replaces an over-claim).** 538 of 33 761 forced-fire rollouts
   (1.59 %, 71 of 2 000 scenarios) drifted ≥ 1 cm before firing; pre-fire TTC traces differed in
   1.21 %; pre-fire **harm status differed in 1 (0.003 %)**. Headline analyses use all scenarios;
   `--drop_diverged` is reported as a robustness check. Decided here, before any outcome analysis.
3. **Data handling.** Already-collected rows (2 000 `av2_cal`, 1 139 `av2_test`) keep their
   forced-fire outcomes (which depend only on the firing tick) and are re-scored by re-running
   only the reference run; a sidecar is used only when its tick count matches the stored row.
   Rows that cannot be matched are dropped from analyses needing per-agent tables and the count
   is reported.


## Amendment 3 — 2026-09-28, BEFORE any Waymo closed-loop outcome exists (3 plumbing rows only)

Added after the AV2/nuScenes evaluation (Outcome log above), on the author's Waymo Open licence
acceptance. It adds a third dataset and one reverse-shift analysis. **H1–H3, α, δ and every
refutation threshold above are unchanged and are not re-tested on new data to rescue a result.**
Seen before writing this: the AV2/nuScenes outcomes (H2 refuted, H3 supported, H1 weak); three Waymo
smoke rows inspected for plumbing only (tick counts, tables present, 0 drift), no harm/miss read.

1. **Data.** Waymo Open Motion Dataset v1.2.1 `uncompressed/scenario/validation`, first 8 finished
   shards in sorted order (~2 350 scenarios), converted by `src/convert_waymo.py` (ScenarioNet's
   converter, pure-Python TFRecord reader). Non-static-ego filter as for nuScenes; the first
   **1 000** qualifying scenarios in database order are simulated (n chosen for compute: 1 000 gives a
   Wilson half-width ≈ 2 pp on a 13 % miss rate). Same predictor (AV2-trained AutoBot), same
   pipeline, same 2 Hz decisions. Waymo scenarios carry 1 s of history vs 5 s in AV2; this is a
   property of the dataset and is reported, not corrected.
2. **H3-W (pre-registered, one-sided in the same sense as H3).** The threshold certified on
   `av2_cal` (LTT, α = 0.05, δ = 0.10) has a Waymo miss rate whose 95 % Wilson CI **excludes α**.
   Refuted if the CI includes α. Expectation written down in advance: miss > α (short history
   degrades the predictor, a larger shift than nuScenes).
3. **Generality claim.** "The AV2-certified guarantee fails under real cross-dataset shift" is
   claimed **only if H3 and H3-W both hold**; if exactly one holds it is reported as
   dataset-dependent, and the abstract is written to that effect.
4. **Repair.** Weighted LTT (domain-classifier density ratio, Kish n_eff) is evaluated on Waymo
   exactly as on nuScenes. It counts as "restoring" the guarantee on a target iff that target's
   miss CI includes α or lies below it. Reported either way.
5. **H1 on Waymo** is a third setting reported alongside AV2 and nuScenes (same τ / same-top rule;
   the overall H1 claim is "refuted only if refuted in ALL settings", which is the stricter reading
   of the original wording and the one that favours refutation).
6. **Reverse shift (nuScenes → AV2).** Calibrate LTT (α = 0.05, δ = 0.10, `geom`) on a random 50 %
   of `ns_val` rows (seed 0, scenario-level), test on all of `av2_test`; and the same with the other
   50 % as a replicate. Claim: violation (test-miss CI excludes α). Reported whichever way it goes;
   n_cal ≈ 750, so LTT may refuse to certify (λ̂ = -inf), which is itself reported.
7. **Not used to select anything:** no threshold, score, hyper-parameter or grid is chosen from
   Waymo rows. Scenarios failing simulation are counted and reported, not silently dropped.
8. **Compute priority.** `waymo_val` runs before the three sensitivity arms. The 10 Hz arm is cut
   from 300 to **100** scenarios (a cost decision made before any 10 Hz data exists; the arm only
   supports the "decision rate is not load-bearing" statement, for which 100 paired scenarios with
   identical seeds suffice, since Amendment 1 already found bit-identical misses at 2 Hz vs 10 Hz on
   the dev split).


## Amendment 3b — 2026-09-28, POST-HOC sensitivity (labelled as such; changes no verdict rule)

Found while running the pre-registered reverse shift (3.6), which returned **"LTT refused to certify"
on both nuScenes halves** (n_cal = 752 each). Cause, measured: 75 of 1 504 nuScenes scenarios (**5.0 %**)
already contain harm at the first decision tick, so even "always fire at t=0" misses them — an
irreducible floor equal to α, so no threshold can be certified at α = 0.05 there. On AV2 the floor is
0.1 %. (nuScenes snippets start mid-interaction; a TTC < 0.95 s state at the first tick is a property
of the log, not something a fallback can undo.)

Post-hoc, NOT replacing any pre-registered result (which stand as reported): re-running with those
"unavoidable-at-start" scenarios excluded.

| Analysis (α = 0.05, δ = 0.10) | Pre-registered / all scenarios | Post-hoc, avoidable only |
|---|---|---|
| H3: AV2-certified λ on nuScenes, miss [95 % CI] | 0.132 [0.116, 0.150] | 0.092 [0.078, 0.108] — still excludes α |
| Weighted-LTT repair on nuScenes | 0.070 (n_eff ≈ 68 in the avoidable set) | 0.022 [0.016, 0.031], unnec. stop 0.545 |
| Reverse shift, nuScenes-certified → AV2 test | refused to certify (both halves) | miss 0.019 / 0.020 (CIs below α), unnec. stop 0.41 / 0.40 |

Reading: H3 is robust to the exclusion. The shift is **asymmetric**: a threshold certified on the
harder domain (nuScenes) transfers safely but conservatively to the easier one (AV2 stops 40 % vs
28.5 % in-domain), while the AV2-certified threshold is unsafe on nuScenes. The weighted repair's
effective sample size is very small (68), so its "restoration" is bought almost entirely with
conservatism and must be reported with n_eff. Any "avoidable-harm" refinement of the miss metric is
a *proposal* for the paper's discussion and is reported next to, never instead of, the
pre-registered miss definition.
