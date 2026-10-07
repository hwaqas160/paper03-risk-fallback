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


## Amendment 3c — 2026-09-28, POST-HOC / EXPLORATORY: few-shot recalibration on the shifted domain

Not pre-registered; run on the existing nuScenes rows (avoidable-at-start subset, n = 1 429) after the
H1-H3 verdicts, at the author's question. k labelled nuScenes scenes drawn at random (60 draws each),
evaluated on the remaining nuScenes scenes; α = 0.05, δ = 0.10. Baseline (AV2-certified, no target
labels): miss 0.092, stops 0.271.

| k | Method | miss (mean) | P(miss > α) | unnec. stops |
|---|---|---|---|---|
| 40 | target-only tuned (no certificate) | 0.066 | 0.67 | 0.356 |
| 40 | target-only LTT | 0.000 | 0.00 | 0.813 |
| 40 | AV2 + k, reweighted LTT | 0.018 | 0.10 | 0.636 |
| 60 | target-only tuned | 0.060 | 0.70 | 0.367 |
| 60 | target-only LTT | 0.008 | 0.00 | 0.725 |
| 60 | AV2 + k, reweighted LTT | 0.028 | 0.00 | 0.533 |
| 100 | AV2 + k, reweighted LTT | 0.033 | 0.12 | 0.481 |

Reading: a few dozen labelled target scenes restore a *valid* guarantee (violation ≤ δ) but only at
a large conservatism cost (53-64 % stops vs 27 % before); uncertified tuning with k = 40-60 fails the
5 % target 2 times in 3. It does not "return to the original operating point". Exploratory; would need
pre-registration and a second target dataset (Waymo) before being claimed.


## Amendment 4 — 2026-09-28, BEFORE any Waymo closed-loop outcome has been read

Two parts: (A) post-hoc findings on AV2/nuScenes from an audit of our own miss definition and a
faithful implementation of Luo et al. (IJRR 2024) inside our framework (`src/costs_and_luo.py`);
(B) a CONFIRMATORY pre-registration of the few-label recalibration study on Waymo. Waymo rows
collected so far were counted (progress only), never read for miss/harm/stop.

### A. Post-hoc findings (AV2 test n = 1 996; nuScenes n = 1 504; none alters an H1–H3 verdict)

1. **Audit.** `missed()` is `harm in the run AND NOT (fallback fired strictly before it)`. Because a
   forced-fire run equals the reference run before firing, the miss indicator is exactly
   `1[reference harm step <= firing step]`; likewise `stop = fired AND reference harmless`. **Miss and
   unnecessary stop are functions of the no-fallback reference run and the firing tick only.** The
   forced-fire rollouts are needed for the fallback's own consequences: induced collisions, route
   completion, and harm that persists or arises after firing. The manuscript wording "measured after
   the fallback has changed the future" for the miss was therefore inaccurate and is corrected.
   Our miss is Luo et al.'s alert-before-unsafe indicator taken over ALL scenarios (marginal) instead
   of over unsafe ones (class-conditional).
2. **Residual harm is not the certified quantity.** At the AV2-certified LTT threshold (test): miss 3.7 %,
   alert-too-late 1.7 %, harm prevented 9.8 %, **new harm in scenarios harmless without the fallback
   1.1 %**, so harm in the intervened run is 6.5 % (no fallback: 15.1 %). Always-firing creates
   new harm in 3.1 % of scenarios. On nuScenes at the same threshold: miss 13.2 %, residual 15.6 %.
3. **Marginal 5 % is a weak-sounding target.** Base harm rate 15.8 % (cal). LTT's class-conditional FNR is
   24.2 % (AV2 test) and 58.1 % (nuScenes): about one harmful scenario in four is not alerted in time.
4. **Luo-style class-conditional certificate** (Mondrian split conformal on the unsafe class, finite-sample
   correction, no tie-randomisation; unsafe = harm in the reference run; alert = firing before it):
   at ε = 0.05 it selects a threshold equivalent to always firing (miss 0.0 %, unnecessary stops 84.9 %)
   on this predictor, i.e. it is vacuous; at the matched ε = α / base rate = 0.32 it selects the same
   threshold as tuned geometric (miss 4.3 %, stops 26.0 %) and, deployed on nuScenes, has conditional
   FNR 66.0 % against a 32 % target, so the class-conditional guarantee does not survive the shift either
   (harm base rate 15.8 % -> 22.1 % does not explain it; P(score | unsafe) moved).
5. **Manuscript consequences (made in the same commit):** the loss is monotone in λ by construction, so
   CRC applies rigorously and LTT is used for its high-probability form; the exact-evaluation claim is
   restated as "reference-only outcomes plus one forced rollout per tick for post-fire consequences".

### B. Pre-registered (confirmatory) Waymo recalibration study

Target data: `waymo_val` (Amendment 3). For k ∈ {40, 60}, draw k labelled Waymo scenarios uniformly at random
(200 draws per k, seed 0), evaluate on the remaining Waymo scenarios (α = 0.05, δ = 0.10, score `geom`),
using ALL non-static scenarios, and also the avoidable-only subset (harm not present at the first decision
tick) as a labelled sensitivity. Methods: (M1) AV2-certified threshold, no target labels; (M2) target-only
tuned; (M3) target-only LTT; (M4) AV2 + k weighted LTT (source weight 1, each target scenario weight
N_src / k, Kish n_eff reported); (M5) Luo-style conditional certificate at matched ε on the k target scenes.
Violation = test miss rate > α in a draw.

- **R1.** M2 (uncertified tuning) violates in >= 50 % of draws for both k. (Nuance: a prediction from the
  nuScenes exploratory result; refuted if < 50 %.)
- **R2.** M4 at k = 60 violates in <= δ = 10 % of draws (validity restored). Refuted if > 10 %.
- **R3.** M4 at k = 60 has unnecessary-stop rate >= 10 points above the AV2 in-domain LTT operating point
  (the cost of recalibrating). Refuted if < 10 points.
- **R4.** M1 (AV2-certified, no labels) has Waymo miss rate whose 95 % CI excludes α (this is H3-W of
  Amendment 3, restated; not re-tested separately).
- The nuScenes exploratory table (Amendment 3c) is NOT pooled with Waymo. Results are reported per dataset.
  Any result not listed here is exploratory and labelled so.

### C. Not yet pre-registered
A second trajectory predictor (different architecture or independently trained) will get its own
amendment, written before its data exist; its arm size will be stated there.


## Amendment 5 — 2026-09-29, BEFORE any new hypothesis is evaluated on new or existing data

A new certification target is added, motivated by a finding already reported above (not by any new
data): at the Amendment-2 alarm-level LTT certificate, **harm in the executed run is 6.5%, not the
certified 3.7%**, because the certificate bounds whether the alert preceded harm, not whether harm
occurred in the run the vehicle actually drove. This amendment fixes N1-N5 and the fresh, disjoint
data each is evaluated on, before touching any of it. H1-H3 and their verdicts above are final and
are not reopened.

### Literature check (2026-09-29, before writing N1-N5)

A targeted search found no paper certifying a one-shot, irreversible fallback trigger's realized
closed-loop outcome on real driving logs with reactive traffic, pricing the intervention cost. The
closest prior art, and why it does not cover this:

- **Joshi, Wang, Hassani & Dobriban, "Risk-Controlled Post-Processing of Decision Policies" (2026,
  arXiv:2605.06479)** certify the outcome of switching from a baseline to a fallback *policy* — the
  same principle as N1 below. Their setting is per-instance, i.i.d., single-shot (radiograph
  diagnosis, LLM routing, synthetic classification): no trajectory, no timing decision, no absorbing
  switch, no over-conservatism metric. Our setting is sequential and absorbing — *when* the switch
  happens determines every outcome after it — which is a different mathematical structure, evaluated
  here for the first time on a real, closed-loop, safety-critical system. Cite and differentiate
  explicitly; do not claim outcome-level certification itself as new.
- **Chang & Ahmed (2026, arXiv:2608.26533)**, CVaR-certified driving-trajectory selection: continuous
  re-planning among candidate trajectories each step, not a one-shot task-abandonment decision; no
  evidence of closed-loop reactive-traffic evaluation or an intervention-cost metric found.
- **Gonzales et al. (IROS 2025, arXiv:2603.10392)**, CRC+CBF for human-robot interaction: certifies
  error in a *predicted* safety value, not the realized outcome; continuous control, not a one-shot
  MRM; no over-conservatism reported.
- **MultiRisk (Joshi, Sun, Hassani & Dobriban, 2025, arXiv:2512.24587)**: the natural tool for jointly
  certifying multiple risks (N-component below), but its dynamic-programming guarantee is stated for
  monotone risk; our own data (this amendment, item 2) shows residual harm is non-monotone in the
  threshold in ~4% of scenarios. Used only where applicable; Pareto Testing (Laufer-Goldshtein et al.
  2023, already the basis of our multi-baseline machinery) is the fallback tool where it is not.
- **Angelopoulos, "Conformal Risk Control for Non-Monotone Losses" (2026)**: confirms non-monotone
  risk is a recognised open problem, and is the citation for why LTT (fixed-sequence testing, no
  monotonicity assumption) is used for N1 rather than plain CRC.
- **Conformal Predictive Safety Filter (2023, arXiv:2306.02551)**: a per-step MPC braking filter with
  online-calibrated bounds — a continuous safety filter, not an irreversible task-abandonment
  decision studied end-to-end on logged scenarios.

### Fixed definitions for this amendment

- **Residual harm** $H(X,\lambda)$ = 1 if the nuPlan harm definition (Section IV-B / notes above) is
  triggered *anywhere* in the executed run under threshold $\lambda$ (before OR after firing),
  0 otherwise. This is `Scenario.harm_run` at the firing tick, already implemented and used
  descriptively in Amendment 4; it has not been certified until now.
- Measured on `av2_cal` (n=1994): residual harm is **non-monotone in $\lambda$ in 4.4% of scenarios**
  (firing earlier sometimes creates harm that firing later avoids, or vice versa). CRC and MultiRisk's
  DP both assume monotonicity; LTT (fixed-sequence testing) does not and is the primary calibrator for
  every new hypothesis below. CRC/MultiRisk are reported as a comparison where they can be applied,
  with the monotonicity violation disclosed, never silently assumed to hold.
- $\alpha_H = 0.05$ for residual harm (same level as the existing miss certificate, for comparability),
  $\delta = 0.10$, unchanged from H1-H3.

### N1 — outcome-level certification is achievable and behaves differently from the alarm-level one
**Claim:** an LTT certificate on residual harm $H(X,\lambda)$ at $\alpha_H=0.05$ is valid (violation
frequency $\le\delta$ over 200 resplits of fresh calibration+test data, defined below), and its
operating point differs materially (in $\lambda$, in unnecessary-stop rate, or both) from the
Amendment-2 alarm-level certificate re-fit on the SAME fresh calibration data.
**Refuted if:** the outcome-level certificate is invalid (violation $>\delta$), OR its operating point
is not distinguishable from the alarm-level one (stop-rate difference's 95% CI includes 0).

### N2 — the existing alarm-level certificate silently violates the outcome-level target
**Claim:** the Amendment-2 alarm-level LTT threshold (re-fit on fresh calibration data, same
procedure), evaluated for residual harm on fresh test data, exceeds $\alpha_H=0.05$, with a 95% CI
that excludes it.
**Refuted if:** that CI includes or is below $\alpha_H$.
This is the paper's central new empirical claim; on the existing (already-analysed) data it reads
6.5% vs a 5% target, which is why N2 is expected to hold, but it is tested on data not used to reach
that expectation.

### N3 — a non-monotone-aware calibrator is necessary, not merely available
**Claim:** on fresh calibration data, CRC (assumes monotonicity) certifies a $\lambda$ for residual
harm whose true (fresh test) violation frequency exceeds $\delta=0.10$, while LTT's does not.
**Refuted if:** CRC's violation frequency is also $\le\delta$ (i.e. the non-monotonicity does not
matter in practice at this alpha).

### N4 — joint multi-risk certification (residual harm + induced collisions, minimise stops)
Using Pareto Testing (primary; MultiRisk reported as a check that is only valid where risks are
confirmed monotone) to jointly certify $H(X,\lambda,d)\le\alpha_H$ and induced-collision rate
$I(X,\lambda,d)\le\alpha_I=0.03$ over the joint grid of threshold $\lambda$ and MRM deceleration
$d\in\{2.0, 4.0\}\,\text{m/s}^2$ (the two levels we already have data pipelines for).
**Claim:** the jointly-certified policy achieves a lower unnecessary-stop rate than certifying
$\lambda$ alone at $d=4.0$ (the current design) while keeping both risks controlled.
**Refuted if:** the joint search does not find a valid policy with stops below the single-risk
baseline's, or joint certification fails to control both risks simultaneously (violation $>\delta$
on either).
Honest expectation on file: our own 2 m/s^2 sensitivity arm already showed no unnecessary-stop
difference from 4 m/s^2 at the alarm level (paired diff +0.000 [+0.000,+0.000]); N4 may well be
refuted, and is reported either way.

### N5 — findings replicate with a second, independently-trained predictor
**Claim:** N1 and N2's verdicts (direction and rough magnitude, not exact numbers) replicate using
`av2_gpu_full/epoch53-minADE0.854.ckpt` (Paper 01 artifact; minADE6 0.854 vs the primary predictor's
1.092 — used here only as a differently-trained model, never as authority for any claim this paper
needs to stand on its own, consistent with the position in the "note on dependencies" above).
**Refuted if:** N2's headline direction (alarm-level miss < outcome-level residual harm at the same
certified operating point) does not hold with this predictor.
Evaluated on the SAME fresh test split as the primary predictor (paired by seed), not a new split.

### Fresh, disjoint data (verified 2026-09-29 against the actual campaign output, not assumed)

Every hypothesis above is evaluated ONLY on scenarios never touched by any row (used or skipped) in
`results/campaign/{av2_cal,av2_test,ns_val,waymo_val}`, so nothing here can be an artifact of a
threshold chosen to fit already-seen scenarios.

| Database | Total pool | Max index already touched | Fresh range (inclusive) | Fresh scenarios |
|---|---|---|---|---|
| `av2_cal` | 4971 | 3594 | 3595-4970 | 1376 |
| `av2_test` | 5027 | 3583 | 3584-5026 | 1443 |
| `ns_val` | 9041 | 2623 | 2624-9040 | 6417 |
| `waymo_val` (8 shards) | 2321 | 1807 | 1808-2320 | 513 |

Waymo: 22 further finished shards (indices 8-29 of 150, ~6,470 more raw scenarios) are already on
disk (`F:\CLAUDE\AI1\paper01-coverage-transfer\data\waymo_raw\validation`, download completed
2026-09-28) and will be converted into a NEW database `waymo_val2` (shards 8-29) so the fresh Waymo
pool is not the scarce 513 left in the current one.

**n for this amendment:** 1300 fresh `av2_cal`, 1300 fresh `av2_test` (leaves a safety margin over the
1376/1443 available after excluding scenarios skipped as static-ego), 1500 fresh `ns_val` (matching
the original H3 target size), 1000 fresh `waymo_val2` scenarios (from the newly-converted shards
8-29). N1-N4 use these with the PRIMARY predictor (`av2_cpu_v2`), collected once via the same
exact-evaluation harness as the main campaign (same MRM, same 2 Hz decisions; N4's $d=2.0$ sweep
reuses the `av2_test_mrm2`-style collection already implemented). **N5 requires its own full rollout
campaign** — the reference run and every forced-fire outcome depend on the predictor
(Proposition 1), so nothing can be reused from the primary-predictor pass — run with
`av2_gpu_full/epoch53-minADE0.854.ckpt` on the SAME fresh `av2_cal`/`av2_test` seed indices as N1/N2
(paired by seed, not a separate pool), so it is a second full collection over the same 1300+1300
scenarios, not additional fresh data.

### Rules carried over unchanged
n=200 resplits, Wilson 95% CIs, scenario-level paired bootstrap for differences (10000 resamples),
$\delta=0.10$ for every validity check, pre-fire exactness measured and reported exactly as in
Amendment 2. No threshold, grid, or hyperparameter for N1-N5 is chosen by looking at fresh-data
outcomes; all are fixed by this amendment or by re-running the existing Amendment-1/2 procedure
verbatim on fresh calibration data.

## Amendment 5a — 2026-09-29, protocol correction found in code testing, BEFORE any fresh data

`src/outcome_cert.py` was smoke-tested on the ALREADY-ANALYSED av2_cal/av2_test data (never fresh
data) purely to check the code runs correctly. That check surfaced a real statistical property, not
a bug, which changes how N1's validity check must be run.

**Finding:** residual harm's own floor ("always fire") is ~3.7-4.1%, close to the alpha_H=0.05
target. Any lambda with true population residual harm below alpha therefore has little margin above
that floor. A finite-sample validity check (certify on a HALF-sized calibration split, measure
whether the OTHER half's empirical rate exceeds alpha) adds its own sampling noise on top of
whatever margin the certifier achieves; that noise is far more visible near a tight floor than it is
for alarm-level miss, whose achievable floor (0.0-0.1%) sits much further from its own alpha=0.05
target. On the old, already-analysed data this pushed the half/half validity check to ~29% apparent
violation for outcome-level LTT, which is a property of checking a near-floor quantity with n_cal
~1000, not evidence LTT itself is broken (the same code, same `learn_then_test` call, on the SAME
data's alarm-level miss reproduces the already-reported ~9.5%, matching Section V of the manuscript).

**Decision, made before any fresh data is touched:**
1. alpha_H stays 0.05 (unchanged) — comparability with the existing alarm-level target is the point
   of N2, and moving the goalpost after seeing it is close to a floor would be indefensible.
2. N1's validity check uses the FULL pre-registered fresh n_cal (1300), not a further half-split of
   it, to give the certifier the calibration size Amendment 5 actually allocates, matching how N1-N4
   are deployed (not re-halved for the resplit diagnostic). The 200-resplit validity check itself
   still resplits the pooled fresh cal+test data half/half, as for H1-H3, so it remains comparable to
   the existing methodology; this decision only concerns not artificially shrinking it further.
3. **If N1 is refuted specifically because outcome-level LTT's validity check exceeds delta near this
   floor, that is reported as a positive finding, not a failure to hide**: outcome-level certification
   near an intrinsic floor is a harder statistical problem than alarm-level certification with more
   headroom, and the paper says so explicitly. N2 (the central new claim) does not depend on N1
   holding — it is a one-sided test of the ALARM-level certificate's residual harm, evaluated once on
   the fresh test set, unaffected by this near-floor validity-check sensitivity.
4. No change to N2-N5, their refutation rules, or the fresh data pools already fixed above.

## Amendment 5b — 2026-09-29, protocol correction found in code testing, BEFORE any fresh data

`src/multi_risk_cert.py` (N4) was smoke-tested using av2_cal/av2_test as a d=4.0 stand-in and the
already-collected av2_test_mrm2 sensitivity arm as a rough d=2.0 proxy for calibration -- a code
check only, not a d=2.0 calibration campaign, and the result is not used for N4 itself.

**Bug found and fixed:** the first implementation used an ascending fixed-sequence walk (same style
as single-risk LTT) with a COMBINED p-value at each lambda. This is invalid here: residual harm falls
as lambda decreases (more intervention) while induced collisions RISE as lambda decreases (more
intervention causes more rear-end collisions) -- the two risks move in opposite directions, so there
is no single walk direction both respect, and the walk failed at its very first (most-intervention)
point every time, certifying nothing. Replaced with a Bonferroni-corrected test of every grid point
independently (always valid regardless of ordering), keeping every point that passes both risks at
level delta/J and returning the one with the lowest calibration stop rate.

**Finding, confirmed structural not statistical:** with the corrected procedure, on the smoke-test
data the two risks' valid regions (residual harm p <= delta/J only for lambda close to the
most-intervention end; induced collision p <= delta/J only for lambda past roughly the 90th
percentile of the score) are COMPLETELY DISJOINT across the whole grid, independent of Bonferroni
strictness (residual harm's p-value is already ~1.0 well before induced collision's drops below 1.0).
A single scalar threshold cannot satisfy a 5% residual-harm target and a 3% induced-collision target
simultaneously with this predictor, at d=4.0 or (per the proxy) at d=2.0. This sharpens the
already-written honest expectation for N4 ("may well be refuted"): it is expected to be refuted at
BOTH d values for a specific, mechanistic reason (opposing risk directions), not merely because the
2 m/s^2 sensitivity arm showed no stop-rate difference. N4 is retained and run on real d=2.0
calibration data (not this proxy) because d changes post-fire kinematics and could plausibly shift
induced collision's achievable floor enough to open an overlap -- that mechanism is exactly what N4
tests. If no overlap is found at either d, that is reported as the result, with this mechanism
explained, not treated as a failed experiment.

No change to alpha_H, alpha_I, delta, or the N4 refutation rule already stated above.

## Amendment 5c — 2026-09-29, protocol correction found before launching collection, BEFORE any data

The "fresh scenarios available" table in Amendment 5 (1376 for av2_cal, 1443 for av2_test) counted
RAW index availability, not the non-static YIELD within that range. The already-collected av2_cal
arm shows an empirical static-ego skip rate of ~25-30% (683 static skips recorded; matches the
codebase's own `STATIC_FRAC = 0.30` sizing constant). Applying that rate to the 1376/1443 raw
indices remaining gives an estimated non-static yield of roughly 960-1010 for av2_cal and 1000-1050
for av2_test -- both BELOW the originally stated n=1300, which would have made av2_cal5/av2_test5
(and their d=2.0 and second-predictor variants) permanently unreachable: `campaign.py`'s per-worker
span is capped at `total - start`, so the arm would exhaust the entire fresh range without ever
reaching 1300 rows, and the unattended pipeline would retry it forever without ever completing.

**Correction, made before any Amendment-5 collection was launched:** av2_cal5, av2_test5,
av2_cal5_d2, av2_test5_d2, av2_cal5_gpu and av2_test5_gpu targets are reduced from 1300 to **900**
each, comfortably below the estimated ceiling. ns_val5 (6417 raw indices available) and waymo_val5
(a new ~6470-scenario database) are unaffected; their margins are large enough that the same failure
mode is not a concern. No change to alpha_H, alpha_I, delta, or any N1-N5 refutation rule; this is a
sample-size correction only, and n=900 remains large enough for the CI widths anticipated when N1-N4
were written (a residual-harm rate near 5% at n=900 has a Wilson half-width of roughly 1.5 points,
similar to the n=1994-2000 splits used for H1-H3).

## Amendment 5d — 2026-09-30, protocol correction found during collection, not from reading outcomes

`ns_val5` stalled at 1342/1500 across 6 unattended restarts (15-min ticks, 06:25-07:40), never
growing. Cause: `campaign.py` sizes its internal search span from the REQUESTED n and the hardcoded
`STATIC_FRAC = 0.30`, independent of how large the remaining raw pool actually is (span =
min(pool_remaining, ceil(n / 0.70 * 1.3))); for `ns_val5` this capped the span at 2786 raw indices
even though 6417 were available. The true static-ego rate observed in that range is ~37% (794
skips in ~2136 accounted scans), above the 30% the sizing formula assumes, so workers exhausted
their span before reaching 1500 non-static scenarios and returned early every time, indistinguishable
from a hang except that `campaign.log` shows each pass completing normally.

This is the same class of error as Amendment 5c (the fresh-pool math did not account for the true
static-skip rate), here triggered by campaign.py's OWN internal sizing rather than by the amount of
raw pool this note allocated. Since `ns_val5` is explicitly an extension pool, not required by N1-N4
or N5 (all of which finished successfully beforehand on `av2_cal5`, `av2_test5`, `av2_cal5_d2`,
`av2_test5_d2`, `av2_cal5_gpu`, `av2_test5_gpu`), the target is reduced from 1500 to **1300**,
already met by the 1342 rows collected. No hypothesis, threshold, or refutation rule depends on this
arm; nothing here is re-tested or re-interpreted based on this fix.


## Amendment 6 — 2026-09-30, BEFORE running any of the analyses below (post-hoc robustness)

Prompted by an internal peer review of the manuscript. H1-H3, H3-W and every verdict above are final
and are not reopened. Everything below is labelled post-hoc robustness in the paper. Each analysis has
a reading rule fixed here, before it runs.

**Already seen before writing this amendment (disclosed):** on `av2_cal5`/`av2_test5` the alarm-level
LTT threshold re-fit on `av2_cal5` gives test miss 3.4 % and unnecessary stops 29.0 % (printed by
`outcome_cert.py`, Amendment 5). On the second-predictor arms only the residual-harm rate (5.2 %) was
read. No validity (resplit) numbers, no T1/CDT/oracle numbers and no harm-definition variants have
been computed on any data.

### R6.1 Score-label alignment (reviewer risk: the geometric score and the nuPlan TTC harm label
are both proximity-based)
Data: `av2_cal` / `av2_test` (the headline splits, both with T1 and T2 scores).
(a) Harm = at-fault collision only (`HarmDef(use_ttc=False)`, base rate 3.2 % on av2_test, measured as
a base rate only). alpha = 0.01, delta = 0.10. If LTT refuses to certify at 0.01 (lambda = -inf), rerun
at alpha = 0.02 and report both.
(b) nuPlan definition with TTC threshold 0.5 s and 1.5 s instead of 0.95 s, alpha = 0.05.
Methods under each definition: LTT on `geom`, empirically tuned `geom`, tuned T1 (`conf_conflict`),
tuned T2 (`ens`). Reading rule: the geometric advantage "survives" a definition if LTT-geom's
unnecessary-stop rate is lower than the better of T1/T2 (among those with test miss <= alpha), with
a paired-bootstrap 95 % CI of the difference excluding 0. Otherwise it is reported as not surviving,
and the paper says the advantage is tied to the harm definition.

### R6.2 Held-out replication (reviewer risk: single split)
Calibrate on `av2_cal5`, test on `av2_test5` (fresh, disjoint from the headline splits, n = 904 each).
Methods: LTT, CRC, tuned geometric, tuned T1, CDT, oracle. T2, T3 and ACI are not available (collected
without MC dropout and without open-loop error files) and are reported as missing. Validity: 200
half/half resplits of the pooled av2_cal5 + av2_test5. Reading rule: "replicates" if LTT validity
<= delta = 0.10 AND LTT test miss <= alpha; the stop-rate ordering (LTT vs tuned vs T1) is reported
descriptively.

### R6.3 Second predictor (reviewer risk: one weak predictor)
Identical to R6.2 on `av2_cal5_gpu` / `av2_test5_gpu` (AutoBot, minADE6 0.854). Same reading rule.

### R6.4 Waymo recalibration (already pre-registered, Amendment 4B, results computed)
Report R1-R4 from `results/final/recal_waymo.json` in the paper in place of the exploratory nuScenes
few-shot table. No new computation.

### R6.5 Descriptive additions (no hypothesis)
Wilson 95 % CIs for every unnecessary-stop rate in the main table, and per-dataset AUROC/stop tables
for nuScenes and Waymo from existing outputs.

### R6.6 Failure characterization (exploratory, descriptive)
Compare missed vs correctly alerted harmful scenarios at the certified threshold on `av2_test` using
the stored pre-deployment covariates (`rollout.FEATURE_NAMES`) and time from first decision tick to
harm. Reported as descriptive statistics only, no test.

### Amendment 6 outcomes (2026-09-30), read against the rules fixed above

| Analysis | Result | Verdict |
|---|---|---|
| R6.1a collision-only harm, alpha=0.01 (LTT certified at 0.01, no fallback to 0.02 needed) | LTT-geom stops 45.1 %, best baseline T2 83.9 %; paired diff -38.8 pts [-41.4, -36.2] | geometric advantage SURVIVES |
| R6.1b TTC 0.5 s, alpha=0.05 | LTT 15.8 % vs T1 38.6 %; diff -22.7 [-25.5, -20.1] | SURVIVES |
| R6.1b TTC 0.95 s (headline, reference) | LTT 28.5 % vs T1 68.3 %; diff -39.8 [-42.4, -37.2] | SURVIVES |
| R6.1b TTC 1.5 s, alpha=0.05 | LTT 34.0 % vs T2 68.3 %; diff -34.4 [-36.8, -31.9] | SURVIVES |
| R6.2 held-out (av2_cal5 -> av2_test5, n=904) | LTT miss 3.4 %, validity 7.5 %; tuned 51.5 %, CRC 46.0 %, T1 50.0 % | REPLICATES |
| R6.3 second predictor (minADE6 0.854) | LTT miss 3.9 %, validity 9.0 %; tuned 51.0 %, CRC 47.5 %, T1 46.0 % | REPLICATES |


## Amendment 7 — 2026-10-02, BEFORE any direct-execution run exists (reviewer risk: "exact" is asserted, not tested)

Prompted by an external review of the manuscript. The paper reports pre-fire drift of forced runs but has
never compared a trigger executed *directly* (live predictor, no score replay, no forced-fire table) with
the lookup. This amendment adds that test. It changes no earlier hypothesis, threshold or verdict.

### V1 Direct execution reproduces the lookup
**Data/predictor:** the 2 000-scenario `av2_test` arm (primary predictor, same checkpoint, `mc=0`,
decide_every = 5, MRM 4.0 m/s^2, reactive traffic). A random subset of **150** stored scenarios (seed 0).
**Triggers:** geometric score, four configurations: LTT threshold certified on `av2_cal` at alpha = 0.05,
delta = 0.10 with 0 and with 1 decision tick of latency (`latency_steps` = 5); and the 25th and 75th
percentile of the calibration scenarios' peak geometric score (thresholds at which about 75 % and 25 % of
scenarios fire), zero latency. (A 3-scenario smoke test, which agreed with the lookup, showed that the
originally planned tuned thresholds at alpha = 0.01 and 0.20 were degenerate, always firing and never
firing, so they were replaced by the percentile thresholds before the main run. The smoke scenarios are not
excluded from it.)
**Procedure:** for each scenario and configuration run `rollout()` live (predictor scoring every decision
tick, no `score_trace`, no `fire_step`, no `ref_ego`), then score it with the nuPlan harm definition used
everywhere else and compare, per scenario, with `Scenario` lookup at the same threshold and latency.
**Quantities:** firing step, miss, unnecessary stop, induced collision, harm anywhere in the run,
route completion (mean absolute difference).
**Claim:** the lookup is a faithful substitute for direct execution.
**Refuted if:** for any of firing step, miss, stop, induced, harm-in-run the per-scenario disagreement
rate over all scenario x configuration pairs exceeds 5 %. Aggregate rate differences (direct minus
lookup) are reported with paired bootstrap 95 % CIs. Every disagreement is listed with its cause when one
can be identified. No scenario is excluded except those whose direct run raises an error, and the count of
those is reported.

**Run-length rule (fixed 2026-10-02, with no disagreement-based information used):** the job was launched for
120 scenarios (2 processes) but throughput on the shared machine is about 1.5 scenarios per minute in total.
It is stopped when 60 scenarios have all four configurations complete, or earlier only if it crashes, and the
analysis uses the scenarios with every configuration complete (`src/direct_validation_report.py`). The
deviation from the planned 150 scenarios is therefore a compute-time decision, not an outcome-based one.
Other quantities in the paper added after the same external review are descriptive and not hypothesis tests:
shift severity (domain-classifier AUROC), induced-collision breakdown, and a scene-level cluster bootstrap of
the nuScenes miss rate (`src/shift_and_cost.py`).

### Amendment 7 outcome (2026-10-02), read against the rule fixed above
V1: 61 scenarios x 4 triggers = 244 pairs, no direct-run errors. Disagreement with the lookup: firing step 0/244,
miss 0/244, unnecessary stop 0/244, induced collision 0/244, harm anywhere in the run 1/244 (seed 1934, certified
threshold, lookup has post-fire harm the direct run lacks; cause not traced), route completion identical in
120/244 pairs and within 0.018 in all. No quantity exceeds the 5 % disagreement rule, so the lookup is NOT refuted
as a substitute for direct execution on this sample. Limits: one dataset, predictor and maneuver; deterministic
traffic only; 61 scenarios. Files: `results/final/direct_validation.json`, `direct_validation.part*.jsonl`.
Descriptive additions: nuScenes failure under scene-level clustering is [6.1, 21.0] (37 scenes) and 9.5 %
[3.9, 16.5] on a disjoint 23-scene sample; domain AUROC 0.97 (nuScenes) vs 0.91 (Waymo); induced collisions
concentrate at early firing (9.2 % vs 3.0 %). See `results/final/shift_and_cost.json`.


## Amendment 8 — 2026-10-05, BEFORE any latency-aware certification exists (second external review)

A strict-referee review noted that the paper shows a certificate calibrated at zero latency breaking at one or two
ticks of latency, but never shows the remedy. The lookup supports the remedy without new simulation: the loss is
`miss[tick + latency]`, so LTT can be run on the latency-ell loss directly. No earlier hypothesis changes.

### L1 Certifying under the deployed latency restores validity
**Data:** `av2_cal` -> `av2_test` (primary predictor, n = 1994 / 1996), alpha = 0.05, delta = 0.10, geometric score.
**Method:** for ell in {0, 1, 2} decision ticks (0, 0.5, 1.0 s), run LTT on calibration scenarios with the loss
evaluated at latency ell, deploy at latency ell on test. 200 half/half resplits of the pooled scenarios
(seed 0, same protocol as `evaluate.validity`) give the violation frequency for the latency-ell certificate and,
for comparison, for the zero-latency certificate deployed at latency ell.
**Claim:** the latency-ell certificate has violation frequency <= delta at every ell (or refuses to certify,
which is reported), and the zero-latency certificate deployed at ell >= 1 does not.
**Refuted if:** the latency-ell certificate exceeds delta at any ell for which it certifies a threshold, or the
zero-latency certificate deployed at ell = 1 is itself valid.
Reported either way, with unnecessary-stop cost per ell.

### Amendment 8 outcome (2026-10-05), read against the rule fixed above
L1 (`results/final/latency_cert.json`, 200 resplits): violation frequency of the latency-aware certificate is 9.5 %
(ell = 0), 12.0 % (ell = 1) and 11.5 % (ell = 2); the zero-latency certificate deployed at ell = 1 and 2 violates
in 92.0 % and 100 %. No resplit refused to certify. Unnecessary stops of the latency-aware certificate rise from
28.5 % to 34.4 % to 42.5 %. **Verdict by the rule as written: REFUTED for the first half of the claim**
(the latency-aware certificate exceeds delta at ell = 1 and ell = 2); the second half (the zero-latency certificate is
not valid under latency) HOLDS. Reading: the violation frequencies' Wilson intervals include delta (see paper), the
metric counts finite-test-half exceedances rather than the population risk the guarantee bounds, and ell = 0 is
itself 9.5 %. The refutation is reported as such; no rule was changed after reading.


### Amendment 8, item I1 (descriptive, no hypothesis; 2026-10-05, before looking at the numbers)
Because every induced collision in the stored `av2_test` rows is a vehicle striking the stopped or braking ego, the
paper's "triggering is not free" claim rests on IDM follower behavior. I1 characterizes the 129 induced contacts at
the certified threshold by (a) time between the fallback's firing and the contact, (b) whether the ego had
already been stationary (stopped_steps) when it was struck, and (c) contact type. Reported as descriptive statistics.
No threshold or rule is derived from it. (`src/induced_check.py`)
I1 outcome (2026-10-05): 129 induced contacts from 51 scenarios; 99 active_rear, 30 active_lateral. Time from firing to
contact: median 8.3 s, 3.9 % within 2 s, 78 % after 5 s. 67 % of events are in scenarios whose ego starts at
standstill. Reading: not a follower surprised by hard braking; consistent with replay traffic not yielding to a
long-stationary ego, so the induced rate is an upper bound on the cost of stopping in this simulator. Paper text
corrected accordingly (it previously said the dangerous case was early stopping with traffic close behind).

### Amendment 8, item C1 (2026-10-05, before running): step sizes for the online baselines chosen on calibration data
The headline table picks the CDT step size and the ACI step size by their test-set performance (stated in the paper
as favouring the baselines). C1 repeats both with the step size chosen ONLY on the calibration scenarios: for each
eta/gamma in {0.01, 0.05, 0.1, 0.5}, run the online rule over 20 random orders of the calibration scenarios with the
same selection criteria as the headline, then deploy the selected value on the test scenarios (20 orders, mean).
Reading rule: if the calibration-selected variant changes either method's miss rate by more than 1 point or its
stop rate by more than 3 points, the main table's online rows are replaced by it in the paper; otherwise the
difference is reported in one sentence. (`src/online_calsel.py`)
C1 outcome (2026-10-05): CDT with the step chosen on calibration (0.01): miss 4.6 %, stops 25.1 % versus test-selected (0.1)
5.0 % / 24.0 %; ACI selects 0.5 either way, identical. Both differences are inside the stated thresholds (1 point miss,
3 points stops), so the main-table rows are kept and the difference is reported in one sentence.


## Amendment 9 — 2026-10-05, BEFORE any of the analyses below exist (third external review: robustness and breadth)

Written before running anything in this amendment. It adds natural shifts and baselines the paper lacks. None of H1-H3,
alpha, delta or any earlier verdict changes; every item below is evaluated once, with the reading rule fixed here.

### A1 Leave-one-city-out: does the certificate survive natural shifts inside Argoverse 2?
City labels come from the raw AV2 scenario parquet (`city`), joined to stored rows through the scenario id in the
ScenarioNet file list. **Data:** all stored `av2_cal` + `av2_test` rows (about 3,990), cities with n >= 150.
**Method:** for each such city c, run LTT (geometric score, alpha = 0.05, delta = 0.10) on all scenarios NOT in c and
deploy the threshold on c, reporting miss and unnecessary stops with Wilson 95 % CIs; the tuned (uncertified)
threshold is reported alongside. **Claim:** the certificate transfers across cities, i.e. the held-out miss rate
(point estimate) is <= alpha for at least 80 % of the evaluated cities. **Refuted if** fewer than 80 % of the cities
have miss <= alpha. Per-city miss at the in-pool threshold is also reported descriptively (heterogeneity), without test.

### B1 Behavior-model shift: traffic that does not react
Thresholds are certified/tuned on the reactive `av2_cal` and deployed on the non-reactive log-replay arm
`av2_test_replay` (n about 504, paired by seed with `av2_test`). **Claim:** the geometric certified trigger still
meets alpha = 0.05 on replay traffic. **Refuted if** its miss exceeds 0.05 with the Wilson interval excluding 0.05
from above. Also reported: whether the stop-rate advantage over tuned T1 persists (paired bootstrap, among methods
meeting the target; if T1 does not meet it, reported as such).

### C1 Induced collisions: severity and striker behavior (re-simulation of the affected scenarios only)
For the scenarios with an induced collision at the certified threshold on `av2_test` (51), re-run the single forced
rollout at the stored firing tick with the same seed and record, for each contact, ego speed, striker speed, relative
closing speed and the striker's distance to the ego at the firing step. Descriptive only. The forced rollout must
reproduce the stored collision steps; scenarios that do not are reported as non-reproduced.

### D1 Larger direct-execution check
Repeat Amendment 7's V1 (same four triggers, same rule: refuted if any quantity disagrees in more than 5 % of pairs)
on a second independent random draw (seed 1) of about 140 further `av2_test` scenarios, merged with the first 61 by seed.

### E1 Second predictor, ensemble baseline
For the second predictor (`av2_gpu_full`, minADE6 0.854) the reference run of every `av2_cal5_gpu` / `av2_test5_gpu`
scenario is re-run with 5 Monte Carlo dropout passes to obtain the ensemble-spread trace; forced-fire outcomes are
reused (they do not depend on any score; the re-run must reproduce the stored tick count and geometric trace, else the
scenario is dropped and the count reported). **Claim:** with this predictor, LTT-geometric stops less than the tuned
ensemble trigger T2 (among triggers meeting miss <= alpha on test; paired bootstrap 95 % CI excludes 0). **Refuted if** it
does not. T3 (open-loop conformal) is not repeated because it needs open-loop error files for this predictor.

### Amendment 9 outcome A1 (2026-10-05), read against the rule fixed above
Leave-one-city-out (`results/final/city_shift.json`, pool n = 3,990, 6 cities, each n >= 229): LTT held-out miss
austin 3.7 %, dearborn 2.0 %, palo-alto 4.4 %, pittsburgh 4.4 % (<= 5 %), miami 6.1 % [4.8, 7.8], washington-dc
5.3 % [3.6, 7.8] (> 5 %). 4 of 6 = 67 % < 80 %, so **A1 is REFUTED** by the rule as written. Tuned (uncertified)
threshold meets alpha in 3 of 6. The two failing cities have the highest harm base rates (19.7 %, 19.3 %); the
in-pool threshold, which saw those cities, also misses 5.2 % and 5.1 % there, so the failure is the marginal
certificate meeting a higher base rate, not a calibration artifact. Miami's interval excludes alpha, DC's does not.

### Amendment 9 outcome B1 (2026-10-05)
Behavior-model shift (`results/final/behavior_shift.json`, 378 scenarios with complete records, paired reactive vs replay):
harm base rate 16.7 % (reactive) vs 17.7 % (replay). LTT-geometric certified on reactive calibration data misses 3.7 %
[2.2, 6.1] on replay traffic, so **B1 is NOT refuted** (miss <= alpha, interval not excluding alpha from above). Tuned T1
also meets the target on replay (3.4 %), at 71.7 % stops against 35.4 % for LTT; paired stop difference -36.2 points
[-41.8, -30.7]. The geometric advantage persists under a different traffic behavior model.

### Amendment 9 process note D1 (2026-10-05)
The stop signal described in Amendment 7's run-length rule did not actually terminate the first direct-execution job
(the process filter missed it), so that job ran to its full planned 120 scenarios (480 rows) after the paper's
61-scenario snapshot was analysed. The paper's earlier figures (61 scenarios, 244 pairs) were taken from that snapshot
as the rule stated. The final D1 report merges the complete first draw (120 scenarios) with the second independent draw
(seed 1), by seed, as pre-registered in spirit; the snapshot figures are superseded, not hidden.

### Amendment 9 outcome C1 (2026-10-05) (`results/final/induced_severity.json`, descriptive)
All 51 scenarios with an induced collision at the certified threshold re-run (no errors). Re-run reproduces the stored
induced-contact steps exactly in 34/51 and the FIRST induced contact in 49/51; later contacts diverge in 17/51 (contact
dynamics after a first impact are not run-to-run reproducible). At the first induced contact: ego speed median 0.58 m/s
(27 % below 0.1 m/s, 65 % below 1 m/s), striker speed median 3.6 m/s, speed difference median 2.4 m/s (IQR 1.2-5.8,
max 11.8), 19 lateral and 32 rear contacts; time after firing median 6.5 s. At firing an agent was in the ego's lane
behind it in 39/51, at median gap 8.3 m and median speed 0 m/s. Reading: low- to moderate-speed contacts with a slow
or stationary ego by traffic proceeding along logged paths; not hard-braking surprises.

### Amendment 9 outcome D1 (2026-10-06) (`results/final/direct_validation.json`)
Merged draws: 252 distinct av2_test scenarios x 4 triggers = 1,008 pairs, 0 direct-run errors. Disagreement with the lookup:
firing step 0, miss 0, unnecessary stop 0, induced collision 0 (each 0/1008, Wilson 95 % upper bound 0.38 %), harm anywhere in
the run 1/1008 (seed 1934 again, certified threshold; lookup has post-fire harm the direct run lacks). Route completion
identical in 454/1008 pairs, mean absolute difference 0.00015, max 0.036. Aggregates identical per trigger (e.g. LTT
miss 3.6 %, stops 27.8 %; one-tick latency miss 4.8 %). No quantity reaches the 5 % rule: lookup NOT refuted as a
substitute for direct execution. The 61-scenario snapshot in earlier text is superseded.

### Amendment 9 outcome E1 (2026-10-06) (`results/final/ens_gpu.json`)
Second predictor (minADE6 0.854), reference runs re-run with 5 MC-dropout passes: 865/904 calibration and 867/904 test
scenarios reproduce the stored reference (tick count and geometric trace); 39 and 37 are dropped (4.3 %, 4.1 %). On the
reproduced scenarios: LTT-geometric miss 3.2 % [2.2, 4.6], stops 26.4 %; tuned T1 5.0 % [3.7, 6.6], 66.2 %; tuned T2 ensemble
5.7 % [4.3, 7.4], 59.6 %. T2 does NOT meet the 5 % target on test, so the rule's condition ("among triggers meeting
miss <= alpha") excludes it and the comparison is not evaluable as written; reported unconditionally: LTT stops fewer
than T2 by 33.2 points (paired 95 % CI [-37.5, -29.0]) and than T1 by 39.8 [-43.7, -35.8], with a lower miss rate than both.
The reading-rule's literal verdict flag is False only because T2 fails the target. The geometric advantage over the
ensemble trigger therefore holds with the second predictor, with the caveat that T2 is not a valid comparator at 5 %.


## Amendment 10 — 2026-10-06, BEFORE any Wayformer rollout or spatial-group analysis exists

Motivated by two remaining reviewer risks: (i) every result uses AutoBot, and (ii) scenarios from one place or drive are
not independent. A stronger AutoBot cannot be trained on this machine (Paper 01's GPU is contended; its best AutoBot,
minADE6 0.854, is already the second predictor), so the generalization test uses a different ARCHITECTURE instead.

### W1 Second architecture (Wayformer)
**Predictor:** `av2_wayformer/epoch19-minADE0.967.ckpt` (Paper 01 artifact, a partly trained checkpoint, a fixed file;
UniTraj Wayformer, same 2.1 s history / 6 s horizon, 6 modes). **Data:** the same seeds as `av2_cal5` / `av2_test5`
(904 / 904, mc = 0, decide every 5 steps, MRM 4.0 m/s^2, reactive IDM traffic), full tick-sweep campaign as in the main
study. T2 and T3 are not repeated. **Before the campaign:** a sanity check that the Wayformer wrapper reproduces
plausible forecasts (median 3 s endpoint disagreement with AutoBot on identical simulator states below 5 m).
**Claim (both parts):** (a) LTT with the Wayformer geometric score is valid: violation frequency over 200 resplits
<= 0.10 and test miss <= 0.05; (b) the geometric certified trigger stops less than tuned T1 (Wayformer confidence)
among triggers meeting the target, paired bootstrap 95 % CI excluding 0. **Refuted if** (a) or (b) fails, reported either
way. Scenarios whose reference run errors are dropped and counted.

### G1 Spatial-group dependence (Argoverse 2 and Waymo)
Scenarios are grouped by the ego's initial position: connected components of scenarios whose ego start points are within
150 m of each other (single linkage), computed separately per dataset (and per city for Argoverse 2). This is a proxy for
"same place or drive"; no recording identifier exists in the converted data. **Group-level validity:** 200 resplits of the
pooled Argoverse 2 calibration + test scenarios in which whole groups are assigned to calibration or test (about half the
scenarios each). **Claim:** LTT's violation frequency under group-level resplits is <= 0.10. **Refuted if** it exceeds 0.10
(point estimate). Also reported: a group-bootstrap 95 % CI (10,000 draws) of the miss rate of the certified threshold on
Argoverse 2 test and on Waymo, compared with the Wilson interval, and the group count and sizes. Descriptive.

### Amendment 10 status (2026-10-07): PAUSED at the author's request
The Wayformer campaign (`results/campaign/av2_cal5_way`, `av2_test5_way`) and the spatial-group job were stopped by hand after
a partial run. Collected rows are kept and the collection is resumable (re-run `run_way_cal.sh` / `run_way_test.sh`; finished
seeds are skipped). No Wayformer or spatial-group outcome has been read, so the pre-registered rules above stand unchanged.
The spatial-group analysis (`src/spatial_groups.py`) restarts from scratch (about 15 min of file reading plus 200 resplits).
Amendment 10 RESUMED 2026-10-07 at the author's request (same commands, detached processes).

### Amendment 10 outcome G1 (2026-10-07) (`results/final/spatial_groups.json`)
Groups = ego start points within 150 m (single linkage, per city for AV2): 547 groups over the 3,990 pooled AV2 scenarios
(largest 649 = 16.3 %, median 1, 303 singletons); Waymo 664 groups over 1,000 scenarios (largest 13). **Group-level resplits
(200): LTT violation frequency 17.0 % > 10 %, so the pre-registered claim is REFUTED** (mean group-resplit test miss 4.2 %;
calibration share about 53 % because whole groups are assigned). Post hoc sensitivity at 50 m (1,796 groups, largest 79):
14.0 %. Group-bootstrap 95 % intervals barely widen: AV2 test miss 3.7 % Wilson [2.9, 4.6] vs groups [3.0, 4.5] (360 groups);
Waymo 2.7 % Wilson [1.9, 3.9] vs groups [1.7, 3.8] (664 groups). Reading: place-level dependence does not widen the miss
intervals, but leaves LTT's high-probability guarantee above delta under group-wise splitting; the violation estimates carry
Monte Carlo error (17 % has Wilson interval [12.4, 22.8] over 200 resplits).
