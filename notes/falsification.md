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
