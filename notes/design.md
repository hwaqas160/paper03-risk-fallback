# Paper 03 — Technical Design (draft, 2026-09-15)

Working design derived from PLAN.md + the five reference papers. Decisions marked
**[OPEN]** need the author's call before experiments; nothing here is pre-registered —
that happens in `falsification.md`.

---

## 1. Closed-loop stack (what runs inside MetaDrive, headless)

```
ScenarioNet scenario (AV2 / nuScenes, replayed)
   │  other agents: logged replay (default) or IDM-reactive  [OPEN: reactive_traffic]
   ▼
Ego nominal policy  ── TrajectoryIDMPolicy following the logged ego route
   │
   ├─ every decision tick Δ (e.g. 1 Hz; latency sweep = Exp 7)
   │     predictor  f(history)  →  K modes per nearby agent        (Paper 01 AutoBot)
   │     uncertainty u_t         →  one of the 4 triggers below
   │     trigger     u_t > λ     →  FALLBACK
   ▼
Fallback = minimum-risk manoeuvre: brake to stop at ≤ a_max in the current lane,
           hold until the conflict clears or the episode ends.
```

Outcome metrics per episode (PLAN §7 wk 3–5): collision (any), TTC < 1.5 s violation,
**unnecessary stop** (fallback fired, but the counterfactual no-fallback rollout of the
same scenario is collision-free), **stopped-in-live-lane** duration, route completion,
comfort (peak |jerk|, peak decel).

The counterfactual "unnecessary stop" label is cheap here: every scenario is also run
once with the trigger disabled (λ = ∞). This is the measurement open-loop papers can't make.

**Implemented** in `src/rollout.py` (2026-09-15): `FallbackIDMPolicy` keeps the lane-keeping
PID steering and commands a fixed brake, so the ego stops *in its lane* — the failure mode
the paper is about. `sweep_scenario()` produces the per-scenario `loss` / `stop` /
`triggered` rows that `src/risk_control.py` consumes. A miss requires the trigger to fire
*strictly before* the first harmful step — firing afterwards is not an intervention.

Measurement choices that turned out to matter (both were bugs first):
- Vehicle extents must be **lateral half-widths**, not half-diagonals. With half-diagonals
  (~2.3 m each) two cars in adjacent lanes 3.5 m apart always read as overlapping, which
  made min-TTC 0 in every scenario.
- TTC uses constant-velocity **closest point of approach**, counting an agent only if its
  closest approach falls inside the combined half-widths + 0.5 m, minus the half-lengths
  over closing speed so a rear-end approach is not credited extra time.

**[SUPERSEDED by §7 and the falsification addendum]** The "~2 %" IDM crash rate below was
a measurement bug (the benchmark only checked the final step); the true figures are 11.1 %
under log replay and 3.7 % with reactive traffic. Original note:
With the stub predictor, 4 of 7 pilot scenarios contain a TTC < 1.5 s event,
i.e. a ~55 % base rate of "harm". That is high for replayed logs and is driven by the TTC
threshold, not by collisions (IDM crash rate is ~2 %). Either the threshold or the
violation definition (instantaneous vs sustained) needs tightening before any real claim —
a base rate that high makes α = 0.1 unreachable by construction.

## 2. The four triggers (Exp 2)

| # | Trigger | u_t |
|---|---|---|
| T1 | Fixed confidence threshold | 1 − max mode probability of the most conflicting agent |
| T2 | Ensemble variance | spread of endpoints across M AutoBot seeds (M = 5, 1.5 M params each — fits P2000/CPU) |
| T3 | Conformal region size | Paper 01 split-CP radius q̂ on the predicted modes, intersected with the ego plan (region overlaps ego corridor within horizon) |
| T4 | **Risk-calibrated (ours)** | same score family as T3, but λ chosen by the procedure in §3, with a guarantee on the *decision outcome* |

T1–T3 thresholds are "hand-tuned": swept, and the best on the calibration split is
reported (a strong, honest baseline — H2 is refuted if a well-tuned T1–T3 matches T4).

## 3. Formalisation — fallback as selective prediction with a decision-level guarantee

For a scenario X and threshold λ, run the closed loop and record the bounded loss

    L(X, λ) = 1{ a harmful event (collision or TTC violation) occurs and no fallback fired before it }

i.e. the **missed-intervention** indicator, measured *in closed loop under λ*.
Goal: choose λ̂ such that

    P( E_X[ L(X, λ̂) ] ≤ α ) ≥ 1 − δ

while maximising autonomy (minimising unnecessary-stop rate).

Why not plain split CP: the loss depends on the whole closed-loop rollout under λ (the
fallback changes the future), so it is neither a coverage event nor guaranteed monotone
in λ. **Learn-then-Test** (Angelopoulos, Bates, Candès, Jordan, Lei 2021) handles exactly
this: for a grid λ₁ > λ₂ > … (less → more conservative), compute on n calibration
scenarios the empirical risk R̂(λ_j), a Hoeffding–Bentkus p-value for H_j: R(λ_j) > α,
and apply fixed-sequence testing from the most permissive λ. Every λ that rejects is
valid with FWER δ; pick the most permissive one. If monotonicity does hold empirically,
Conformal Risk Control (Angelopoulos et al. 2022) gives the tighter single-λ version —
report both.

Guarantee holds under exchangeability of calibration and test *scenarios*. H3 is then a
direct statement: cross-dataset shift (AV2 → nuScenes) breaks exchangeability, and the
realised miss rate exceeds α. The Paper 01 recalibration methods (weighted CP via
`domain_classifier_weights`) extend to weighted LTT — a natural repair experiment.

Compute cost: |grid| × n_cal closed-loop rollouts (+1 counterfactual per scenario).
E.g. 20 λ × 2 000 scenarios = 40 000 rollouts → this is why the throughput gate matters.
Measured with the stub predictor: ~38 s per scenario for a 9-λ grid + counterfactual in one
process (scoring costs ~3× the bare sim: 17 vs 45 steps/s).

**The λ grid must come from the data.** A grid fixed a priori is useless: the score is in
metres of predicted intrusion, so every λ below the bulk of the distribution triggers always
(autonomy 0) and every λ above it never (autonomy 1). `run_sweep.py` runs a *probe* set at
λ = ∞ and puts the grid on quantiles of the per-scenario peak score. Probe scenarios are not
reused for calibration — that would break exchangeability and void the guarantee.

Relation to the reference papers:
- **SafePath** — CP set → act / delegate; guarantee is on set membership, closed loop only
  in highway-env with synthetic traffic. We guarantee the *closed-loop outcome* on replayed
  real scenarios and measure the cost of delegating (unnecessary stops).
- **Task-relevant failure detection (Farid et al.)** — cost-based p-quantile anomaly with
  FPR/FNR bounds, evaluated open-loop against hand labels. Their QAD is a fifth candidate
  trigger **[OPEN: include as baseline?]**; their "task-relevance" motivates using
  plan-intersecting region size in T3/T4 rather than raw region size.
- **Robust CP under shift (Rahaman et al.)** — robust quantile inflation with a nuisance
  parameter, ORCA toy corridor. Candidate shift-repair arm for H3.
- **CVaR safety (Chapman et al.)** — vocabulary for a severity-aware variant: replace the
  0/1 loss by min-TTC shortfall and bound CVaR_β instead of the mean **[OPEN: stretch]**.

## 4. Data

| Role | Source | Status (2026-09-15) |
|---|---|---|
| Calibration + in-domain test | AV2 val `av2_splits/val/{cal,test}` (4 971 / 5 027) | converted, **replays headless ✔** |
| Shifted test (H3, Exp 6) | nuScenes | prediction-challenge `val` conversion partial (val_4..7 still `_tmp`); those snippets are ~2.5 s — **too short for closed loop** |
| | nuScenes full logs (`v1.0-trainval`, ~20 s scenes) | not converted; metadata + maps are already on disk, no sensor blobs needed **[OPEN]** |

AV2 scenarios are 11 s (≈110 steps at 10 Hz). ~2 of the first 11 have a static ego
(< 10 m travel) and must be filtered, as ScenarioNet does.

## 5. Dependencies on Paper 01

- **Trained AutoBot checkpoint** — needed for T1–T4 and everything in H1/H3. As of today
  `av2_full_v1` (GPU) died at epoch 0 step 34 on 2026-09-11; `av2_cpu_v1` (60 k scenes,
  10 epochs) is ~73 % through epoch 0 at ~7.5 s/it. Until a checkpoint exists, Paper 03
  proceeds with plumbing + an oracle/noisy-GT predictor stub so the triggers and LTT
  machinery can be built and tested end-to-end.
- **Closed-loop predictor adapter** — AutoBot consumes UniTraj-format agent-centric
  tensors built from ScenarioNet scenario dicts (`unitraj_bridge.build_loader`). This is
  the largest plumbing item and the main schedule risk. Two routes:
  - *(a) Offline cache (recommended first).* With log-replayed traffic the other agents'
    histories at every tick are fixed, so predictions for every (scenario, tick, agent)
    can be computed once in batch — re-slicing each scenario at `current_time_index = t`
    — and looked up during rollout. Decouples the sim (CPU, 18 cores) from the network
    and makes the λ sweep cheap. Cost: the ego's *deviated* history is not seen by the
    predictor (it sees the logged ego). Consistent with non-reactive traffic.
  - *(b) Online.* Build a truncated scenario dict from sim state at each tick and run the
    UniTraj preprocess + AutoBot in the loop. Needed only if `reactive_traffic=True`.
  **[OPEN]** (a) vs (b) — decides whether other agents may react to the ego's stop.
- `conformal.py` — ported (imported, not copied) from Paper 01.

## 6. Environment

Reusing Paper 01's venv (`F:\CLAUDE\AI1\shared\envs\unitraj`, Python 3.10.11,
metadrive-simulator 0.4.2.3, scenarionet editable from Paper 01's clone) — the same
versions that converted the data. The clones under `code/` (MetaDrive 0.4.3) are **not**
installed, to avoid changing packages under Paper 01's running jobs.

---

## 7. Decisions forced by data (2026-09-21)

**Reactive traffic is the default.** Pure log replay tripled collisions (11.1 % vs 3.7 %
on 1 000 dev scenarios) because logged agents cannot react to an ego that leaves its logged
trajectory — and a fallback *is* such a deviation. Keeping replay would bake a simulator
artifact into "freezing is unsafe". `run_sweep.py` uses `reactive_traffic=True`; `--replay`
exists only to reproduce the contrast. This resolves the §1 **[OPEN]** on traffic mode.

**Consequence for the predictor adapter (§5).** With reactive agents, other agents'
histories depend on what the ego did, so route (a) "offline cache" is only exact until the
ego first deviates from its log. Options, in order of preference:
1. *Online* AutoBot in the loop (route b) at the decision rate, batched per tick on CPU —
   correct by construction; cost to be measured.
2. *Hybrid*: offline cache until the first trigger / first deviation > x m, online after.
3. Offline cache as an approximation, with the approximation error reported.

**Common random numbers.** Every rollout of a scenario reseeds the (stochastic) predictor
from the scenario seed, so different λ see identical noise. Without it the trigger was not
monotone in λ (autonomy fell 0.714 → 0.571 as λ rose). Test: `test_rollouts_are_reproducible`.
The 5 % "non-monotone in λ" rate reported from the first (pre-fix) pilot was most likely
this bug: with CRN + reactive traffic, 300 scenarios show **0 %**. Non-monotonicity is
therefore NOT an established finding; LTT stays the default because it doesn't need it.

**Harm definition is post hoc.** Rollouts store the per-step TTC trace and collision step;
`rollout.HarmDef` scores any definition afterwards, and `run_sweep.py` reports LTT under
every candidate from one set of simulations. Choosing the headline definition is the
author's call (`notes/falsification_proposal.md`, addendum).

---

## 8. Harm definition adopted: nuPlan (2026-09-22)

On the author's instruction ("take the best harm definition used in top journal papers"),
the headline harm is the **nuPlan closed-loop benchmark** pair (Karnchanachari et al.,
ICRA 2024; nuplan-devkit metrics, constants verified in the devkit source):
`no_ego_at_fault_collisions` + `time_to_collision_within_bound` (TTC < 0.95 s, 3 s horizon,
0.1 s step, skipped while ego stopped, tracks behind ego ignored). Implemented in
`src/nuplan_metrics.py` (6 unit tests), deviations listed in its docstring.

Why it fits this paper specifically: nuPlan does **not** blame the ego for being rear-ended
or hit while stopped. Those are exactly the collisions a fallback *causes*. So they are
scored as `induced_collision` (a cost of over-conservatism) and can never be counted as a
"miss" — the accounting separates the two failure modes the paper is about.

Base rates, 1 000 dev scenarios, reactive traffic, no fallback: nuPlan harm **12.9 %**
[11.0, 15.1]; at-fault collision alone 3.1 %; any collision 3.7 %. Chosen α = 0.05, δ = 0.1.

Also fixed on the way: MetaDrive's `info["crash"]` includes sidewalk / boundary contact;
collisions now use agent contact only (vehicle, human, object), with road-edge contact
tracked separately as `offroad` (nuPlan's drivable-area metric). Pedestrians and cyclists
are now included in both TTC and the trigger score (previously vehicles only).

## 9. AutoBot in the loop (2026-09-22)

`src/autobot_predictor.py` runs Paper 01's `epoch08-minADE1.349` checkpoint online:
- builds UniTraj's track table from the **simulator's** recorded 2.1 s history, reuses
  UniTraj's own `process()` / `collate_fn` (map features cached per scenario), rotates the
  agent-frame output back to world coordinates;
- vehicles only (Paper 01 trained with `object_type: ['VEHICLE']`); pedestrians / cyclists
  fall back to constant velocity — a stated limitation;
- cost on one CPU thread: ~0.35 s + 0.16 s per predicted vehicle per tick; a whole scenario
  scored at 10 Hz ≈ 98 s, at 2 Hz ≈ 26 s; ~2.8 GB commit per worker.

**Score-trace reuse (exact, not an approximation).** The sim is deterministic under common
random numbers, so every λ-rollout equals the λ = ∞ reference run until its own trigger,
and after the trigger no score is needed. The reference run's score trace is therefore
replayed for all λ; a per-step ego-position check (`diverged`) verifies the identity and
must report 0. Test: `test_score_trace_reuse_is_exact` (3 scenarios × 3 λ, identical
trigger step, length, collisions and TTC trace). Predictor cost drops from
|λ grid| + 1 runs per scenario to 1.

Budget for the pre-registered study at 10 Hz: 4 000 scenarios × ~100 s / 8 workers ≈ 14 h.

## 10. Fallback manoeuvre = UN R157 MRM; episode length; simulator determinism (2026-09-22)

**The fallback was an emergency stop — fixed.** The first `FallbackIDMPolicy` commanded a
fixed brake action (0.5), which MetaDrive turned into ~10 m/s² peak deceleration (p50 9.9,
p90 13.2). The rear-end "induced collisions" measured with it (26/300 dev scenarios) are
therefore an artifact and are NOT reported as evidence that stopping is unsafe.
The fallback is now the regulatory minimum-risk manoeuvre of **UN R157 (ALKS) §5.5.1**:
slow down inside the lane with deceleration demand ≤ 4.0 m/s². A P-controller tracks
v(t) = max(v0 − 4.0 t, 0): measured median 4.0 m/s², brief transients ≤ 5.6 m/s², stops in
v0/4 s. Test: `tests/test_mrm.py`. Induced collisions must be re-measured under it.

**Episode length.** With the gentler stop, a stopped ego never "arrives", and episodes ran
to the 1 000-step horizon (100 s) on 11 s scenarios. Episodes now end 2 s (20 steps) after
the logged scenario (`allowed_more_steps=20`): ~7× less simulation per stopped rollout, and
no traffic driving on past its log.

**Determinism.** Pre-trigger, runs of a scenario are identical across λ (0 divergences), so
score-trace reuse stays exact. Post-trigger MetaDrive is not bit-repeatable: two identical
live runs differed in 2/30 (scenario, λ) pairs. LTT treats each scenario's closed-loop loss
as a random variable and does not require a deterministic simulator; the rate is reported
as a property of the testbed.

**Miss-floor diagnosis (stub predictor, dev).** Of 13 misses at λ = 0, 10 never triggered:
the stub's score stayed exactly 0 because the harmful agent was never predicted into the ego
corridor. The floor is a predictor limitation, which is what AutoBot should reduce.

## 11. First AutoBot-driven pilot (2026-09-22, dev split, 150 scenarios)

|              | lam=0 (aggressive) | lam=2.40 (permissive) |
|---|---|---|
| miss rate    | 0.080 | 0.113 |
| autonomy     | 0.587 | 0.900 |
| unnecessary stop | 0.360 | 0.080 |
| induced collision | 0.040 | 0.007 |

No-trigger base rate (nuPlan headline harm): 0.127. **LTT does not certify at α=0.05 for the
headline harm definition at any grid point** — even lam=0 (score threshold 0, the most
aggressive setting tested) leaves a 0.080 miss rate. It DOES certify for narrower
definitions on the same rollouts: "at-fault collision only" and "any collision" both
certify at lam=0.471 (rescored post hoc from the same data, no re-simulation).

**Diagnosis: the miss floor is a detection gap in the trigger score, not the harm
definition.** Of 12 misses at lam=0, 11 never triggered at all because `trigger_score()`
stayed exactly 0 for the whole rollout — AutoBot predicted the conflicting agent, but its
predicted mode never intersected the ego's *nominal* plan corridor (a constant-speed
extrapolation of the reference trajectory) closely enough to register. This is the same
failure mode seen with the stub predictor (§ note above: 10/13 misses, "never triggered
because score stayed 0"), so switching predictors alone does not fix it. Candidate causes
worth investigating before claiming H2: the "ego plan" proxy ignores that IDM itself
brakes/turns in response to traffic (so the corridor is wrong exactly when it matters most);
`SAFETY_MARGIN` and the corridor width may be too narrow; scoring the single worst timestep
across K modes may under-weight a threat that is real but not the argmax mode.

**Score-trace reuse: exact where it matters, honest where it doesn't.** 3/150 scenarios
(all at the most permissive lam, all `triggered=False`) showed `diverged=True` — but in
every case the harm label and collision count were IDENTICAL between the live and replayed
run; only the ego's exact position drifted by centimetres. Root cause: MetaDrive is not
bit-repeatable across two independently-run full-length episodes of the same seed (the
same simulator-level noise already noted for post-trigger runs in §10, now also observed
pre-trigger at low rate — 2% of never-triggered scenarios here). This does not corrupt any
loss label in this run and LTT does not require a deterministic simulator, but the earlier
claim "reuse is exact" should be read as "exact up to simulator-level physics noise that
does not change collision/TTC outcomes in the cases observed" — not a stronger guarantee.

Runtime: 39.4 min for 150 scenarios x 7 rollouts (1 live AutoBot run + 6 replayed) on 3
workers (commit-capped: AutoBot needs ~2.8 GB/worker vs 1.8 GB for the stub).

## 12. Trigger geometry fix confirmed; a second, distinct floor remains (2026-09-22)

**The length/width fix worked as intended.** Re-running the 150-scenario dev pilot with the
path-relative box scorer (§11 diagnosis): "at-fault collision" and "any collision" now
**certify cleanly** at λ=0.564 (miss 0.007–0.02, vs. no certification at all before the
fix). The headline nuPlan harm (which also counts TTC<0.95s) still does not certify at
α=0.05: min miss 0.073 (down from 0.080 — a real but small improvement), autonomy up to
0.76 at the top of a narrower grid (0–1.70 this run; grid width depends on the probe sample).

**Diagnosis of what's left.** Of 11 misses at λ=0, all 11 are TTC-only (no collision) and
10/11 never triggered at all — but this time NOT because of the width/length bug: spot-
checking 4 cases (seeds 60, 74, 86, 127), the trigger score was exactly 0.000 at the
decision tick immediately before each TTC<0.95 event (0.5 s earlier, `decide_every=5`), the
culprit was a VEHICLE (never a pedestrian) at 9.3–9.6 m gap, and AutoBot's own predicted
trajectory for that vehicle 0.5 s earlier did not intrude the ego's corridor. So the model's
prediction itself, not the scoring geometry, missed these.

**Working hypothesis, not yet confirmed:** AutoBot was trained on Paper 01's real human
driving logs, but this pipeline replays scenarios with `reactive_traffic=True` — other
vehicles are controlled by MetaDrive's rule-based **IDM**, which can react abruptly (e.g. to
the ego's own fallback braking) in ways a model trained on human trajectories would not
anticipate. If true, this is not a bug to patch away: it is exactly the kind of
train/deployment mismatch the whole 3-paper programme is about, and arguably belongs in the
paper as a finding, not a limitation to hide. **Not yet tested**: whether the effect is
specific to IDM-reactive agents (vs. equally present in log-replay) — that comparison would
confirm or kill the hypothesis directly.

**Alternative, uninteresting explanation also live:** 3 s prediction horizon at 0.5 s
decision cadence may simply be too infrequent for fast-developing conflicts. Testing this
now: rerunning at `decide_every=1` (10 Hz, every sim step) on 40 scenarios. If the floor
drops substantially, latency (not IDM-vs-human distribution) is the primary cause — still a
legitimate, reportable finding (Exp 7 is a latency sweep for exactly this reason), but a
different one from the domain-gap hypothesis above.

## 13. Correction: the first decide_every comparison was invalid (2026-09-22)

The exploratory `decide_every=1` run (n=40, `--n_probe 15`) used a **different probe span**
than the `decide_every=5` run it was meant to compare against (`--n_probe 30`), so it drew
scenarios from an earlier, disjoint slice of `av2_dev` seeds. Its higher miss rate
(0.103 vs 0.073) is therefore NOT evidence about decision cadence — it may simply be a
harder sample of scenarios. **Do not cite that number.**

`run_sweep.py` has no `--start` flag; the seed offset is `probe_span`, a pure function of
`--n_probe` (`ceil(n_probe / 0.7 * 1.3)`) — it doesn't depend on whether `--lams` is given.
Re-running as a properly paired test: matching `--n_probe 30` (so both runs start at seed
56) with an explicit `--lams` (skips simulating the probe, but keeps the offset), `--n 36`,
varying only `decide_every` ∈ {1, 5}. Results below once both finish. Controlled version of
the §12 hypothesis test.

## 14. Paired decide_every result (valid); root cause of the near-miss floor found

**Decision cadence is not the cause.** Properly paired (identical 35 scenarios, identical
λ grid, only `decide_every` varied): miss rate was IDENTICAL at every λ between 2 Hz
(`decide_every=5`) and 10 Hz (`decide_every=1`) decisions — 0.057, 0.057, 0.086, 0.086,
0.086 in both. Autonomy was marginally lower at 10 Hz (more frequent checks catch a few
more borderline triggers), but the miss floor itself did not move at all. §12/13's earlier,
invalid numbers are superseded; this is the number to cite.

**Root cause of the floor: the trigger's ego-motion model disagreed with the harm metric's.**
`trigger_score()` compared predicted agent positions to a corridor built from
`ego_plan_route()` — the ego's position along the **route's** curved reference path at
current speed. The harm label (`nuplan_metrics.ttc_nuplan`) — both here and in the real
nuPlan definition — extrapolates ego with a **route-agnostic constant-heading** kinematic
model instead (deliberately: assuming the driver's planned steering will save them is
exactly the complacency a safety metric should not grant). Measured divergence between the
two ego-position models on the two spot-checked near-misses: up to 8 m at a 1 s horizon,
30 m at 3 s (seeds 60, 74). Separately confirmed AutoBot's own agent-position prediction was
accurate (0.03 m error against the eventual culprit position) — the predictor was not at
fault; the trigger was scoring against the wrong notion of where the EGO would be.

**Fix:** `trigger_score()` now uses `ego_plan_kinematic()` — same route-agnostic
constant-heading model as the harm metric — so the score and the thing it is trying to
pre-empt are evaluated under one consistent assumption about ego's own near-term motion.
`ego_plan_route()` is kept (unused by default) for a future ablation. Regression test
`test_trigger_score_sees_lead_vehicle_not_just_width` still passes; re-running the same
paired 35-scenario validation now to measure the effect directly (results appended below /
in STATUS.md once done, rather than assumed).

**Validation result: miss rate 0.000 at every λ tested** (same 35 scenarios, same grid,
only the corridor model changed) — collapsed from 0.057–0.086 before the fix. "At-fault
collision" and "any collision" also hit exactly 0. The looser `TTC<1.5s` definition still
has a small nonzero floor (0.029–0.057), which is expected: a wider TTC threshold catches
genuine brief near-misses no corridor model will fully anticipate.

**LTT still returns `lam_hat=None` despite zero observed misses — this is correct, not a
bug.** At n=35, Hoeffding–Bentkus gives p≈0.166 for zero events at α=0.05 (> δ=0.1), so LTT
correctly refuses to certify: 35 scenarios is not enough evidence at this confidence level,
regardless of how clean the result looks. CRC (expectation guarantee, less conservative)
DOES certify at λ=2.0. This is exactly why the pre-registered study uses n≈2 000
(`notes/falsification.md`) — this 35-scenario run is a diagnostic sample, not a substitute.

## 15. T1/T3 baselines implemented; checkpoint switched; H3 data resolved (2026-09-23)

**Checkpoint switched to `av2_cpu_v2/epoch03-minADE1.092`** (was `av2_cpu_v1/epoch08`,
minADE 1.349). Verified before switching: same architecture (zero missing/unexpected
state_dict keys), same past/future length (21/60), predicts correctly through the adapter —
a drop-in swap, not a new integration. `av2_cpu_v2` is Paper 01's own LR-decay fine-tune of
`av2_cpu_v1`'s best checkpoint (`run/train_cpu_v2.cmd`).

**Correction: the "nuScenes clips are ~2.5s, too short for closed loop" note (§4, §7) was
wrong.** They are 81 steps @ 0.1s = 8.1s — comparable to AV2's ~11s, workable for the MRM
(a stop from 11 m/s at 4.0 m/s² takes 2.75s). No new conversion was needed: Paper 01 had
already produced a larger, chunked conversion (`convert_ns_chunked.py`, val_0/1/2, 9041
scenarios total) that this session hadn't seen before. Merged the three chunks into a single
database via `scenarionet.merge` (copy-free — only summary/mapping pkls, no scenario data
duplicated) at `data/ns_val_merged`, wired in as `simenv.DBS["ns_val"]`. Verified: loads,
9041 scenarios, replays correctly. **H3's data blocker is resolved.**

**T1 and T3 baseline triggers implemented** (`src/rollout.py`):
- Refactored trigger scoring so the predictor is called ONCE per decision tick
  (`predict_agents`) and every score is computed from that one call
  (`SCORE_FNS = {"geom": score_geometric, "conf": score_confidence}`) — `score_geometric` is
  the existing T3/T4 family unchanged; `score_confidence` is new: T1's
  "1 − max mode probability", restricted to the same nearby-vehicle set the predictor
  already scores. `rollout()`/`sweep_scenario()`/`run_sweep.py --score {geom,conf}` select
  which one drives the physical fallback — unlike the harm definition, the active trigger
  changes the rollout itself, so T1 vs T3/T4 needs its own set of rollouts, not post-hoc
  rescoring.
- T3's threshold is a plain split-conformal quantile (Vovk et al. 2005, reusing
  `src/conformal.py`) of calibration peak scores under the `score_geometric` family — "hand-
  tuned via conformal calibration", explicitly without the closed-loop LTT/CRC guarantee.
  Computed for free from the λ=∞ rollouts already in the sweep, no extra simulation.
- 2 new tests (13/13 passing): `test_confidence_score_ignores_geometry` pins T1's defining
  property precisely — for the stub predictor (uniform, uninformative probabilities), the
  score is the *constant* 1−1/K, unchanged when the same agent is moved 500 m away, proving
  the score is invariant to geometry by construction, not by accident. First version of this
  test asserted the wrong constant (0.0, when 1−1/K≈0.83 is correct — uniform probabilities
  are the LOWEST-confidence case a distribution can report, not the highest); caught and
  fixed before it could hide a real bug.
- `test_score_key_selects_which_trigger_fires` confirms `score_key` actually changes which
  score drives the rollout.

**A real, load-bearing empirical finding from a smoke test (n=20, real AutoBot, not the
stub):** the T1 confidence score was the SAME constant (0.833) across all 15 probe
scenarios, collapsing the λ grid to one point. AutoBot's per-scenario mode-probability
output has essentially no discriminative range in this sample — T1 cannot usefully rank
scenarios by risk here. This is exactly the "hand-tuned confidence threshold is a weak
baseline" result H2 is built to show, not a bug to chase down; whether it holds at n≈2000 is
an open question for the pre-registered run, and if the flatness turns out to trace to a
narrower cause (e.g. the "most conflicting agent" selection always landing on a similarly-
scored agent) that is itself worth reporting, not hiding.

Not yet done: **T2 (ensemble variance across 5 independently-trained AutoBot seeds)**
requires training 4 additional checkpoints — multiple hours of CPU time each, a substantial,
disruptive compute commitment on a machine shared with other active projects. Deliberately
not launched unprompted this session; flagged for a decision rather than silently deferred
or silently started.


## 16. Correction of §15's T1 finding; two bugs found by inspecting the calibration score distribution (2026-09-24)

**§15's claim — "AutoBot's confidence score is a constant, T1 may be nearly useless" — is
RETRACTED.** It was a pipeline bug.

1. *Fabricated confidences.* `AutoBotPredictor.predict_env` gives every agent the network did
   not predict (pedestrians, cyclists, vehicles beyond 60 m or beyond the 16 nearest) a
   constant-velocity forecast with uniform mode probabilities `1/K`. Their "confidence"
   `1 − max p = 1 − 1/6 = 0.833` is a constant of the code, not a model output. The T1 score
   took a maximum over all agents, so any scene containing one pinned it at 0.833 — on 2 000
   calibration scenarios its scenario-peak had 8 distinct values.
2. *Wrong agent-selection rule.* The design (§2) always said "most conflicting agent"; the code
   took the least-confident of ANY agent.

How it was caught: the score distribution of the finished calibration campaign was inspected
descriptively (no outcomes, no test data) and looked implausibly degenerate; Paper 01's
independent inference dump (`pred_probs`, not routed through the adapter) showed real
variation, which located the bug in the adapter. Fix: `predict_env` now exposes
`last_is_net`; `rollout.agent_table` stores `[conf, gap, dist, ahead, net]` per agent per tick;
`evaluate.py` derives T1 (`conf_conflict`) from network-predicted agents only, argmin predicted
gap, plus an `conf_ahead` variant. Regression tests:
`test_t1_uses_most_conflicting_agent_not_any_agent`,
`test_t1_ignores_agents_the_network_did_not_predict`.

Consequence for data already collected: the stored `conf` traces (2 000 `av2_cal`, ~1 140
`av2_test` rows) are unusable. The forced-fire rollouts (all outcomes, ~20 % of cost) are
valid — they depend only on the firing tick — so a **rescoring pass** (`campaign.py
--rescore`) re-runs only the reference run (~50 s/scenario, MC dropout off) and writes
per-agent-table sidecars, merged by seed when the tick count matches. `geom`, `gap`, `ens`
traces are unaffected.

**Exactness at scale (replaces the "zero divergences" statement).** On 33 761 forced-fire
rollouts of 2 000 scenarios: 538 (1.59 %) showed ≥ 1 cm pre-fire ego drift from the reference,
in 71 scenarios (3.6 %); the pre-fire TTC trace differed in 407 (1.21 %); the pre-fire **harm
status differed in 1 (0.003 %)**. Cause: MetaDrive is not bit-repeatable across independently
run episodes (§10, §11). Handling: headline uses all scenarios; `evaluate.py --drop_diverged`
reports the robustness check.
