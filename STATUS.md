# Paper 03 — Status

_Updated 2026-09-21 (evening)_

## Where things stand

| Area | State |
|---|---|
| Headless closed-loop sim on Paper 01's scenarios | **working** (AV2 + nuScenes; replay + reactive IDM traffic) |
| Week-2 throughput gate | **PASSED**: 17.9 k rollouts/h at 8 workers, near-linear, 0 errors in 867 scenarios |
| Rollout harness (fallback policy, metrics, counterfactual) | **working + tested** (9/9), `src/rollout.py` |
| Decision-level risk calibration (LTT, CRC) | **working + tested** (4/4), `src/risk_control.py` |
| Parallel λ sweep + post-hoc rescoring under any harm definition | **working**, `src/run_sweep.py` |
| Harm base rates (1 000 dev scenarios, replay vs reactive) | **measured**, `src/harm_base_rate.py` |
| **First certified threshold** (stub predictor, 300 scenarios) | **done** — below |
| Trained predictor from Paper 01 | available: `av2_cpu_v1` epoch 8, minADE₆ 1.349 — **not yet wired in** |
| Paper 01 cross-dataset coverage | **known**: AV2→nuScenes drops 3–4 pp (90 % → 86.0 %) |
| Falsification note | proposal + dated addendum in `notes/falsification_proposal.md`; **yours to commit** |

## Pilot: first certified threshold (STUB predictor — plumbing, NOT a paper result)

300 AV2 **test** scenarios (40-scenario probe held out), reactive traffic, 8 λ + counterfactual
per scenario, 19 min on 8 workers. Harm = collision or TTC < 1 s sustained ≥ 3 steps
(no-trigger base rate 15.3 %). α = δ = 0.1.

| λ | miss rate | autonomy | unnecessary stops | |
|---|---|---|---|---|
| 0.00 | 0.027 | 0.287 | 0.587 | |
| 1.67 | 0.057 | 0.450 | 0.447 | ← **LTT λ̂** (guarantee w.p. ≥ 0.9) |
| 2.12 | 0.070 | 0.520 | 0.380 | |
| 2.45 | 0.093 | 0.673 | 0.253 | ← **CRC λ̂** (guarantee in expectation) |
| 2.52 | 0.110 | 0.757 | 0.183 | |
| 2.60 | 0.120 | 0.840 | 0.113 | |

Reading it:
- **The pipeline certifies.** Calibration → LTT → a threshold with a stated guarantee on the
  closed-loop outcome, end to end, on replayed real scenarios.
- **The price of the guarantee is visible.** At the LTT threshold the stub trigger stops
  unnecessarily in 45 % of scenarios. That is the over-conservatism open-loop evaluation
  cannot see, and the number a real predictor (T1–T4) must drive down. A constant-velocity
  stub is a deliberately weak baseline, so this is the floor to beat, not a finding.
- **LTT vs CRC:** LTT is more conservative (high-probability guarantee + FWER control);
  CRC's λ̂ buys 22 pp more autonomy but only guarantees risk in expectation and needs
  per-scenario monotonicity — which held in **0 %-violation** here.
- **"Collision only" makes the problem trivial:** base rate 3.0 % < α, so "never trigger"
  already satisfies α = 0.1. The definition and α must be chosen together (see addendum).

### Correction

The first pilot (2026-09-15, log replay) reported 5 % of scenarios non-monotone in λ and
I cited that as evidence for LTT over CRC. That run had a bug: the stub predictor drew
different noise for each λ. With common random numbers the rate is **0 %**, so that
evidence doesn't stand. Whether real closed-loop losses are ever non-monotone is now an
open empirical question for the real-predictor runs. LTT remains the safe default because
it doesn't need monotonicity, but "we observed non-monotonicity" isn't a claim we can make.

## Paper 01 changes since 2026-09-15

- CPU training `av2_cpu_v1` finished 2026-09-16; best `epoch08-minADE1.349` (still above
  Paper 01's ~0.85 kill threshold — a Paper 01 decision).
- Coverage, AV2-calibrated, seed 0: in-domain holds (95.0 % at nominal 95 %); on nuScenes
  95 → 92.2 %, 90 → 86.0 %, 80 → 76.8 %.

## Incident log

2026-09-15: my 4-worker benchmark exhausted Windows commit and crashed Paper 01's CPU
training (auto-resumed, ~30 min lost; the run later finished). Workers are now commit-
capped and torch-free.

## Split hygiene

Both stub pilots ran on `av2_test`. No hypothesis was tested and the predictor was a stub,
so no reported result is contaminated — but from now on **development runs use
`av2_dev`** (AV2 val/train); `av2_cal` is for calibration and `av2_test` is touched only for
pre-registered evaluations.

## Checkpoint, H3 data, T1/T3 baselines (2026-09-23)

- **Checkpoint switched** to `av2_cpu_v2/epoch03-minADE1.092` (was `av2_cpu_v1/epoch08`,
  1.349) — verified compatible (same arch, zero missing/unexpected keys) before switching.
- **H3's data blocker is gone.** Earlier note that nuScenes clips are "~2.5s, too short"
  was wrong -- they're 8.1s, workable for the MRM. Paper 01 already had a larger converted
  set (9041 scenarios) this session hadn't seen; merged into `data/ns_val_merged`
  (`simenv.DBS["ns_val"]`), verified loads and replays.
- **T1 (fixed confidence) and T3 (plain conformal quantile) baselines implemented and
  tested** (13/13 tests pass). `run_sweep.py --score {geom,conf}` selects the active trigger.
- **A real finding from a small smoke test**: AutoBot's confidence score was the exact same
  constant across all 15 probe scenarios -- T1 may have near-zero discriminative power in
  this setup. Not yet confirmed at scale; flagged honestly rather than either hidden or
  over-claimed.
- **Deliberately not started**: T2 (ensemble variance) needs training 4 more AutoBot
  checkpoints -- multiple hours of CPU each, a real resource commitment on a shared machine.
  Flagging for a decision rather than launching it or silently dropping it.

## Trigger geometry: two bugs found and fixed, validated (2026-09-22)

First AutoBot pilot showed the risk-calibrated trigger couldn't certify anything at
alpha=0.05. Root-caused and fixed in two stages, each validated on a paired 35-scenario
sample (identical scenarios, identical lambda grid, one variable changed at a time):

1. Clearance used vehicle WIDTH only (no LENGTH) -> missed straight-ahead lead vehicles.
   Fixed: path-relative longitudinal/lateral box margins. Collision-only harm went from
   "never certifies" to certifying cleanly.
2. Remaining floor (TTC-only near-misses): ruled out decision latency first (miss rate was
   IDENTICAL at 2 Hz vs 10 Hz decisions, properly paired -- not the cause). Root cause:
   the trigger's ego-motion corridor followed the ROUTE's curvature; the harm metric
   (matching nuPlan's real definition) deliberately does not -- it assumes constant
   heading, precisely so a driver can't get credit for "the plan says I'll steer away."
   Divergence between the two models: up to 30 m at a 3 s horizon. AutoBot's own
   predictions were separately confirmed accurate (0.03 m error) -- not the predictor's fault.
   Fixed: trigger now uses the same route-agnostic ego model as the harm metric.

**Validated result: miss rate 0.000 at every lambda tested**, same 35 scenarios, down from
0.057-0.086. LTT still won't certify at n=35 (correctly -- Hoeffding-Bentkus needs more
evidence than 35 scenarios can give at alpha=0.05, delta=0.1, even at zero observed misses).
CRC does certify. The pre-registered study needs the full n~2000 calibration set.

## Dependency posture (2026-09-22)

The author raised a real risk: if a manuscript's claims lean on an unpublished companion
paper ("Paper 01"), a rejection or absence of that paper undermines this one. Addressed:
- AutoBot checkpoint + converted data = infrastructure (like a pretrained backbone or a
  public dataset), not a cited finding.
- The AV2 → nuScenes coverage-drop number behind H3 is now **computed by this paper's own
  code** (`src/coverage_selfcheck.py` + `src/conformal.py`, copied not imported), from raw
  model prediction dumps. It reproduces closely (94.9% vs 95.0% in-domain at α=0.05) —
  confirms correctness, not dependency.
- See `notes/falsification.md` "A note on dependencies" for the citation policy this implies.

## Needs your decision

1. **Harm definition + α** (blocks every headline number). Data in the addendum of
   `notes/falsification_proposal.md`. The pilot used "collision or TTC<1 s × 3" at α = 0.1
   as a placeholder; α = 0.05 would make the task harder and more meaningful.
2. **Predictor route under reactive traffic** — online AutoBot in the loop (correct) vs
   hybrid/offline cache (faster, approximate). `notes/design.md` §7. Recommend online.
3. **nuScenes full logs** (~20 s scenes) for closed-loop H3; current snippets are ~2.5 s.
4. PLAN.md §13: Paper 02 doesn't exist on disk; continuing Paper 03 is a deliberate choice.

## Next steps

1. Wire AutoBot (`epoch08`) into the loop and measure its per-tick cost on CPU.
2. Implement T1–T3 score variants on the real predictor; rerun this pilot per trigger.
3. Convert nuScenes full logs → first H3 run.
