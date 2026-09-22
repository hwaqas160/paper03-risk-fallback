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
