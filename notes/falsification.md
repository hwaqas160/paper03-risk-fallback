# Falsification Note

> Fill this in BEFORE running any experiment. Commit it. Do not edit after results arrive.
> Copy the hypotheses from PLAN.md section 3 and make them numeric.

**Date written:** 2026-09-22
**Git SHA at time of writing:** none — this folder is not yet a git repository. **Commit this
file before the first `av2_test` or nuScenes evaluation**; until then its only timestamp is
the file system's.
**Written by:** Claude, on the author's explicit instruction ("take the best harm definition
used in top journal papers and keep working"). The author may amend it up to the first commit.

## What had been seen when this was written (full disclosure)

- Harm base rates on 1 000 **dev** scenarios (AV2 val/train, no fallback, reactive traffic):
  nuPlan headline 12.9 % [11.0, 15.1]; at-fault collisions 3.1 %.
- Two pilots with a **stub constant-velocity predictor** (no AutoBot, no trigger comparison):
  plumbing checks only. The earlier of them touched `av2_test` (300 scenarios); no
  hypothesis below was evaluated on it.
- Paper 01's open-loop coverage, AV2 → nuScenes, AV2 calibration: 95 → 92.2 %,
  90 → 86.0 %, 80 → 76.8 % (seed 0).
- **Not seen:** any closed-loop result with a real predictor; any trigger comparison; any
  nuScenes closed-loop run.

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
