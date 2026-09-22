# Falsification note — PROPOSAL for the author to edit and commit

Draft only. `notes/falsification.md` stays empty until **you** fix these numbers; the
programme's one rule is that the refutation thresholds are committed before results exist.
Every number below is a proposal with its reasoning, not a decision.

Two things must be settled first, because they change what α is even reachable:

1. **Harm base rate.** With the stub predictor and TTC < 1.5 s counted instantaneously,
   ~55 % of pilot scenarios contain a "harmful" event. If over half the scenarios are
   harmful by definition, a risk target of α = 0.1 is unreachable no matter how good the
   trigger is. Tighten the definition first — proposal: a harmful event is a **collision**,
   or a **sustained** TTC < 1.0 s (≥ 3 consecutive decision ticks, i.e. 0.3 s). Then
   re-measure the base rate on ~200 scenarios and pick α from it.
2. **n and power.** At n = 2 000 calibration scenarios a risk estimate has a ±1.3 pp
   95 % CI, so differences below ~3 pp are not resolvable. Any threshold below that is
   untestable — see H2.

---

## H1 — open-loop metrics mislead

- **Claim.** Ranking the four triggers (T1 fixed confidence, T2 ensemble variance,
  T3 conformal region size, T4 risk-calibrated) by open-loop coverage/efficiency gives a
  different ordering than ranking them by closed-loop safety outcome.
- **Metric.** Kendall's τ between the two rankings, computed per dataset setting
  (AV2→AV2, AV2→nuScenes), with a bootstrap CI over scenarios.
- **Refuted if:** τ ≥ 0.8 **and** the top-ranked method is the same under both metrics, in
  every dataset setting. (Rationale: 4–5 arms make τ coarse; requiring both a high τ and
  agreement on the winner avoids declaring victory on ranking noise among tied arms.)

## H2 — calibrated triggering wins

- **Claim.** T4 achieves a better safety / over-conservatism trade-off than the best
  hand-tuned fixed threshold.
- **Metric.** Unnecessary-stop rate on held-out test scenarios **at matched miss rate**
  (both arms constrained to realised miss rate ≤ α), plus whether each arm's realised miss
  rate actually respects α.
- **Refuted if:** the best of T1–T3, tuned on the calibration split, reaches an
  unnecessary-stop rate within **3 pp** of T4 (the resolution limit at n = 2 000) in at
  least 2 of 3 settings, **or** T4 violates its own α on test data in more than δ of
  20 calibration reseeds.
- Honest note: T1–T3 must be tuned properly (swept on the calibration split, evaluated on
  test). A weak baseline here would make the paper unpublishable at T-ITS/T-IV.

## H3 — the arc closes

- **Claim.** Paper 01's coverage collapse under cross-dataset shift produces measurably
  worse closed-loop outcomes.
- **Metric.** λ̂ calibrated on AV2, deployed on nuScenes: realised closed-loop miss rate,
  against α; and the paired change in miss rate AV2→nuScenes.
- **Refuted if:** the realised miss rate under shift stays **≤ α + 3 pp** even though
  open-loop conformal coverage drops by ≥ 5 pp (i.e. the statistical collapse Paper 01
  reports does not propagate to control), **or** the miss-rate increase's 95 % CI includes
  zero.
- Prerequisite: nuScenes full-log scenarios (~20 s), which do not exist yet — the converted
  prediction-challenge snippets are ~2.5 s and cannot support a fallback decision.

---

## Pre-registration checklist before any of this runs

- [ ] Harm definition fixed (collision + sustained TTC), base rate measured
- [ ] α, δ chosen from that base rate, not from habit
- [ ] Calibration / test scenario split fixed by scenario-ID hash (reuse Paper 01's
      `split_db.py` so the splits are the same objects across papers)
- [ ] Probe set for the λ grid held out from both
- [ ] Trigger latency and decision rate fixed (Exp 7 sweeps them; the headline number needs
      one committed setting)

---

## Addendum 2026-09-21 — data that arrived AFTER the proposal above was drafted

Read this before fixing any number. It is appended rather than edited in, so the record
shows which thresholds were proposed blind and which were proposed after seeing data.

### 1. Traffic must be reactive — and that changes every base rate

On 1 000 **dev** scenarios (AV2 `val/train`, disjoint from cal/test), IDM ego, no fallback
(`src/harm_base_rate.py`, `results/harm_traces_av2_dev*.summary.json`):

| harm definition | log replay | **reactive traffic** |
|---|---|---|
| collision only | 0.111 | **0.037** |
| collision or TTC < 1.5 s (instant) | 0.348 | 0.244 |
| collision or TTC < 1.0 s (instant) | 0.268 | 0.176 |
| collision or TTC < 1.0 s × 3 steps | 0.214 | **0.122** |
| collision or TTC < 1.0 s × 5 steps | 0.168 | 0.078 |
| collision or TTC < 0.5 s (instant) | 0.170 | 0.082 |
| collision or TTC < 0.5 s × 3 steps | 0.139 | 0.055 |
| share of harm starting in the first 1 s | 11–20 % | 3–6 % |

Log replay triples collisions because logged agents cannot react to an ego that deviates —
exactly the behaviour a fallback produces. Reactive traffic is now the default.

### 2. How α and the harm definition constrain each other

- **α must be BELOW the no-trigger base rate**, or "never trigger" already satisfies the
  guarantee and the problem is trivial. E.g. with "collision or TTC<1 s × 3" (base 12.2 %),
  α = 0.05 asks the trigger to pre-empt ~60 % of harm — a real task. With "collision only"
  (3.7 %), α = 0.05 is met by doing nothing.
- **α must be ABOVE the unpreventable floor** (harm beginning in the first second:
  ~0.5 % of scenarios under reactive traffic) — no longer a binding constraint.
- LTT needs n ≳ 22 scenarios at zero observed loss to certify α = 0.1 at δ = 0.1, and far
  more near the boundary; n = 2 000 gives room.

### 3. Paper 01's coverage drop is ~3–4 pp — H3's "≥ 5 pp" premise does not hold

AV2 → nuScenes, same AV2 calibration (`paper01-coverage-transfer/results/coverage_transfer.json`,
seed 0): 95 % → 92.2 %, 90 % → 86.0 %, 80 % → 76.8 %. The H3 refutation clause above
("…even though open-loop coverage drops by ≥ 5 pp") can therefore never fire as written.
Suggest restating H3 purely on the closed-loop outcome, e.g. **refuted if the realised miss
rate on nuScenes, at the λ̂ certified on AV2, has a 95 % CI that includes α** — and note
explicitly in the committed note that the ~4 pp open-loop number was already known when
the threshold was set.
