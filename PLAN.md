# PAPER 03 — Risk-Calibrated Fallback
### Deciding When an Autonomous Vehicle Should Give Up

**Months 8–12 · closes the arc · most ambitious · DROPPABLE if the year gets tight**

---

## 1. PROBLEM STATEMENT

Every deployed autonomous vehicle has a minimum-risk manoeuvre: pull over, slow down, hand
back to the driver. **When to trigger it is almost universally a hand-tuned threshold on a
confidence score.**

That is precisely the quantity Papers 01 and 02 show to be untrustworthy under distribution
shift. The trigger for the safety behaviour is calibrated on the assumption that the
calibration holds. When the vehicle enters a new city or a snowstorm — exactly when you most
need the fallback — the trigger is operating on a guarantee that has silently expired.

---

## 2. THE GAP

Conformal prediction papers in this space are almost entirely **open-loop**: they compute
prediction regions and stop.

What happens when those regions drive an actual control decision, in closed loop, over a full
scenario, is largely unstudied. And it is where the interesting failure modes live — because
a conservative region makes the vehicle freeze, and **freezing is itself unsafe**. An
open-loop paper cannot see this. It reports wider regions as "safer" when in closed loop they
cause the vehicle to stop in a live lane.

---

## 3. HYPOTHESES

- **H1 (open-loop metrics mislead).** Ranking of uncertainty methods by open-loop coverage
  differs from their ranking by closed-loop safety outcome. *Refuted if* the rankings agree.
- **H2 (calibrated triggering wins).** A fallback trigger with a formal risk guarantee
  achieves a better safety / over-conservatism trade-off than threshold tuning. *Refuted if*
  a well-tuned fixed threshold matches it across scenarios.
- **H3 (the arc closes).** Paper 01's coverage collapse causes measurable downstream control
  failures. *Refuted if* degraded coverage does not translate into worse closed-loop outcomes.

**H3 is the payoff of the whole programme** — it converts Paper 01's statistical finding into
a demonstrated safety consequence.

---

## 4. CONTRIBUTIONS TO CLAIM

1. Fallback triggering formalised as a **selective-prediction** problem with a risk guarantee
   on the *decision*, not merely on the prediction.
2. Closed-loop evaluation of conformal uncertainty on replayed real scenarios — measuring both
   missed interventions **and** the over-conservatism open-loop papers cannot see.
3. Direct demonstration that coverage collapse causes control failures, closing the argument
   across all three papers.

---

## 5. THE CRITICAL COMPUTE CONSTRAINT

> **Run headless. Never render.**

ScenarioNet documents that advanced *rendering* wants better than an RTX 2060. Rendering is
the only part that needs a good GPU. Closed-loop simulation without rendering is **CPU-bound**
— and 18 cores running parallel scenario rollouts is a genuine advantage over most researchers.

Policies and triggers are small MLPs. Nothing here stresses a P2000.

Render only for the handful of qualitative figures in the paper, one scenario at a time.

---

## 6. CODE (already cloned)

- **`code/scenarionet/`** — replays real scenarios from Waymo, nuScenes, nuPlan and Lyft L5 as
  interactive closed-loop environments. **Reuses the exact data pipeline from Paper 01.**
- **`code/metadrive/`** — the underlying lightweight driving simulator.

This reuse is why Paper 03 is scheduled last: by month 8 the data conversion work is already
done and paid for.

---

## 7. WEEK-BY-WEEK PLAN

### Weeks 1–2 (programme months 8–9) · Simulator plumbing
- [x] Install MetaDrive + ScenarioNet, confirm **headless** rollouts run. _(reusing Paper 01's venv)_
- [x] Replay converted Paper 01 scenarios in closed loop. _(AV2 + nuScenes)_
- [x] Benchmark rollout throughput across 18 cores → `notes/sim_throughput.md`. _(17.9 k/h @ W=8)_
- **GATE: PASSED 2026-09-21.** N parallel scenarios per hour measured. If throughput is too low, cut scenario
  count now rather than discovering it in month 11.

### Weeks 3–5 · Baseline fallback policies
- [ ] Implement three triggers: fixed confidence threshold; ensemble variance; conformal
      region size.
- [ ] Define the outcome metric set: collision rate, time-to-collision violations,
      **unnecessary stops**, route completion, comfort.
- **GATE:** all three baselines run end-to-end in closed loop.

### Weeks 6–8 · Risk-calibrated trigger
- [ ] Port `conformal.py` from Paper 01. Formalise triggering as selective prediction with a
      guarantee on the decision.
- [ ] Sweep the risk level; produce risk–coverage curves for *control outcomes*.
- **GATE:** H2 answered.

### Weeks 9–10 · The arc-closing experiment
- [ ] Take the miscalibrated cross-dataset predictors from Paper 01. Run them in closed loop.
- [ ] Show coverage collapse → downstream control failures.
- **GATE:** H3 answered. This is the figure that makes the three papers one story.

### Weeks 11–14 · Write and submit
- [ ] Include qualitative rendered cases — this is when you turn rendering on.
- [ ] arXiv on submission day.

---

## 8. EXPERIMENTS

| # | Experiment | Proves |
|---|---|---|
| 1 | Open-loop coverage vs closed-loop safety ranking | H1 — open-loop metrics mislead |
| 2 | Three baseline triggers vs risk-calibrated trigger | H2 — the core claim |
| 3 | Risk–coverage curves over control outcomes | The trade-off is explicit, not hidden |
| 4 | Over-conservatism analysis (unnecessary stops) | The failure mode open-loop work cannot see |
| 5 | Miscalibrated predictors in closed loop | H3 — closes the arc |
| 6 | Cross-scenario-source transfer (nuScenes → Waymo scenarios) | Not one-dataset-specific |
| 7 | Sensitivity to trigger latency | Realism — decisions are not instantaneous |
| 8 | Qualitative cases incl. a failure | Required at these venues |

---

## 9. TARGET JOURNALS

| Priority | Venue | IF | Quartile | Cost |
|---|---|---|---|---|
| 1 | **IEEE T-ITS** | 9.1 | Q1 | Free traditional route |
| 2 | **IEEE T-IV** | 14.3 | Q1 | Free traditional route |
| 3 | Transportation Research Part C | — | Q1 | Free subscription route |
| 4 | IEEE RA-L | 5.3 | Q2 | Free (6 pages) |

---

## 10. TARGET LABS

| Lab | Institution | Why |
|---|---|---|
| [xLAB — Safe Autonomous Systems](https://xlab.upenn.edu/) | UPenn 🇺🇸 | Closed-loop safety guarantees is exactly their agenda |
| [Stanford ASL](https://stanfordasl.github.io/) | Stanford 🇺🇸 | Pavone: risk-sensitive planning. See also [CARS](https://cars.stanford.edu/) and [CAESAR](https://caesar.stanford.edu/) |
| [AMRL](https://amrl.cs.utexas.edu/) | UT Austin 🇺🇸 | Biswas: failure-aware autonomy |

---

## 11. RISKS

| Risk | Severity | Mitigation |
|---|---|---|
| Closed-loop work always overruns — simulator plumbing eats weeks budgeted for science | **High** | Hard gate at week 2. If throughput is inadequate, cut scenario count immediately. |
| Simulator realism challenged by reviewers | Medium | Use **replayed real scenarios**, not synthetic ones — that is why ScenarioNet was chosen |
| Paper 01 and 02 overrun, leaving no time | **High** | **This paper is droppable.** Papers 01 and 02 stand alone. Start only once 02 is submitted. |
| Rendering temptation | Low | Headless. Render only final figures. |

---

## 12. REFERENCE PAPERS (downloaded in `papers/`)

| File | Why it matters |
|---|---|
| `ScenarioNet_NeurIPS2023.pdf` | Your simulation platform. **Read first.** |
| `TaskRelevant_failure_detection.pdf` | Failure detection for trajectory predictors — closest prior framing |
| `SafePath_conformal_navigation.pdf` | Conformal prediction driving navigation decisions |
| `RiskSensitive_CVaR_safety.pdf` | Risk-sensitive safety analysis — the formal vocabulary |
| `RobustCP_environments_shift.pdf` | Robust CP under shift; shared with Paper 01 |

---

## 13. DECISION RULE

If at month 10 Paper 02 is not yet submitted, **abandon Paper 03 without regret.**

Two strong papers with released code beat three thin ones. Paper 03 is upside, not the
foundation. The programme succeeds on Papers 01 and 02.
