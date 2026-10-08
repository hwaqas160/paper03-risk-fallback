# Revision plan v3 (2026-10-08): strict-review fixes + one new contribution

Source: strict Q1-style review in session (critical C1-C3, major M1-M8, minor m1-m7).
Rule for every new analysis: write the reading rule into notes/falsification.md (Amendment 11) BEFORE running it.
Page budget: main paper must stay at 10 pages (no overlength fee).

## Phase 0 — Pre-register (Amendment 11) — ~1 h
Rules for: T0 physics baseline, noise-corrected validity, CDT/ACI validity, T1 variants, and the new contribution (N-A, N-B, N-C).

## Phase 1 — Fixes that need only stored data — ~1 day
| Item | Review issue | Work |
|---|---|---|
| 1.1 | C3 | T0 baselines by lookup: (a) current-state TTC trigger (score = -TTC from stored `ttc_trace` at decision ticks), (b) constant-velocity geometric trigger if reconstructable; LTT-certified + tuned; add to Table I and Table V |
| 1.2 | C1 | Noise-corrected validity: per resplit, risk of the selected threshold estimated on the full pool (large-sample proxy of population risk) and violation counted as test miss > alpha + 1.96 SE; re-read every verdict near delta (LTT 9.5, latency-aware 12, group 17, Wayformer 7) |
| 1.3 | M3 | CDT and ACI violation over 200 resplits x 20 orderings |
| 1.4 | M2 | T1 variants: mode entropy, probability-weighted spread; reliability (calibration) of mode probabilities on held-out logs |
| 1.5 | m1 | Fix alpha/pi error; full number audit (script that greps every number in main.tex against results/final/*.json) |

## Phase 2 — Text fixes — ~0.5 day
C2 nominal ego policy subsection + scope; M1 reframe "risk score vs uncertainty score", lead with collision-only;
M7 call weighted repair a heuristic; M8 hypothesis ledger table (supplement) + RQ3 before RQ4; m2 shorter abstract;
m3 Observation -> lemma/prose; m7 conclusion ends with a practical rule.

## Phase 3 — NEW CONTRIBUTION (stored data, no new simulation) — ~2-3 days
Motivation (our own finding): held-out miss tracks the target's harm base rate (Spearman 0.90, n = 8).
A marginal miss rate = pi x P(no alert | harm). Covariate reweighting cannot see a change in pi.

N-A  Label-shift-aware certificate ("prevalence-corrected LTT")
  - Estimate the target harm prevalence pi_T WITHOUT target labels by black-box shift estimation (BBSE; Lipton et al.,
    ICML 2018) using the trigger score's confusion structure on source, then certify the class-conditional miss
    P(no alert | harm) <= alpha / pi_T_upper with LTT (pi_T_upper = upper confidence bound of the BBSE estimate).
  - Test on 8 natural shift targets already stored: 6 held-out AV2 cities, nuScenes, Waymo.
  - Pre-registered claim: fewer targets violated than the unweighted and covariate-weighted certificates, at a reported
    stop cost. Refuted otherwise (reported either way).
N-B  Latency-robust certificate, re-read with the corrected validity metric (1.2); formal statement that LTT on the
     shifted loss Phi(tau + l) inherits the guarantee (one-line proof).
N-C  Two-stage outcome audit: choose lambda by alert-level LTT on split A, then report a (1 - delta) upper confidence bound
     on OUTCOME harm (residual + induced) at that fixed lambda on split B. Valid by sample splitting with no monotonicity,
     so it gives the guarantee on the outcome that direct certification could not.
Packaged as: "a certification-and-audit procedure for fallback triggers: alert certificate + prevalence correction +
latency shift + outcome UCB", with the lookup protocol making all four cheap.

## Phase 3b — Proposed NEW METHOD: Counterfactual Safe-Window Certification (CSWC)
Object: from the forced-rollout table, each scenario X gets its safe firing window
  W(X) = { firing ticks j (incl. "never" if the reference run is harmless) whose executed run has no harm }.
  W(X) empty  -> harm unavoidable by this fallback (the "floor"); reported, not counted against the trigger.
Outcome loss  O(X, tau) = 1[tau(X) + latency not in W(X)]  (harm in the run the vehicle actually drives).
Decomposition: O <= LATE + EARLY + GAP, where
  LATE  = 1[tau > last safe tick]   (monotone non-decreasing in lambda)
  EARLY = 1[tau < first safe tick]  (monotone non-increasing in lambda)
  GAP   = 1[tau falls between two safe ticks but is itself unsafe] (measured; bounded separately)
Steps:
  1. Lookup table -> W(X) for every calibration scenario (no new simulation).
  2. Learn a trigger from counterfactual supervision: target = "must fire now" (current tick >= last safe tick - margin),
     features = all stored score traces + scenario covariates; small gradient-boosted model, trained on a separate split.
  3. Certify the OUTCOME loss O directly with split-ordered LTT: order the threshold grid by risk estimated on the
     training split, then run fixed-sequence testing on the calibration split (valid for any pre-specified order, so no
     monotonicity is needed; the learned order restores power). LATE / EARLY / GAP are reported as a diagnostic
     decomposition. Unavoidable floor reported with its own UCB.
     Feasibility check on CALIBRATION data only (2026-10-08): unavoidable (empty window) 0.35 %, non-contiguous windows
     12.7 % -> the GAP term is too large for a pure LATE+EARLY union bound, hence direct certification of O.
  4. Latency: certify on tau + l (guarantee carries over exactly).
  5. Shift: prevalence correction of the avoidable-harm rate by black-box shift estimation (N-A).
  6. Evaluate on: AV2 test, 6 held-out cities, nuScenes, Waymo, second predictor, Wayformer.
Pre-registered claims: (i) CSWC's outcome-level violation <= delta in distribution; (ii) at the same alert-level miss,
CSWC's stops are lower than LTT-geometric (learned trigger); (iii) fewer shift targets violated than unweighted LTT.
Each refuted independently and reported either way.
Novelty check needed: time-to-react / time-to-brake criticality metrics are related; claim novelty only for certifying a
trigger against counterfactual safe windows, not for the window idea itself.
Splits: train = av2_cal (1,994), calibrate = av2_cal5 (904), test = av2_test (1,996) + av2_test5 (904); shift targets as stored.

## Phase 4 — Paper integration — ~1 day
Contribution list rewritten around the new procedure; one consolidated robustness table in the main text (M5);
move ablation table detail to supplement for space; check 10 pages; supplement sections renumbered.

## Phase 5 — Submission hygiene — author + ~0.5 day
Verify every 2025-2026 arXiv reference (M6); publish Zenodo (m6); photos; final read.

## Not planned (cost too high for this cycle)
Learned traffic agents, second simulator, retraining a stronger predictor (machine/GPU limits), collision severity
re-simulation at scale.

Estimated total: ~5-6 working days of my work, almost no new simulation; plus the author's checks.

## Status 2026-10-08 (end of day)
Phases 0-4 done (Amendments 11 and 11b; paper reframed, 10 pages). Phase 5: all eleven arXiv-ID references and the Phil. Trans. R. Soc. A, L4DC 2026, NeurIPS 2025,
ITSC 2025 and Proc. IMechE D 2025 references checked against the arXiv, publisher or proceedings pages (titles and authors match). Headline numbers checked
by src/audit_numbers.py (18 of 18 present in main.tex). Open (author): photos for Rahat and Hina, publish the Zenodo deposit (refresh the code archive first), final read of III-F, V-C, V-D.
