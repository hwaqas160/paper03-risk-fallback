# Headless rollout throughput — Week-2 GATE: **PASSED** (2026-09-21)

## Result: multi-worker sweep (2026-09-21)

Machine idle at launch (2 % CPU, 81 GB free commit); Paper 01's nuScenes `predict.py`
(CPU) and a SupChain2 job started ~1 min in, so W=16/24 were dropped to protect them and
these numbers are *lightly loaded*. Budget 120 s per worker, IDM ego, no trigger scoring.

| workers | rollouts/h | sim steps/s | fps/process | speed-up | rollouts | errors | static skipped |
|---|---|---|---|---|---|---|---|
| 1 | 2 363 | 55 | 80 | 1.0× | 80 | 0 | 37 |
| 4 | 7 962 | 194 | 77 | 3.4× | 268 | 0 | 106 |
| 8 | **17 881** | 428 | 76 | **7.6×** | 599 | 0 | 216 |

- Scaling is near-linear: per-process speed barely drops (80 → 76 fps), so 16 workers
  should give ~35 k rollouts/h on an idle machine (not measured).
- **0 errors in 867 scenarios**: the crosswalk patch in `simenv.py` holds.
- ~27 % of AV2 scenarios have a static ego and are skipped.
- Engine start-up is ~2 s per worker when warm (the 25 s seen on 2026-09-15 was a cold
  disk cache).

### What this means for the experiments

With the trigger scored every step, a rollout costs ~3× the bare sim (pilot: ~17 vs
~45–80 steps/s). So the realistic rate is **~6 k scored rollouts/h at W=8**. The core LTT
experiment (20 λ × 2 000 calibration scenarios + counterfactuals ≈ 42 k rollouts) takes
**~7 h at W=8** or ~3.5 h at W=16. That fits in an overnight run: the scenario count
does **not** need to be cut. Scoring cost is the next thing worth optimising (score every
2–5 ticks; vectorise `ego_plan`).

---

# Earlier partial measurement (2026-09-15, heavily loaded)

Measured 2026-09-15 on the i9-10980XE (18C/36T), `src/bench_throughput.py`,
AV2 `av2_splits/val/test` (5 027 scenarios), IDM closed-loop ego, log-replayed traffic,
rendering OFF.

## Result: single worker

| metric | value |
|---|---|
| rollouts / hour / process | **1 482** |
| scenarios / hour / process (incl. skipped) | 2 130 |
| sim steps / s / process | 34 |
| mean steps per rollout | ~100 (AV2 scenarios are 11 s @ 10 Hz) |
| mean reset | 0.39 s (first reset of a process: ~25 s engine start-up) |
| static-ego scenarios skipped (< 10 m) | **21 / 71 = 30 %** |
| IDM crash rate | 2 % |

Taken while Paper 01's CPU training (~16 threads) and an AI2 job were running, so this is
a *loaded-machine* number, i.e. conservative.

## The real constraint is Windows COMMIT, not cores

The multi-worker sweep could not be completed. At W=4 the workers died with
`ImportError: ... The paging file is too small for this operation to complete`.

- Machine commit limit ≈ 192 GB; other projects (Paper 01 training + AI2) were holding
  ~165 GB, leaving ~16–25 GB.
- A sim worker commits **~2.3 GB**, of which ~0.5 GB is `torch` + CUDA DLLs, pulled in only
  by MetaDrive's unused PPO expert policy (`metadrive.policy.manual_control_policy ->
  examples.ppo_expert -> torch_expert`). `simenv.block_torch()` stubs that module out:
  **~1.8 GB/worker** and no torch loaded.

**This cost real damage:** the W=4 attempt exhausted commit and crashed Paper 01's CPU
training with `DefaultCPUAllocator: not enough memory` at 14:53:50. Its retry wrapper
auto-resumed from `last.ckpt` ~16 s later, losing ~30 min of epoch-9 progress.
`bench_throughput.py` now refuses to start more workers than free commit allows
(`affordable_workers`, 1.8 GB/worker + 6 GB reserve) and uses `ProcessPoolExecutor` so
dead workers raise instead of being respawned forever.

## Extrapolation (to be replaced by the real sweep)

Assuming near-linear scaling on free cores (each worker is single-threaded and CPU-bound):

| workers | rollouts/h | commit needed |
|---|---|---|
| 4 | ~5.9 k | 7.2 GB |
| 8 | ~11.9 k | 14.4 GB |
| 16 | ~23.7 k | 28.8 GB |

Budget check for the core experiment (LTT, `notes/design.md` §3): 20 λ × 2 000 calibration
scenarios + 1 counterfactual each ≈ 42 k rollouts ≈ **3.5 h at W=8**, ~1.8 h at W=16.
That is affordable — so the gate passes *provided memory is available*.

## GATE decision (2026-09-15, superseded above)

The throughput requirement is met per-core; the blocker is machine-level memory
contention with the other projects. Choose one:

1. Run the sweep when Paper 01 / AI2 jobs are idle (cheapest, just needs a window).
2. Raise the Windows page file (commit limit) — system setting, needs a reboot.
3. Cap Paper 03 at W=4–6 permanently and accept ~6 k rollouts/h.

## Todo when the sweep runs

- [ ] W = 1, 4, 8, 16 with `--budget 120`, machine otherwise idle
- [ ] Repeat on nuScenes (reset cost there is ~8 s/scenario — map loading, likely the
      dominant cost; may need `store_map` tuning)
- [ ] Re-measure with the predictor in the loop (or confirm the offline-cache route)
