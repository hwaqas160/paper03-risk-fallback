# RECOVERY — running Paper 03 without a Claude session

Everything below works with **no Claude session**. A Windows scheduled task (`P03_Pipeline`) runs
`src/pipeline.py tick` every 15 minutes, and the same script is also launched at every logon from your
Startup folder. A tick is short, idempotent, and takes a lock so two ticks never overlap.

## What it does by itself
| Situation | What happens |
|---|---|
| Data collection stopped (crash, reboot, ^C, out of memory) | next tick relaunches `run/campaign.cmd`; collection is resumable (one fsynced line per scenario), nothing is re-simulated |
| Processes alive but no data written for 3 h (hang) | tick kills the campaign tree and relaunches |
| Repeated restarts with no progress | restarts slow to once per 3 h (still keeps trying, never gives up) |
| An analysis stage fails | retried after 30 min, 1 h, 2 h, 4 h, then every 6 h forever; the traceback is in `results/pipeline/logs/` |
| New data arrives after an analysis ran | that analysis re-runs automatically (it is keyed on the row counts of its inputs) |
| Everything finished | `results/pipeline/ALL_DONE` is written and `results/final/FINAL_SUMMARY.md` holds every pre-registered verdict, computed mechanically |

How the campaign is kept alive: the supervisor (re)defines a trigger-less, no-time-limit task `P03_Campaign` and starts it with
`schtasks /run`, so the simulation runs as that task's process tree and outlives the short tick that launched it. While an analysis
stage is running (usually 5-20 min, only when its input data changed) the tick holds its lock, so a campaign crash in that window is
repaired at the next tick after the stage ends.

## Look at it (30 seconds)
```
python src\pipeline.py status        # same as opening results\pipeline\STATE.md
python src\pipeline.py tail 30       # last 30 journal events
python src\campaign_status.py        # rows per data arm
```
Files: `results/pipeline/STATE.md` (human summary + ETA), `results/pipeline/journal.jsonl` (every action, append-only),
`results/pipeline/logs/*.log` (each analysis run), `results/campaign/campaign.log` (simulator output),
`results/final/FINAL_SUMMARY.md` (verdicts and numbers), `results/pipeline/tick_crash.log` (if the supervisor itself crashed).

Interpreter: `F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe` (Paper 01's environment; never `pip install` into it).

## What it does NOT survive (be aware)
* **Signing out of Windows.** A task registered without your password only runs while you are logged in
  (locking the screen, disconnecting a remote session, or closing the lid with sleep disabled are all fine).
  Sleep and hibernate are already disabled on this machine. After a reboot it resumes as soon as you log in.
* To make it run **while logged out**, run this once yourself in a normal PowerShell (it asks for your Windows
  password; Claude never sees it):
  `schtasks /change /tn P03_Pipeline /ru %USERNAME% /rp *`
  If Windows refuses (policy), keep the session logged in instead.
* Power loss stops everything; recovery is automatic at the next logon.
* If the PC is shared with Paper 01 / AI2 jobs the collection slows down (memory guard reduces workers). That is expected.

## Manual controls
```
python src\pipeline.py tick            # one supervision pass now
python src\pipeline.py run eval_waymo  # force a stage
python src\pipeline.py reset eval_waymo  # clear its failure/backoff state
python src\pipeline.py install         # (re)register the scheduled task + Startup launcher
python src\pipeline.py uninstall       # remove them
schtasks /query /tn P03_Pipeline       # is the task there?
```
**Never delete `results\campaign\*`** (it is the only copy of hours of simulation). To stop everything:
`python src\pipeline.py uninstall`, then end `python.exe` processes whose command line contains `campaign.py`.

## Stages (in `src/pipeline.py`, `STAGES`)
1. `campaign` (supervised, not a stage) — all data arms in `src/campaign_status.py` (`ARMS`).
2. `eval_main` — waits for `av2_cal`, `av2_test`, `ns_val`: `evaluate.py` (+ `--drop_diverged`), `costs_and_luo.py`.
3. `eval_waymo` — waits for `waymo_val` (1000 scenarios): the same with Waymo as a target, plus the
   **pre-registered** recalibration study `recalibration_study.py` (Amendment 4B, verdicts R1-R4) and its nuScenes
   exploratory replicate.
4. `eval_sensitivity` — waits for the replay / 2 m/s^2 / 10 Hz arms: `sensitivity.py` (paired by seed).
5. `final_report.py` runs every tick and rewrites `results/final/FINAL_SUMMARY.md`.

## When collection and analysis are finished (what still needs a human or Claude)
Open `results/final/FINAL_SUMMARY.md`. Then: (1) paste the Waymo / sensitivity numbers into `paper/main.tex` (search
`[pending]` and `[Waymo`), (2) recompile (`pdflatex` twice), (3) copy the verdicts into the Outcome log of
`notes/falsification.md`, (4) commit. No experiment needs to be re-run for this.

## Pre-registration integrity
`notes/falsification.md` (Amendments 1-4) fixes every hypothesis and refutation rule before the data existed.
`final_report.py` applies them mechanically. Do not change a rule after reading results; add a dated, labelled
post-hoc amendment instead.
