@echo off
REM Paper 03 data campaign (notes/falsification.md Amendments 1-2). RESUMABLE and IDEMPOTENT:
REM re-running this file continues where it stopped; finished scenarios are never re-simulated
REM and a finished arm returns in seconds. That is what lets P03_Campaign re-launch it every
REM 30 minutes (schtasks default: do not start a new instance while one is running), so an
REM interruption (the 2026-09-24 10:50 stop) heals itself instead of waiting for a human.
REM
REM Runs at BELOW-NORMAL priority so Paper 01 / AI2 jobs sharing this machine keep priority.
REM Stop:  schtasks /delete /tn P03_Campaign /f      Progress: results\campaign\campaign.log
cd /d F:\CLAUDE\AI3\paper03-risk-fallback
set PY=F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe
set LOG=F:\CLAUDE\AI3\paper03-risk-fallback\results\campaign\campaign.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist results\campaign mkdir results\campaign
set W=8

echo [%DATE% %TIME%] campaign (re)start >> "%LOG%"
REM --- 1. finish the main arms with the corrected pipeline (per-agent tables recorded) ---
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 2000 --workers %W% --out results\campaign\av2_test >> "%LOG%" 2>&1
REM --- 2. rescore rows collected before per-agent tables existed (reference run only) ---
REM     --n/--workers MUST match the original collection so seed blocks line up.
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_cal  --n 2000 --workers %W% --rescore --mc 0 --out results\campaign\av2_cal  >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 2000 --workers %W% --rescore --mc 0 --out results\campaign\av2_test >> "%LOG%" 2>&1
REM --- 3. shifted target (H3) ---
start "" /belownormal /wait /b %PY% src\campaign.py --db ns_val   --n 1500 --workers %W% --out results\campaign\ns_val >> "%LOG%" 2>&1
REM --- 3b. second shifted target: Waymo Open Motion (Amendment 3), BEFORE the sensitivity arms ---
start "" /belownormal /wait /b %PY% src\campaign.py --db waymo_val --n 1000 --workers %W% --out results\campaign\waymo_val >> "%LOG%" 2>&1
REM --- 4. sensitivity arms (reported, not used for H1-H3) ---
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 500 --workers %W% --replay        --out results\campaign\av2_test_replay >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 500 --workers %W% --mrm_decel 2.0 --out results\campaign\av2_test_mrm2   >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 100 --workers %W% --decide_every 1 --out results\campaign\av2_test_10hz  >> "%LOG%" 2>&1
REM --- 5. Amendment 5 (outcome-level certification, N1-N5): fresh, disjoint scenario ranges ---
REM     verified against the arms above before being fixed in notes/falsification.md. --mc 0: T1/T2
REM     per-agent tables are not needed by N1-N5. N1-N3, primary predictor, d=4.0 (default):
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_cal  --start 3595 --n 900 --workers %W% --mc 0 --out results\campaign\av2_cal5  >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --start 3584 --n 900 --workers %W% --mc 0 --out results\campaign\av2_test5 >> "%LOG%" 2>&1
REM     N4: same seed ranges, d=2.0 (paired with the d=4.0 rows above):
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_cal  --start 3595 --n 900 --workers %W% --mc 0 --mrm_decel 2.0 --out results\campaign\av2_cal5_d2  >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --start 3584 --n 900 --workers %W% --mc 0 --mrm_decel 2.0 --out results\campaign\av2_test5_d2 >> "%LOG%" 2>&1
REM     N5: same seed ranges again, second predictor (Paper 01 artifact, minADE6 0.854):
set GPUCKPT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_gpu_full\epoch53-minADE0.854.ckpt
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_cal  --start 3595 --n 900 --workers %W% --mc 0 --ckpt "%GPUCKPT%" --out results\campaign\av2_cal5_gpu  >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --start 3584 --n 900 --workers %W% --mc 0 --ckpt "%GPUCKPT%" --out results\campaign\av2_test5_gpu >> "%LOG%" 2>&1
REM     extension pools, not required by N1-N4, lowest priority:
start "" /belownormal /wait /b %PY% src\campaign.py --db ns_val     --start 2624 --n 1500 --workers %W% --mc 0 --out results\campaign\ns_val5    >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db waymo_val2 --start 0    --n 1000 --workers %W% --mc 0 --out results\campaign\waymo_val5 >> "%LOG%" 2>&1
echo [%DATE% %TIME%] campaign pass complete >> "%LOG%"
REM --- self-retire ONLY when every arm is genuinely complete ---
%PY% src\campaign_status.py --quiet && schtasks /delete /tn P03_Campaign /f >> "%LOG%" 2>&1
