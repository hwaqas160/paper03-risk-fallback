@echo off
REM Paper 03 data campaign (notes/falsification.md Amendment 1). Resumable: re-running this
REM file continues where it stopped -- finished scenarios are never re-simulated.
REM Runs at BELOW-NORMAL priority so Paper 01 / AI2 jobs sharing this machine keep priority.
REM Stop:   schtasks /end /tn P03_Campaign      Progress: results\campaign\campaign.log
cd /d F:\CLAUDE\AI3\paper03-risk-fallback
set PY=F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe
set LOG=F:\CLAUDE\AI3\paper03-risk-fallback\results\campaign\campaign.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist results\campaign mkdir results\campaign
set W=8

echo [%DATE% %TIME%] campaign start >> "%LOG%"
REM --- main arms (H1-H3) ---
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_cal  --n 2000 --workers %W% --out results\campaign\av2_cal  >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 2000 --workers %W% --out results\campaign\av2_test >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db ns_val   --n 1500 --workers %W% --out results\campaign\ns_val   >> "%LOG%" 2>&1
REM --- sensitivity arms (reported, not used for H1-H3) ---
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 500 --workers %W% --replay        --out results\campaign\av2_test_replay >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 500 --workers %W% --mrm_decel 2.0 --out results\campaign\av2_test_mrm2   >> "%LOG%" 2>&1
start "" /belownormal /wait /b %PY% src\campaign.py --db av2_test --n 300 --workers %W% --decide_every 1 --out results\campaign\av2_test_10hz  >> "%LOG%" 2>&1
echo [%DATE% %TIME%] campaign done >> "%LOG%"
