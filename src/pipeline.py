"""
Unattended pipeline supervisor for Paper 03. Runs with NO Claude session: a Windows scheduled task
calls `pipeline.py tick` every 15 minutes. A tick is short, idempotent and safe to interrupt.

  tick     one supervision pass (this is what the scheduler runs)
  status   print results/pipeline/STATE.md
  tail N   last N journal events
  run STG  force one stage to run now (ignores backoff)
  reset STG  clear a stage's failure state
  install  register scheduled task P03_Pipeline + a Startup-folder launcher (no admin needed)
  uninstall

What a tick does
  1. takes a lock (a second tick exits at once);
  2. campaign: if data collection is incomplete and no campaign process is alive, relaunch
     run/campaign.cmd (resumable, one fsynced line per scenario). If processes are alive but NO data
     file has grown for HANG_HOURS, kills the tree and relaunches;
  3. analysis stages (eval, Waymo recalibration, sensitivity), each started only when its input data
     is complete, re-run only if the input changed, retried with exponential backoff on failure;
  4. rewrites results/final/FINAL_SUMMARY.md, results/pipeline/STATE.md, appends journal.jsonl.
Everything it does is in results/pipeline/journal.jsonl; nothing is silent.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PDIR = ROOT / "results" / "pipeline"
LOGS = PDIR / "logs"
FINAL = ROOT / "results" / "final"
JOURNAL = PDIR / "journal.jsonl"
STATE = PDIR / "state.json"
STATEMD = PDIR / "STATE.md"
LOCK = PDIR / "tick.lock"
ALLDONE = PDIR / "ALL_DONE"

PY = os.environ.get("P03_PY", r"F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe")
if not Path(PY).exists():
    PY = sys.executable

HANG_HOURS = 3.0
BACKOFF_BASE_S = 30 * 60
BACKOFF_MAX_S = 6 * 3600
STAGE_TIMEOUT_S = 3 * 3600
CAMPAIGN_RESTART_BACKOFF_S = 3 * 3600   # after >= 6 consecutive restarts with no data growth
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000
DETACHED = 0x00000008 | 0x00000200

CAL, TEST, NS, WAY = "results/campaign/av2_cal", "results/campaign/av2_test", "results/campaign/ns_val", "results/campaign/waymo_val"
PRED = "results/preds/av2cal_from_av2_cpu_v2.npz"


# --------------------------------------------------------------------------- journal / state
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def journal(event: str, stage: str = "", msg: str = "", **kw):
    PDIR.mkdir(parents=True, exist_ok=True)
    rec = dict(ts=now_iso(), event=event, stage=stage, msg=msg, **kw)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return dict(stages={}, campaign=dict(restarts_without_growth=0, last_growth_ts=None, last_rows=None, history=[]))


def save_state(st: dict):
    PDIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    os.replace(tmp, STATE)


def backoff_s(attempts: int) -> float:
    """Wait after the n-th consecutive failure: 30 min, 1 h, 2 h, 4 h, then 6 h forever."""
    return min(BACKOFF_BASE_S * 2 ** max(attempts - 1, 0), BACKOFF_MAX_S)


# --------------------------------------------------------------------------- lock
def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             creationflags=NO_WINDOW, timeout=30).stdout
        return str(pid) in out
    except Exception:
        return False


def acquire_lock(max_age_s: float = 4 * 3600) -> bool:
    PDIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            d = json.loads(LOCK.read_text())
            if pid_alive(d["pid"]) and time.time() - d["t"] < max_age_s:
                return False
        except Exception:
            pass
    LOCK.write_text(json.dumps(dict(pid=os.getpid(), t=time.time())))
    return True


def release_lock():
    try:
        d = json.loads(LOCK.read_text())
        if d.get("pid") == os.getpid():
            LOCK.unlink()
    except Exception:
        pass


# --------------------------------------------------------------------------- processes
def campaign_procs() -> list[dict]:
    """Live processes belonging to the data campaign (campaign.py workers' parents, campaign.cmd)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'campaign\\.py' -or "
          "$_.CommandLine -match 'campaign\\.cmd' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine "
          "| ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             creationflags=NO_WINDOW, timeout=120).stdout.strip()
    except Exception:
        return []
    if not out:
        return []
    data = json.loads(out)
    data = [data] if isinstance(data, dict) else data
    keep = []
    for d in data:
        c = d.get("CommandLine") or ""
        nm = (d.get("Name") or "").lower()
        if nm in ("python.exe", "pythonw.exe") and re.search(r"campaign\.py\s", c + " ") and "campaign_status" not in c:
            keep.append(d)
        elif nm == "cmd.exe" and "campaign.cmd" in c:
            keep.append(d)
    return keep


def heartbeat_ts() -> float:
    """Newest modification time of any data file the campaign writes."""
    newest = 0.0
    base = ROOT / "results" / "campaign"
    for pat in ("*/part_*.jsonl", "*/agents_*.jsonl", "*/skip_*.jsonl"):
        for f in base.glob(pat):
            newest = max(newest, f.stat().st_mtime)
    return newest


def kill_tree(pid: int):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW, timeout=60)


def _ps(script: str, timeout=120) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                       creationflags=NO_WINDOW, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def ensure_campaign_task():
    """(Re)define task P03_Campaign as a trigger-less launcher with no time limit. Only called when no
    campaign process is alive, so it never disturbs a running instance. The campaign runs as THIS task's
    process tree (not as a child of the short-lived tick), so it outlives the tick that started it."""
    cmd = str(ROOT / "run" / "campaign.cmd")
    return _ps(f"""
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c "{cmd}"' -WorkingDirectory '{ROOT}'
$set = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName P03_Campaign -Action $act -Settings $set -Force | Out-Null
'task P03_Campaign defined'
""")


def start_campaign():
    msg = ensure_campaign_task()
    r = subprocess.run(["schtasks", "/run", "/tn", "P03_Campaign"], capture_output=True, text=True,
                       creationflags=NO_WINDOW, timeout=60)
    journal("campaign_task", "campaign", f"{msg} | run: {(r.stdout + r.stderr).strip()[:150]}")


def arm_rows() -> dict:
    from campaign_status import ARMS, arm_state
    return {n: arm_state(n, t) for n, t in ARMS.items()}


def supervise_campaign(st: dict, arms: dict) -> dict:
    total = sum(a["rows"] for a in arms.values())
    complete = all(a["complete"] for a in arms.values())
    c = st["campaign"]
    hist = c.setdefault("history", [])
    hist.append([time.time(), total])
    del hist[:-200]
    if c.get("last_rows") is not None and total > c["last_rows"]:
        c["last_growth_ts"] = time.time()
        c["restarts_without_growth"] = 0
    c["last_rows"] = total
    info = dict(complete=complete, total_rows=total, arms={k: (v["rows"], v["target"]) for k, v in arms.items()})
    if complete:
        if not c.get("complete_logged"):
            journal("campaign_complete", "campaign", "all arms complete", rows=total)
            c["complete_logged"] = True
        return info
    c["complete_logged"] = False
    procs = campaign_procs()
    hb = heartbeat_ts()
    age_h = (time.time() - hb) / 3600 if hb else 1e9
    if procs:
        roots = [p for p in procs if p["ParentProcessId"] not in {q["ProcessId"] for q in procs}]
        if age_h > HANG_HOURS:
            journal("campaign_hang", "campaign", f"processes alive but no data written for {age_h:.1f} h; killing tree",
                    pids=[p["ProcessId"] for p in roots])
            subprocess.run(["schtasks", "/end", "/tn", "P03_Campaign"], capture_output=True,
                           creationflags=NO_WINDOW, timeout=60)
            for p in roots:
                kill_tree(p["ProcessId"])
            time.sleep(5)
            procs = []
        else:
            info["status"] = f"running ({len(procs)} procs, last data {age_h*60:.0f} min ago)"
            return info
    # no live campaign process
    wait = 0.0
    if c.get("restarts_without_growth", 0) >= 6:
        last = c.get("last_start_ts", 0)
        wait = max(0.0, CAMPAIGN_RESTART_BACKOFF_S - (time.time() - last))
    if wait > 0:
        info["status"] = f"restart backoff {wait/60:.0f} min (6+ restarts without progress)"
        return info
    start_campaign()
    c["last_start_ts"] = time.time()
    c["restarts_without_growth"] = c.get("restarts_without_growth", 0) + 1
    journal("campaign_start", "campaign", "no live campaign process; launched run/campaign.cmd",
            rows=total, restarts_without_growth=c["restarts_without_growth"])
    info["status"] = "restarted"
    return info


# --------------------------------------------------------------------------- analysis stages
def _fp(arms: dict, names) -> dict:
    return {n: arms[n]["rows"] for n in names}


STAGES = [
    dict(name="eval_main", needs=["av2_cal", "av2_test", "ns_val"], cmds=[
        ("evaluate", [PY, "src/evaluate.py", "--cal", CAL, "--test", TEST, "--shift", NS, "--pred_err", PRED,
                      "--out", "results/final/eval_main.json"], "results/final/eval_main.txt"),
        ("evaluate_dropdiv", [PY, "src/evaluate.py", "--cal", CAL, "--test", TEST, "--shift", NS, "--pred_err", PRED,
                              "--drop_diverged", "--out", "results/final/eval_main_dropdiv.json"], "results/final/eval_main_dropdiv.txt"),
        ("costs_luo", [PY, "src/costs_and_luo.py", "--cal", CAL, "--test", TEST, "--shift", NS,
                       "--out", "results/final/costs_luo_main.json"], "results/final/costs_luo_main.txt"),
    ]),
    dict(name="eval_waymo", needs=["av2_cal", "av2_test", "ns_val", "waymo_val"], cmds=[
        ("evaluate_all", [PY, "src/evaluate.py", "--cal", CAL, "--test", TEST, "--shift", NS, "--target", f"waymo={WAY}",
                          "--pred_err", PRED, "--out", "results/final/eval_all.json"], "results/final/eval_all.txt"),
        ("evaluate_all_dropdiv", [PY, "src/evaluate.py", "--cal", CAL, "--test", TEST, "--shift", NS, "--target", f"waymo={WAY}",
                                  "--pred_err", PRED, "--drop_diverged", "--out", "results/final/eval_all_dropdiv.json"],
         "results/final/eval_all_dropdiv.txt"),
        ("costs_luo_all", [PY, "src/costs_and_luo.py", "--cal", CAL, "--test", TEST, "--shift", NS, "--target", f"waymo={WAY}",
                           "--out", "results/final/costs_luo_all.json"], "results/final/costs_luo_all.txt"),
        ("recal_waymo", [PY, "src/recalibration_study.py", "--cal", CAL, "--test", TEST, "--target", WAY, "--name", "waymo",
                         "--out", "results/final/recal_waymo.json"], "results/final/recal_waymo.txt"),
        ("recal_nuscenes_exploratory", [PY, "src/recalibration_study.py", "--cal", CAL, "--test", TEST, "--target", NS,
                                        "--name", "nuscenes_exploratory", "--out", "results/final/recal_nuscenes_exploratory.json"],
         "results/final/recal_nuscenes_exploratory.txt"),
    ]),
    dict(name="eval_sensitivity", needs=["av2_test_replay", "av2_test_mrm2", "av2_test_10hz"], cmds=[
        ("sensitivity", [PY, "src/sensitivity.py", "--cal", CAL, "--main", TEST,
                         "--arm", "replay=results/campaign/av2_test_replay", "--arm", "mrm2=results/campaign/av2_test_mrm2",
                         "--arm", "hz10=results/campaign/av2_test_10hz", "--out", "results/final/sensitivity.json"],
         "results/final/sensitivity.txt"),
    ]),
]


def run_cmd(label: str, argv: list, out_txt: str, logf) -> int:
    logf.write(f"\n===== {label}  {now_iso()} =====\n$ {' '.join(argv)}\n")
    logf.flush()
    outp = ROOT / out_txt
    outp.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(outp, "w", encoding="utf-8") as of:
        p = subprocess.run(argv, cwd=str(ROOT), stdout=of, stderr=subprocess.STDOUT, env=env,
                           creationflags=BELOW_NORMAL | NO_WINDOW, timeout=STAGE_TIMEOUT_S)
    tail = "\n".join(l for l in outp.read_text(encoding="utf-8", errors="replace").splitlines()
                     if "cuda" not in l)[-2500:]
    logf.write(tail + f"\n[exit {p.returncode}]\n")
    logf.flush()
    return p.returncode


def run_stage(stg: dict, st: dict, arms: dict, force=False):
    name = stg["name"]
    s = st["stages"].setdefault(name, dict(status="waiting", attempts=0))
    missing = [n for n in stg["needs"] if not arms[n]["complete"]]
    if missing:
        s["status"] = "waiting: " + ", ".join(f"{n} {arms[n]['rows']}/{arms[n]['target']}" for n in missing)
        return
    fp = _fp(arms, stg["needs"])
    if s.get("done_fp") == fp and not force:
        s["status"] = "ok"
        return
    if not force and s.get("next_try", 0) > time.time():
        s["status"] = f"backoff until {datetime.fromtimestamp(s['next_try']):%H:%M} (attempt {s['attempts']})"
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    logp = LOGS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    journal("stage_start", name, "starting", inputs=fp, log=str(logp))
    s["status"] = "running"
    save_state(st)
    rc, err = 0, ""
    with open(logp, "w", encoding="utf-8") as logf:
        for label, argv, out_txt in stg["cmds"]:
            try:
                rc = run_cmd(label, argv, out_txt, logf)
            except subprocess.TimeoutExpired:
                rc, err = 124, f"{label}: timeout after {STAGE_TIMEOUT_S}s"
            except Exception as e:                       # noqa: BLE001
                rc, err = 1, f"{label}: {type(e).__name__}: {e}"
            if rc != 0:
                err = err or f"{label}: exit {rc}"
                break
    if rc == 0:
        s.update(status="ok", attempts=0, done_fp=fp, last_ok=now_iso(), last_error="", next_try=0)
        journal("stage_ok", name, "completed", inputs=fp)
    else:
        s["attempts"] = s.get("attempts", 0) + 1
        s.update(status=f"FAILED (attempt {s['attempts']})", last_error=err, next_try=time.time() + backoff_s(s["attempts"]))
        journal("stage_failed", name, err, attempts=s["attempts"], retry_in_min=backoff_s(s["attempts"]) / 60, log=str(logp))


# --------------------------------------------------------------------------- reporting
def eta_hours(st: dict, arms: dict) -> str:
    hist = st["campaign"].get("history", [])
    remaining = sum(max(a["target"] - a["rows"], 0) for a in arms.values()) + sum(a["need_tables"] for a in arms.values())
    if remaining == 0:
        return "0"
    win = [h for h in hist if h[0] >= time.time() - 6 * 3600]
    if len(win) < 2 or win[-1][1] <= win[0][1]:
        return "unknown (no recent growth)"
    rate = (win[-1][1] - win[0][1]) / ((win[-1][0] - win[0][0]) / 3600)
    return f"{remaining / rate:.1f} h at {rate:.1f} scenarios/h ({remaining} remaining)"


def write_state_md(st: dict, arms: dict, camp: dict):
    L = [f"# Pipeline state ({now_iso()})", "", f"ALL_DONE: {'YES' if ALLDONE.exists() else 'no'}", "",
         "## Campaign (data collection)", "", f"status: {camp.get('status', 'complete' if camp.get('complete') else '?')}",
         f"estimated time to finish collection: {eta_hours(st, arms)}", "",
         "| Arm | Rows | Target | Unmatched | Errors | State |", "|---|---|---|---|---|---|"]
    for n, a in arms.items():
        L.append(f"| {n} | {a['rows']} | {a['target']} | {a['unmatched']} | {a['errors']} | {'COMPLETE' if a['complete'] else 'in progress'} |")
    L += ["", "## Analysis stages", "", "| Stage | Status | Attempts | Last OK | Last error |", "|---|---|---|---|---|"]
    for stg in STAGES:
        s = st["stages"].get(stg["name"], {})
        L.append(f"| {stg['name']} | {s.get('status', 'waiting')} | {s.get('attempts', 0)} | {s.get('last_ok', '-')} | {(s.get('last_error') or '')[:100]} |")
    L += ["", "## Last 12 journal events", ""]
    try:
        for line in JOURNAL.read_text(encoding="utf-8").splitlines()[-12:]:
            r = json.loads(line)
            L.append(f"- {r['ts']} **{r['event']}** {r.get('stage', '')} {r.get('msg', '')}")
    except Exception:
        L.append("- (journal empty)")
    L += ["", "## If something looks wrong", "",
          "- Read this file, `results/pipeline/journal.jsonl`, `results/pipeline/logs/`, `results/campaign/campaign.log`.",
          "- `python src/pipeline.py tick` runs one supervision pass by hand. `python src/pipeline.py run <stage>` forces a stage.",
          "- Data collection is resumable: never delete `results/campaign/*`. See RECOVERY.md."]
    tmp = STATEMD.with_suffix(".tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    os.replace(tmp, STATEMD)


# --------------------------------------------------------------------------- tick
def tick(force_stage: str | None = None) -> int:
    if not acquire_lock():
        print("another tick is running; exiting")
        return 0
    try:
        st = load_state()
        journal("tick", "", "supervision pass")
        arms = arm_rows()
        camp = supervise_campaign(st, arms)
        for stg in STAGES:
            try:
                run_stage(stg, st, arms, force=(force_stage == stg["name"]))
            except Exception as e:                       # noqa: BLE001
                journal("stage_error", stg["name"], f"{type(e).__name__}: {e}")
            save_state(st)
        try:
            subprocess.run([PY, "src/final_report.py"], cwd=str(ROOT), capture_output=True, creationflags=NO_WINDOW, timeout=600)
        except Exception as e:                           # noqa: BLE001
            journal("report_error", "final_report", str(e))
        all_ok = camp.get("complete") and all(st["stages"].get(s["name"], {}).get("status") == "ok" for s in STAGES)
        if all_ok and not ALLDONE.exists():
            ALLDONE.write_text(now_iso())
            journal("all_done", "", "campaign complete and every analysis stage ok; see results/final/FINAL_SUMMARY.md")
        save_state(st)
        write_state_md(st, arms, camp)
        return 0
    finally:
        release_lock()


# --------------------------------------------------------------------------- install
def install():
    pyw = Path(PY).with_name("pythonw.exe")
    exe = str(pyw if pyw.exists() else PY)
    script = str(ROOT / "src" / "pipeline.py")
    ps = f"""
$act = New-ScheduledTaskAction -Execute '{exe}' -Argument '"{script}" tick' -WorkingDirectory '{ROOT}'
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 15)
$set = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName P03_Pipeline -Action $act -Trigger $trg -Settings $set -Force | Out-Null
'registered P03_Pipeline'
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    startup = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if startup.exists():
        (startup / "P03_Pipeline.cmd").write_text(
            f'@echo off\r\nstart "" /min "{exe}" "{script}" tick\r\n', encoding="ascii")
        print("startup launcher:", startup / "P03_Pipeline.cmd")
    journal("install", "", "scheduled task P03_Pipeline + Startup launcher installed")


def uninstall():
    subprocess.run(["schtasks", "/delete", "/tn", "P03_Pipeline", "/f"])
    startup = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "P03_Pipeline.cmd"
    if startup.exists():
        startup.unlink()
    journal("uninstall", "", "removed")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else "tick"
    try:
        if cmd == "tick":
            return tick()
        if cmd == "status":
            print(STATEMD.read_text(encoding="utf-8") if STATEMD.exists() else "no state yet; run: pipeline.py tick")
        elif cmd == "tail":
            n = int(a[1]) if len(a) > 1 else 20
            print("\n".join(JOURNAL.read_text(encoding="utf-8").splitlines()[-n:]))
        elif cmd == "run":
            return tick(force_stage=a[1])
        elif cmd == "reset":
            st = load_state()
            st["stages"].pop(a[1], None)
            save_state(st)
            journal("reset", a[1], "state cleared by hand")
        elif cmd == "install":
            install()
        elif cmd == "uninstall":
            uninstall()
        else:
            print(__doc__)
    except Exception as e:                               # noqa: BLE001
        PDIR.mkdir(parents=True, exist_ok=True)
        import traceback
        (PDIR / "tick_crash.log").open("a", encoding="utf-8").write(f"\n[{now_iso()}] {cmd}\n{traceback.format_exc()}\n")
        journal("tick_crash", "", f"{type(e).__name__}: {e}")
        release_lock()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
