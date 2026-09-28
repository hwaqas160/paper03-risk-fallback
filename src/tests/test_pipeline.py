import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_report  # noqa: E402
import pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_pipeline(tmp_path, monkeypatch):
    p = tmp_path / "pipeline"
    monkeypatch.setattr(pipeline, "PDIR", p)
    monkeypatch.setattr(pipeline, "LOGS", p / "logs")
    monkeypatch.setattr(pipeline, "JOURNAL", p / "journal.jsonl")
    monkeypatch.setattr(pipeline, "STATE", p / "state.json")
    monkeypatch.setattr(pipeline, "STATEMD", p / "STATE.md")
    monkeypatch.setattr(pipeline, "LOCK", p / "tick.lock")
    monkeypatch.setattr(pipeline, "ALLDONE", p / "ALL_DONE")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    return p


def arms(rows_a=10, complete=True):
    return {"a": dict(rows=rows_a, target=10, complete=complete, need_tables=0, unmatched=0, errors=0)}


def stage(cmd_ok=True):
    code = "print(1)" if cmd_ok else "import sys; sys.exit(3)"
    return dict(name="t", needs=["a"], cmds=[("c", [sys.executable, "-c", code], "out/t.txt")])


def test_backoff_is_monotone_and_capped():
    xs = [pipeline.backoff_s(n) for n in range(1, 12)]
    assert xs == sorted(xs) and xs[0] == 1800 and max(xs) == 6 * 3600


def test_lock_blocks_live_holder_and_recovers_stale():
    assert pipeline.acquire_lock()
    assert not pipeline.acquire_lock()                       # same live pid, fresh -> refused
    pipeline.LOCK.write_text(json.dumps(dict(pid=99999999, t=time.time())))
    assert pipeline.acquire_lock()                           # dead pid -> taken over
    pipeline.LOCK.write_text(json.dumps(dict(pid=os.getpid(), t=time.time() - 10 * 3600)))
    assert pipeline.acquire_lock()                           # too old -> taken over
    pipeline.release_lock()
    assert not pipeline.LOCK.exists()


def test_stage_waits_until_inputs_complete():
    st = pipeline.load_state()
    pipeline.run_stage(stage(), st, arms(rows_a=4, complete=False))
    assert st["stages"]["t"]["status"].startswith("waiting")
    assert "done_fp" not in st["stages"]["t"]


def test_stage_runs_once_then_skips_until_inputs_change():
    st = pipeline.load_state()
    pipeline.run_stage(stage(), st, arms())
    assert st["stages"]["t"]["status"] == "ok"
    n_events = len(pipeline.JOURNAL.read_text().splitlines())
    pipeline.run_stage(stage(), st, arms())                  # unchanged inputs -> no new run
    assert len(pipeline.JOURNAL.read_text().splitlines()) == n_events
    pipeline.run_stage(stage(), st, arms(rows_a=11))         # inputs changed -> reruns
    assert len(pipeline.JOURNAL.read_text().splitlines()) > n_events


def test_failed_stage_backs_off_then_retries_and_recovers():
    st = pipeline.load_state()
    pipeline.run_stage(stage(cmd_ok=False), st, arms())
    s = st["stages"]["t"]
    assert s["attempts"] == 1 and s["status"].startswith("FAILED") and s["next_try"] > time.time()
    pipeline.run_stage(stage(cmd_ok=False), st, arms())      # inside backoff -> not retried
    assert s["attempts"] == 1 and s["status"].startswith("backoff")
    s["next_try"] = 0
    pipeline.run_stage(stage(cmd_ok=False), st, arms())      # backoff over -> retried
    assert s["attempts"] == 2
    s["next_try"] = 0
    pipeline.run_stage(stage(cmd_ok=True), st, arms())       # cause fixed -> recovers
    assert s["status"] == "ok" and s["attempts"] == 0


def test_state_file_survives_roundtrip_and_journal_is_json_lines():
    st = pipeline.load_state()
    st["stages"]["x"] = dict(status="ok", attempts=0)
    pipeline.save_state(st)
    assert pipeline.load_state()["stages"]["x"]["status"] == "ok"
    pipeline.journal("unit", "s", "hello")
    assert json.loads(pipeline.JOURNAL.read_text().splitlines()[-1])["event"] == "unit"


def test_final_report_on_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(final_report, "FINAL", tmp_path)
    assert "No evaluation output yet" in final_report.build()
