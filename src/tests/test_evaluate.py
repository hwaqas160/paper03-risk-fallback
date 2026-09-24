"""Synthetic checks for src/evaluate.py (no simulator): lookups are exact, certificates are valid."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

DE = 5


def _outcome(steps, harm_step=None, trigger_step=None, harm_after_trigger=True):
    """A rollout dict. Harm happens at harm_step unless the fallback fired earlier."""
    triggered = trigger_step is not None
    prevented = triggered and harm_step is not None and trigger_step < harm_step and not harm_after_trigger
    coll = [] if harm_step is None or prevented else [dict(step=harm_step, type="stopped_track", at_fault=True)]
    return dict(steps=steps, collision=bool(coll), collision_step=harm_step if coll else None,
                at_fault_step=harm_step if coll else None, collisions=coll, offroad=False,
                min_ttc=99.0, ttc_violations=0, triggered=triggered, trigger_step=trigger_step,
                diverged=False, stopped_steps=0, route_completion=0.5 if triggered else 1.0,
                arrive=not triggered, peak_decel=0.0, ttc_trace=[99.0] * steps)


def make_row(seed, rng, steps=60):
    harmful = rng.random() < 0.3
    h = int(rng.integers(20, 50)) if harmful else None
    n_ticks = len(range(0, steps, DE))
    base = rng.random(n_ticks) * 0.5
    if harmful:                                   # an informative score: rises before harm
        base[max(0, h // DE - 2):] += 1.0 + rng.random()
    traces = {k: list(base) for k in ("geom", "gap", "conf", "geom_route", "geom_iso")}
    ref = _outcome(steps, harm_step=h)
    ref.update(score_traces=traces, feats=[1, 0, 0, 10.0, 8.0, 5.0, 100.0])
    fired = {str(k): _outcome(steps, harm_step=h, trigger_step=k, harm_after_trigger=False)
             for k in range(0, steps, DE)}
    return dict(seed=seed, decide_every=DE, ref=ref, fired=fired, n_diverged=0)


def test_tick_lookup_exact():
    rng = np.random.default_rng(0)
    row = make_row(0, rng)
    row["ref"]["score_traces"]["geom"] = [0.1, 0.5, 0.3] + [0.0] * 9
    s = ev.Scenario(row)
    assert s.tick("geom", 0.2)[0] == 1          # first tick with running max > 0.2
    assert s.tick("geom", 0.6)[0] == s.n_ticks  # never
    assert s.tick("geom", -np.inf)[0] == 0      # always, immediately
    assert s.tick("geom", 0.2, lat=2)[0] == 3   # latency shifts in ticks


def test_never_and_always_bounds():
    rng = np.random.default_rng(1)
    scns = [ev.Scenario(make_row(i, rng)) for i in range(400)]
    never = ev.summarize(scns, "geom", np.inf)
    always = ev.summarize(scns, "geom", -np.inf)
    assert never["unnecessary_stop"] == 0.0 and never["autonomy"] == 1.0
    assert always["miss"] == 0.0 and always["autonomy"] == 0.0
    assert abs(never["miss"] - np.mean([s.ref_harmful for s in scns])) < 1e-12


def test_ltt_certificate_valid_over_resplits():
    rng = np.random.default_rng(2)
    scns = [ev.Scenario(make_row(i, rng)) for i in range(1200)]
    v = ev.validity(scns, alpha=0.05, delta=0.10, reps=60, seed=3)
    assert v["T4 LTT (ours)"]["violation_freq"] <= 0.10 + 0.05, v["T4 LTT (ours)"]


def test_online_cdt_tracks_alpha():
    rng = np.random.default_rng(4)
    test = [ev.Scenario(make_row(i, rng)) for i in range(800)]
    r = ev.online(test, "geom", lam0=5.0, eta=0.5, alpha=0.05, order=np.arange(len(test)))
    assert r["miss"] < 0.15, r      # starts far too permissive; the update must pull it down


def _row_with_agents(seed, tabs):
    rng = np.random.default_rng(seed)
    row = make_row(seed, rng, steps=5 * len(tabs))
    row["ref"]["score_traces"] = {k: [0.1] * len(tabs) for k in ("geom", "gap", "conf")}
    if tabs is not None:
        row["ref"]["agent_tables"] = tabs
    return row


def test_t1_uses_most_conflicting_agent_not_any_agent():
    """
    Regression for the T1 straw man (2026-09-24): the stored 'conf' trace is the LEAST
    confident of ANY agent and saturates at 1-1/K. T1 must be the confidence of the agent
    whose predicted modes come closest to the ego (smallest gap), not the max over agents.
    """
    # rows are [conf, gap, dist, ahead, net]
    tab = [[0.30, 0.5, 30.0, 1, 1],      # confident, modest gap
           [0.83, 20.0, 5.0, 1, 1],      # near-uniform (uncertain), but predicted far from the ego path
           [0.60, -1.0, 20.0, 0, 1]]     # predicted overlap with the ego path (behind), moderate conf
    s = ev.Scenario(_row_with_agents(0, [tab, tab]))
    assert abs(s.traces["conf_conflict"][0] - 0.60) < 1e-9      # argmin gap -> agent 2
    assert abs(s.traces["conf_ahead"][0] - 0.83) < 1e-9         # nearest ahead -> agent 1
    assert abs(s.traces["conf_any"][0] - 0.10) < 1e-9           # stored saturating trace, renamed
    assert "conf" not in s.traces


def test_sidecar_merge_and_mismatch_guard(tmp_path=None):
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    tab = [[0.4, 1.0, 10.0, 1, 1]]
    row = _row_with_agents(1, [tab, tab, tab]); row["ref"].pop("agent_tables")
    bad = _row_with_agents(2, [tab, tab, tab]); bad["ref"].pop("agent_tables")
    nl = chr(10)
    (d / "part_00.jsonl").write_text(json.dumps(row) + nl + json.dumps(bad) + nl)
    (d / "agents_00.jsonl").write_text(
        json.dumps(dict(seed=1, n_ticks=3, agent_tables=[tab] * 3)) + nl +
        json.dumps(dict(seed=2, n_ticks=2, agent_tables=[tab] * 2)) + nl)      # tick count mismatch
    rows = {r["seed"]: r for r in ev.load_rows([d])}
    assert rows[1]["ref"]["agent_tables"], "matching sidecar must be merged"
    assert not rows[2]["ref"].get("agent_tables"), "a sidecar with a different tick count must NOT be used"
    s1, s2 = ev.Scenario(rows[1]), ev.Scenario(rows[2])
    assert "conf_conflict" in s1.traces and "conf_conflict" not in s2.traces
    assert ev.coverage_report([s1, s2])["conf_conflict"] == 1


def test_drop_diverged():
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    a, b = make_row(0, rng), make_row(1, rng)
    b["n_diverged"] = 2
    nl = chr(10)
    (d / "part_00.jsonl").write_text(json.dumps(a) + nl + json.dumps(b) + nl)
    assert len(ev.load_rows([d])) == 2
    assert len(ev.load_rows([d], drop_diverged=True)) == 1


def test_t1_ignores_agents_the_network_did_not_predict():
    """
    Regression for the fabricated-confidence bug (2026-09-24): agents the network did not
    predict (pedestrians, cyclists, far vehicles) carry a constant-velocity forecast with
    FABRICATED uniform probabilities (conf = 1 - 1/K = 0.833). Even when such an agent is the
    most conflicting one, T1 must not read its 'confidence'.
    """
    tab = [[0.35, 4.0, 15.0, 1, 1],      # network-predicted, real confidence 0.35
           [0.8333, -2.0, 6.0, 1, 0]]    # pedestrian: smallest gap, fabricated uniform confidence
    s = ev.Scenario(_row_with_agents(0, [tab, tab]))
    assert abs(s.traces["conf_conflict"][0] - 0.35) < 1e-9, s.traces["conf_conflict"]
    assert abs(s.traces["conf_ahead"][0] - 0.35) < 1e-9
    # a scene with ONLY fallback agents has no model confidence at all -> 0 (nothing to trust)
    only_fake = [[0.8333, -2.0, 6.0, 1, 0]]
    s2 = ev.Scenario(_row_with_agents(1, [only_fake, only_fake]))
    assert s2.traces["conf_conflict"][0] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
