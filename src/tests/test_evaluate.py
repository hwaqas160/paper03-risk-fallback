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
    traces = {k: list(base + (0.0 if k != "conf" else 0.0)) for k in ("geom", "gap", "conf", "geom_route", "geom_iso")}
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
