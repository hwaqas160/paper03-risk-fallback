"""
Paper 01's trained AutoBot, run ONLINE inside the closed loop.

Why online: traffic is IDM-reactive, so other agents' histories depend on what the ego did
(e.g. braking). Predictions precomputed from the log would describe a world that no longer
exists after the first deviation (notes/design.md §7).

How: at each decision tick the adapter assembles a ScenarioNet-style track table from the
SIMULATOR's recorded history (last 2.1 s, 10 Hz) and hands it to UniTraj's own
`process()` + `collate_fn` — the exact code path Paper 01 trained and evaluated with — so
the model sees its training input distribution. The map part of UniTraj's `preprocess()`
depends only on the scenario, so it is computed once per scenario and cached.

Scope limits (documented, not hidden):
  * Paper 01 trained AutoBot on VEHICLES only (config object_type: ['VEHICLE']). Pedestrians
    and cyclists get constant-velocity predictions (no spread) from this adapter.
  * At the first ticks of an episode the history is shorter than 2.1 s; missing frames are
    marked invalid, as UniTraj does for tracks that appear mid-log.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np

P01 = Path(r"F:\CLAUDE\AI1\paper01-coverage-transfer")
# av2_cpu_v2 is a same-architecture LR-decay fine-tune of av2_cpu_v1's best checkpoint
# (Paper 01's run/train_cpu_v2.cmd), minADE6 1.092 vs v1's 1.349. Verified 2026-09-23: loads
# with zero missing/unexpected state_dict keys, same past_len/future_len (21/60), predicts
# correctly through this adapter -- a safe drop-in swap, not a new integration.
DEFAULT_CKPT = P01 / "results" / "ckpts" / "av2_cpu_v2" / "epoch03-minADE1.092.ckpt"
PRED_RADIUS = 60.0        # m; only predict vehicles this close to ego (the rest cannot matter in 3 s)
MAX_CENTER = 16           # cap per tick; nearest first
MC_AGENTS = 4             # T2: MC-dropout passes only for the nearest vehicles

_TYPE = {"SVehicle": "VEHICLE", "Pedestrian": "PEDESTRIAN", "Cyclist": "CYCLIST"}


def _import_bridge():
    src = str(P01 / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import unitraj_bridge  # noqa: E402  (puts UniTraj on sys.path)
    return unitraj_bridge


class AutoBotPredictor:
    def __init__(self, ckpt: str | Path = DEFAULT_CKPT, device: str = "cpu",
                 threads: int = 1, horizon: int = 30, mc: int = 5):
        import torch
        torch.set_num_threads(threads)
        bridge = _import_bridge()
        self.model, self.cfg = bridge.build_model("autobot", str(ckpt), device)
        self.device = device
        self.horizon = horizon
        self.mc = mc                       # MC-dropout passes for T2 (0 = off)
        self.last_spread = {}
        self.past = int(self.cfg["past_len"])
        self.future = int(self.cfg["future_len"])

        cwd = os.getcwd()
        os.chdir(bridge.UNITRAJ_PKG)
        try:
            from unitraj.datasets.autobot_dataset import AutoBotDataset
        finally:
            os.chdir(cwd)
        ds = AutoBotDataset.__new__(AutoBotDataset)   # skip load_data(): we feed samples ourselves
        ds.config = self.cfg
        ds.is_validation = True
        ds.starting_frame = 0
        self.ds = ds
        self._hist: dict[str, list] = {}
        self._types: dict[str, str] = {}
        self._map_infos = None
        self._meta = None
        self._step = 0
        self.last_timing = {}

    # ------------------------------------------------------------------ episode hooks
    def begin_scenario(self, env):
        """Cache the scenario's map features (UniTraj format) and clear the history."""
        sd = env.engine.data_manager.current_scenario
        stub_tracks = {
            "__stub__": dict(type="VEHICLE", state=dict(
                position=np.zeros((self.past + self.future, 3)), length=np.ones(self.past + self.future),
                width=np.ones(self.past + self.future), height=np.ones(self.past + self.future),
                heading=np.zeros(self.past + self.future), velocity=np.zeros((self.past + self.future, 2)),
                valid=np.ones(self.past + self.future)))}
        scen = dict(tracks=stub_tracks, map_features=sd["map_features"], dynamic_map_states={},
                    metadata=dict(sdc_id="__stub__", ts=np.arange(self.past + self.future) * 0.1,
                                  scenario_id=str(sd["id"]), dataset=str(sd["metadata"].get("dataset", "av2"))))
        self.cfg.only_train_on_ego = True
        ret = self.ds.preprocess(copy.deepcopy(scen))
        self._map_infos = ret["map_infos"]
        self._meta = dict(scenario_id=str(sd["id"]), dataset=scen["metadata"]["dataset"],
                          map_center=ret["map_center"])
        self._hist, self._types, self._step = {}, {}, 0
        self.observe(env)

    def observe(self, env):
        """Record every road user's state at the current sim step (call once per step)."""
        for name, o in list(env.engine.get_objects().items()) + [(env.agent.name, env.agent)]:
            if not hasattr(o, "heading_theta") or not hasattr(o, "velocity"):
                continue
            t = _TYPE.get(type(o).__name__, "VEHICLE")
            if o is env.agent:
                t = "VEHICLE"
            self._types[name] = t
            h = self._hist.setdefault(name, [None] * self._step)
            h.append([o.position[0], o.position[1], 0.0, o.LENGTH, o.WIDTH, 1.5,
                      o.heading_theta, o.velocity[0], o.velocity[1], 1.0])
        self._step += 1
        for h in self._hist.values():
            if len(h) < self._step:
                h.append(None)

    # ------------------------------------------------------------------ prediction
    def predict_env(self, env):
        """
        Predictions for every other road user at the current tick.
        Returns preds (A, K, T, 2) world frame, probs (A, K), extents (A, 2) [L, W].
        """
        import time
        import torch

        t0 = time.perf_counter()
        ego_name = env.agent.name
        names = [n for n in self._hist if n != ego_name and self._hist[n][-1] is not None]
        if not names:
            self.last_is_net = np.zeros(0, bool)
            return np.zeros((0, 6, self.horizon, 2)), np.zeros((0, 6)), np.zeros((0, 2))

        order = list(self._hist)                 # includes ego (needed as context)
        T = self.past + self.future
        trajs = np.zeros((len(order), T, 10), np.float32)
        for i, n in enumerate(order):
            h = self._hist[n][-self.past:]
            off = self.past - len(h)
            for j, s in enumerate(h):
                if s is not None:
                    trajs[i, off + j] = s
        types = [self._types[n] for n in order]

        ego = np.array(env.agent.position)
        cur = {n: np.array(self._hist[n][-1][:2]) for n in names}
        veh = sorted((n for n in names if self._types[n] == "VEHICLE" and np.linalg.norm(cur[n] - ego) < PRED_RADIUS),
                     key=lambda n: np.linalg.norm(cur[n] - ego))[:MAX_CENTER]

        from unitraj.datasets.base_dataset import object_type as OT
        info = dict(
            scenario_id=self._meta["scenario_id"], dataset=self._meta["dataset"],
            sdc_track_index=order.index(ego_name), current_time_index=self.past - 1,
            timestamps_seconds=np.arange(T) * 0.1,
            track_infos=dict(object_id=order, object_type=[OT[t] for t in types], trajs=trajs),
            tracks_to_predict=dict(track_index=[order.index(n) for n in veh]),
            map_infos=self._map_infos, map_center=self._meta["map_center"],
        )
        preds, probs, ext = {}, {}, {}
        if veh:
            samples = self.ds.process(info)
            batch = self.ds.collate_fn(samples)
            inp = batch["input_dict"]
            for k, v in list(inp.items()):
                if torch.is_tensor(v):
                    inp[k] = v.to(self.device)
            t1 = time.perf_counter()
            with torch.no_grad():
                out = self.model.predict(batch)
            t2 = time.perf_counter()
            local = out["predicted_trajectory"][..., :2].cpu().numpy()[:, :, :self.horizon]   # (B,K,T,2)
            pp = out["predicted_probability"].cpu().numpy()
            centre = inp["center_objects_world"].cpu().numpy()
            c, s = np.cos(centre[:, 6]), np.sin(centre[:, 6])
            wx = local[..., 0] * c[:, None, None] - local[..., 1] * s[:, None, None] + centre[:, None, None, 0]
            wy = local[..., 0] * s[:, None, None] + local[..., 1] * c[:, None, None] + centre[:, None, None, 1]
            ids = list(np.asarray(inp["center_objects_id"]))
            for b, n in enumerate(ids):
                preds[n], probs[n] = np.stack([wx[b], wy[b]], -1), pp[b]
            self.last_timing = dict(process_s=t1 - t0, model_s=t2 - t1, n_center=len(veh))
            self.last_spread = self._mc_spread(samples, ids) if self.mc > 0 else {}
            self.last_timing["mc_s"] = time.perf_counter() - t2
        else:
            self.last_spread = {}

        # everyone not predicted by AutoBot: constant velocity, one mode replicated
        k = 6
        tt = np.arange(1, self.horizon + 1) * 0.1
        P, Q, E, NET = [], [], [], []
        for n in names:
            s = self._hist[n][-1]
            NET.append(n in preds)
            if n in preds:
                P.append(preds[n]); Q.append(probs[n])
            else:
                cv = np.stack([s[0] + s[7] * tt, s[1] + s[8] * tt], -1)
                P.append(np.repeat(cv[None], k, 0)); Q.append(np.full(k, 1.0 / k))
            E.append([s[3], s[4]])
        self.last_timing["total_s"] = time.perf_counter() - t0
        # Agents the NETWORK did not predict (pedestrians, cyclists, vehicles beyond
        # PRED_RADIUS / MAX_CENTER) get constant-velocity forecasts with FABRICATED uniform mode
        # probabilities (1/K). Those are not model confidences; a confidence trigger must not
        # see them. Found 2026-09-24: they pinned a "confidence" score at 1-1/K in every scenario.
        self.last_is_net = np.asarray(NET, bool)
        return np.stack(P), np.stack(Q), np.asarray(E)

    def _mc_spread(self, samples, ids) -> dict:
        """
        T2 (ensemble variance) via MC dropout (Gal & Ghahramani, 2016): `mc` stochastic passes
        with dropout active, batched into ONE forward call, over the MC_AGENTS nearest vehicles
        (the only ones a fallback trigger acts on). Spread = norm of the across-pass std of the
        probability-weighted mean endpoint at the scoring horizon, in metres. A deep ensemble of
        independently trained checkpoints is the stronger variant and is a separate arm.
        """
        import torch
        k = min(MC_AGENTS, len(samples))
        rep = [samples[i] for i in range(k)] * self.mc
        batch = self.ds.collate_fn(rep)
        inp = batch["input_dict"]
        for key, v in list(inp.items()):
            if torch.is_tensor(v):
                inp[key] = v.to(self.device)
        stoch = [m for m in self.model.modules()
                 if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention))]
        for m in stoch:
            m.train()
        try:
            with torch.no_grad():
                out = self.model.predict(batch)
        finally:
            for m in stoch:
                m.eval()
        traj = out["predicted_trajectory"][..., :2].cpu().numpy()[:, :, self.horizon - 1]   # (kM,K,2)
        pp = out["predicted_probability"].cpu().numpy()                                        # (kM,K)
        endpoint = (pp[..., None] * traj).sum(1) / pp.sum(1, keepdims=True)                   # (kM,2)
        endpoint = endpoint.reshape(self.mc, k, 2)
        spread = np.linalg.norm(endpoint.std(0), axis=-1)                                      # (k,)
        return {ids[i]: float(spread[i]) for i in range(k)}
