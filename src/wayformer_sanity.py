"""Amendment 10, W1 pre-check: do AutoBot and the Wayformer wrapper give comparable forecasts on identical simulator states?"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["OMP_NUM_THREADS"] = "2"
import simenv as se
from autobot_predictor import AutoBotPredictor, DEFAULT_CKPT
from rollout import make_fallback_policy_cls

WAY = r"F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_wayformer\epoch19-minADE0.967.ckpt"
pa = AutoBotPredictor(threads=2, mc=0, ckpt=DEFAULT_CKPT)
pw = AutoBotPredictor(threads=2, mc=0, ckpt=WAY, method="wayformer")
print("wayformer cfg: past", pw.past, "future", pw.future, "modes", pw.cfg.get("num_modes"), flush=True)
env = se.make_env("av2_test", 0, 400, policy=make_fallback_policy_cls(4.0), reactive_traffic=True)
diffs, n_sc, ade_cv = [], 0, []
for seed in range(0, 400):
    if n_sc >= 12:
        break
    env.reset(seed=seed)
    if se.ego_is_static(env):
        continue
    pa.begin_scenario(env); pw.begin_scenario(env)
    for _ in range(40):
        env.step([0.0, 0.0]); pa.observe(env); pw.observe(env)
    A = pa.predict_env(env); W = pw.predict_env(env)
    net = pa.last_is_net & pw.last_is_net
    if not net.any():
        continue
    n_sc += 1
    def endpoint(P, Q):
        q = Q / Q.sum(1, keepdims=True)
        return (q[:, :, None] * P[:, :, 29]).sum(1)
    ea, ew = endpoint(A[0], A[1]), endpoint(W[0], W[1])
    d = np.linalg.norm(ea - ew, axis=1)[net]
    diffs += d.tolist()
    # spread across modes as a sanity check on mode diversity
    print(f"seed {seed}: {int(net.sum())} agents, median 3 s endpoint disagreement {np.median(d):.2f} m", flush=True)
diffs = np.array(diffs)
out = dict(n_scenarios=n_sc, n_agents=len(diffs), median_m=float(np.median(diffs)), q75_m=float(np.quantile(diffs, .75)),
           q90_m=float(np.quantile(diffs, .9)), passes_5m_rule=bool(np.median(diffs) < 5.0))
Path("results/final/wayformer_sanity.json").write_text(json.dumps(out, indent=1))
print(out)
