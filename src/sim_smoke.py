"""
Week-1 smoke test: replay converted Paper 01 scenarios in MetaDrive, HEADLESS.

Checks that (a) ScenarioEnv loads a ScenarioNet database with rendering disabled,
(b) the ego can be driven by the logged trajectory (ReplayEgoCarPolicy) and by an
IDM closed-loop policy, and (c) per-step wall time is reasonable.

    F:\\CLAUDE\\AI1\\shared\\envs\\unitraj\\Scripts\\python.exe src/sim_smoke.py --db av2_test --n 3
"""
from __future__ import annotations

import argparse
import time

from simenv import DBS, ego_is_static, make_env
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

POLICIES = {"replay": ReplayEgoCarPolicy, "idm": TrajectoryIDMPolicy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="av2_test", choices=list(DBS))
    ap.add_argument("--policy", default="replay", choices=list(POLICIES))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    env = make_env(args.db, args.start, args.n, POLICIES[args.policy])
    try:
        for seed in range(args.start, args.start + args.n):
            t0 = time.perf_counter()
            env.reset(seed=seed)
            t_reset = time.perf_counter() - t0
            steps, info = 0, {}
            t1 = time.perf_counter()
            while True:
                _, _, tm, tc, info = env.step([0.0, 0.0])
                steps += 1
                if tm or tc:
                    break
            dt = time.perf_counter() - t1
            sid = env.engine.data_manager.current_scenario_id
            print(f"[{args.db}/{args.policy}] seed={seed} id={sid} static={ego_is_static(env)} steps={steps} "
                  f"reset={t_reset:.2f}s step_fps={steps / dt:.0f} "
                  f"arrive={info.get('arrive_dest')} crash={info.get('crash')} "
                  f"out_of_road={info.get('out_of_road')} rc={info.get('route_completion', float('nan')):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
