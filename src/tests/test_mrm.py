"""
The fallback is UN R157 §5.5.1's minimum-risk manoeuvre: in-lane, deceleration demand
<= 4.0 m/s² (short transients allowed). Own process: MetaDrive allows one engine per process.
Checked with no traffic so a collision impulse can't masquerade as braking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simenv  # noqa: E402

simenv.block_torch()
from rollout import DT, make_fallback_policy_cls  # noqa: E402


def test_mrm_deceleration_within_r157():
    env = simenv.make_env("av2_dev", 0, 30, policy=make_fallback_policy_cls(), no_traffic=True)
    checked = 0
    try:
        for seed in range(30):
            env.reset(seed=seed)
            if simenv.ego_is_static(env):
                continue
            pol = env.engine.get_policy(env.agent.name)
            v = []
            for t in range(70):
                pol.fallback_engaged = t >= 25
                env.step([0.0, 0.0])
                v.append(env.agent.speed)
            if v[24] < 5.0:
                continue
            dec = -np.diff(v[25:]) / DT
            braking = dec[dec > 0.5]
            assert np.median(braking) < 4.3, f"seed {seed}: median {np.median(braking):.2f} m/s^2"
            assert dec.max() < 6.0, f"seed {seed}: transient {dec.max():.2f} m/s^2"
            assert v[-1] < 0.5, f"seed {seed}: did not stop ({v[-1]:.2f} m/s)"
            checked += 1
    finally:
        env.close()
    assert checked >= 10, f"only {checked} scenarios checked"


if __name__ == "__main__":
    test_mrm_deceleration_within_r157()
    print("PASS test_mrm_deceleration_within_r157")
