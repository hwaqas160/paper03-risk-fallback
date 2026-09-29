"""
Headless ScenarioEnv factory shared by every Paper 03 script.

MetaDrive is imported lazily (inside make_env) so that a sim-only worker can call
block_torch() first. Nothing here imports metadrive at module level — keep it that way.
"""
from __future__ import annotations

import sys

DBS = {
    # dev set: AV2 val "train" split — unused by Paper 03 otherwise, so design choices
    # (harm definition, score form) can be made here without touching cal/test.
    "av2_dev": r"F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\train",
    "av2_cal": r"F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\cal",
    "av2_test": r"F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\test",
    "ns_val0": r"F:\CLAUDE\AI1\paper01-coverage-transfer\data\nuscenes_scenarionet\val\val_0",
    # Merged val_0+val_1+val_2 (9041 scenarios total), built 2026-09-23 via `scenarionet.merge`
    # (copy-free: only dataset_summary.pkl/dataset_mapping.pkl live here, scenario data stays
    # in Paper 01's folder). H3's shifted test set. Correction to an earlier note: these
    # nuScenes prediction-challenge scenarios are 81 steps @ 0.1s = 8.1s, NOT "~2.5s, too
    # short for closed loop" as previously written in notes/design.md -- comparable to AV2's
    # ~11s and workable for the MRM (a stop from 11 m/s at 4.0 m/s^2 takes 2.75s).
    "ns_val": r"F:\CLAUDE\AI3\paper03-risk-fallback\data\ns_val_merged",
    # Waymo Open Motion v1.2.1 validation, converted by src/convert_waymo.py (no TensorFlow).
    # Third dataset: a second, independent shift target. 9.1 s scenarios (1 s history + 8 s).
    "waymo_val": r"F:\CLAUDE\AI3\paper03-risk-fallback\data\waymo_val",
    "waymo_smoke": r"F:\CLAUDE\AI3\paper03-risk-fallback\data\waymo_smoke",
    # Amendment 5: shards 8-29 (the 22 NOT used to build waymo_val), independent scenario indices,
    # so this pool cannot overlap with the already-analysed waymo_val scenarios.
    "waymo_val2": r"F:\CLAUDE\AI3\paper03-risk-fallback\data\waymo_val2",
}

# ScenarioNet's own filter: drop scenarios where the ego moves < 10 m.
MIN_TRACK_M = 10.0


def block_torch():
    """
    Keep a sim-only worker from importing torch, without changing any installed package.

    MUST be called before MetaDrive is imported. Otherwise
    metadrive.policy.manual_control_policy -> examples.ppo_expert -> torch_expert -> torch
    (+ CUDA DLLs) reserves ~2 GB of Windows *commit* per worker for a PPO expert policy we
    never use. With several sim workers that exhausted the machine's commit limit on
    2026-09-15 and crashed a training job in another project.

    Mechanism: pre-register a stub under the name `...ppo_expert.torch_expert`, so the
    `from ... import torch_expert` in ppo_expert/__init__ is served from sys.modules and
    the real file (line 1: `import torch`) never executes. The stub's expert raises if
    anything actually tries to drive with it. A process that runs the predictor must NOT
    call this.
    """
    if "metadrive" in sys.modules:
        raise RuntimeError("block_torch() must run before metadrive is imported")

    import types

    name = "metadrive.examples.ppo_expert.torch_expert"
    stub = types.ModuleType(name)

    def torch_expert(*_a, **_kw):
        raise RuntimeError("PPO torch expert disabled by simenv.block_torch()")

    stub.torch_expert = torch_expert
    sys.modules[name] = stub


def _patch_vertex():
    """
    AV2 crosswalk polygons carry (x, y, z) points; metadrive 0.4.2.3's is_anticlockwise
    unpacks two values and raises ValueError (first hit: av2_test scenario index 31).
    MetaDrive 0.4.3 (code/metadrive) slices [:2]; this is that exact fix, applied without
    touching the venv Paper 01 runs from.
    """
    import metadrive.utils.vertex as vertex

    def is_anticlockwise(points):
        s = 0
        n = len(points)
        for i in range(n):
            x1, y1 = points[i][:2]
            x2, y2 = points[(i + 1) % n][:2]
            s += (x2 - x1) * (y2 + y1)
        return s > 0

    vertex.is_anticlockwise = is_anticlockwise


def make_env(db: str, start: int = 0, count: int = 1, policy=None, **overrides):
    """ScenarioEnv with rendering off. `db` is a key of DBS or a path."""
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.idm_policy import TrajectoryIDMPolicy

    _patch_vertex()
    cfg = dict(
        use_render=False,
        image_observation=False,
        data_directory=DBS.get(db, db),
        start_scenario_index=start,
        num_scenarios=count,
        agent_policy=policy or TrajectoryIDMPolicy,
        horizon=1000,
        # End the episode 2 s after the logged scenario ends. Without this a stopped ego
        # (fallback engaged) never "arrives" and the episode runs to `horizon` — 100 s of
        # simulation on an 11 s scenario, with traffic driving on past its log.
        allowed_more_steps=20,
        truncate_as_terminate=True,
        log_level=50,
    )
    cfg.update(overrides)
    return ScenarioEnv(cfg)


def ego_is_static(env) -> bool:
    return env.agent.navigation.reference_trajectory.length < MIN_TRACK_M


def commit_mb() -> float:
    """Current process private commit (Windows pagefile usage), MB. 0.0 if unavailable."""
    import ctypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32")
    fn = getattr(k32, "K32GetProcessMemoryInfo", None) or ctypes.WinDLL("psapi").GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    fn.restype = ctypes.c_int
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    m = PMC()
    m.cb = ctypes.sizeof(PMC)
    if not fn(k32.GetCurrentProcess(), ctypes.byref(m), m.cb):
        return 0.0
    return m.PagefileUsage / 2 ** 20


def free_commit_gb() -> float:
    """Machine-wide free commit (page file headroom), GB. This is the limit that bit us."""
    import ctypes

    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in
            ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
             "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

    s = MEMSTAT()
    s.dwLength = ctypes.sizeof(MEMSTAT)
    if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(s)):
        return float("inf")
    return s.ullAvailPageFile / 2 ** 30
