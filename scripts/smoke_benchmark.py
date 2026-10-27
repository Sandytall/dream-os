"""Stage 0 smoke test: the selected prebuilt MuJoCo benchmarks.

Primary benchmark : FetchPush-v3   (gymnasium-robotics) -- object pushing, goal,
                                     explicit `is_success`, structured state.
Secondary / smoke : Pusher-v5      (gymnasium MuJoCo)    -- zero-extra-dep push.

For each we: reset reproducibly, step valid actions, read observation / reward /
termination / truncation / task outcome, retrieve simulator state, and verify
that a fixed seed + fixed action sequence reproduces the trajectory exactly.
Nothing here forces a success; the task outcome is reported as measured.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

try:  # gymnasium-robotics registers its envs on import (gymnasium 1.x pattern)
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    _GR_OK = True
except Exception as exc:  # pragma: no cover - reported, not hidden
    _GR_OK = False
    _GR_ERR = repr(exc)


def _get_qpos(env):
    """Best-effort retrieval of structured simulator state."""
    u = env.unwrapped
    data = getattr(u, "data", None)
    if data is not None and hasattr(data, "qpos"):
        return np.asarray(data.qpos).copy()
    return None


def _fixed_actions(env, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    return [rng.uniform(low, high).astype(np.float32) for _ in range(n)]


def _obs_vec(obs) -> np.ndarray:
    if isinstance(obs, dict):  # goal-conditioned envs (Fetch)
        return np.asarray(obs["observation"], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64)


def run_env(env_id: str, steps: int = 40, seed: int = 0) -> dict:
    env = gym.make(env_id)
    actions = _fixed_actions(env, steps, seed=123)

    def rollout() -> tuple[list, dict, float, bool]:
        obs, info = env.reset(seed=seed)
        traj, total, success = [_obs_vec(obs)], 0.0, False
        term = trunc = False
        for a in actions:
            obs, r, term, trunc, info = env.step(a)
            traj.append(_obs_vec(obs))
            total += float(r)
            success = success or bool(info.get("is_success", 0.0))
            if term or trunc:
                break
        return traj, info, total, success

    traj1, info1, ret1, success1 = rollout()
    qpos = _get_qpos(env)
    traj2, _, ret2, _ = rollout()  # same seed + same actions -> must match
    env.close()

    reproducible = np.allclose(np.array(traj1), np.array(traj2), atol=1e-8)

    sample_obs = traj1[0]
    return {
        "env_id": env_id,
        "obs_is_dict": bool(isinstance(env.observation_space, gym.spaces.Dict)),
        "obs_dim": int(sample_obs.shape[0]),
        "qpos_dim": None if qpos is None else int(qpos.shape[0]),
        "episode_return": round(ret1, 4),
        "is_success": bool(success1),
        "reproducible": bool(reproducible),
        "passed": bool(reproducible and sample_obs.size > 0),
    }


def run() -> dict:
    results = {"gymnasium_version": gym.__version__, "gymnasium_robotics": _GR_OK}
    if not _GR_OK:
        results["gymnasium_robotics_error"] = _GR_ERR

    results["pusher"] = run_env("Pusher-v5")

    if _GR_OK:
        fetch_id = None
        for cand in ("FetchPush-v4", "FetchPush-v3", "FetchPush-v2", "FetchPush-v1"):
            if cand in gym.registry:
                fetch_id = cand
                break
        if fetch_id is None:
            fetch_avail = sorted(e for e in gym.registry if "Fetch" in e)
            results["fetchpush"] = {"passed": False, "error": "FetchPush not registered",
                                    "available_fetch": fetch_avail}
        else:
            results["fetchpush"] = run_env(fetch_id)

    checks = [v for k, v in results.items() if isinstance(v, dict) and "passed" in v]
    results["passed"] = bool(checks) and all(c["passed"] for c in checks)
    return results


if __name__ == "__main__":
    import json

    r = run()
    print(json.dumps(r, indent=2))
    print("BENCHMARK SMOKE:", "PASS" if r["passed"] else "FAIL")
