"""Thin, documented adapter around the selected prebuilt MuJoCo benchmarks.

Keeps DREAM-OS code decoupled from gymnasium-robotics registration details and
provides uniform access to (a) the learning observation and (b) the raw simulator
state needed for exact replay/verification. Preserves the reference physics; adds
no dynamics of its own.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

_REGISTERED = False


def _ensure_registered() -> None:
    global _REGISTERED
    if not _REGISTERED:
        import gymnasium_robotics  # noqa: WPS433 (registers envs on import)
        gym.register_envs(gymnasium_robotics)
        _REGISTERED = True


def make_env(env_id: str = "FetchPush-v4", **kwargs):
    """Create a registered benchmark env (gymnasium-robotics or core gymnasium)."""
    _ensure_registered()
    return gym.make(env_id, **kwargs)


def get_sim_state(env) -> tuple[np.ndarray, np.ndarray]:
    """Return a copy of the MuJoCo (qpos, qvel) — the ground-truth simulator state."""
    data = env.unwrapped.data
    return np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()


def obs_parts(obs):
    """Split an observation into (observation_vec, achieved_goal, desired_goal).

    Goal-conditioned envs (Fetch) return a dict; flat envs (Pusher) return a vector,
    for which the goal fields are None.
    """
    if isinstance(obs, dict):
        ag = obs.get("achieved_goal")
        dg = obs.get("desired_goal")
        return (
            np.asarray(obs["observation"], dtype=np.float32),
            None if ag is None else np.asarray(ag, dtype=np.float32),
            None if dg is None else np.asarray(dg, dtype=np.float32),
        )
    return np.asarray(obs, dtype=np.float32), None, None
