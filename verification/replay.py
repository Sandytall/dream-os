"""Reference-simulator verifier -- replay (Component F).

Reconstructs a dream's real start state WITHOUT state surgery: reset(seed) then
replay the real recorded actions up to step t (proven exact in Phase 1). From that
true state it replays the DREAM's action sequence in real MuJoCo physics and
returns the true trajectory, so predicted (s-hat) and real (s) can be compared.
"""
from __future__ import annotations

import numpy as np

from envs.benchmark_interface import make_env, obs_parts
from memory import episode_store as store


def episode_seed(dataset_id: str, episode: int) -> int:
    for e in store.manifest(dataset_id)["episode_list"]:
        if e["episode"] == episode:
            return int(e["seed"])
    raise KeyError(f"episode {episode} not in manifest for {dataset_id}")


def reconstruct_and_replay(env, dataset_id: str, episode: int, t: int,
                           dream_actions: np.ndarray):
    """Return (real_states[H+1, D], reconstruction_error).

    reconstruction_error is how far the rebuilt start state is from the recorded
    one; it should be ~0 because reset+replay is deterministic. A large value means
    the start could not be reconstructed and the comparison is unresolved.
    """
    seed = episode_seed(dataset_id, episode)
    d = store.load_episode(dataset_id, episode)

    obs, _ = env.reset(seed=seed)
    for a in d["actions"][:t]:                 # advance to the dream's start state
        obs, *_ = env.step(a)

    start_vec = obs_parts(obs)[0]
    recon_err = float(np.max(np.abs(start_vec - d["observation"][t])))

    real = [start_vec]
    for a in dream_actions:                    # replay the DREAM's actions for real
        obs, *_ = env.step(a)
        real.append(obs_parts(obs)[0])

    return np.asarray(real, np.float32), recon_err
