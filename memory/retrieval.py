"""Episodic memory -- retrieval (Component C).

Selects real recorded states to start dreams from. Each returned reference keeps
enough to (a) trace it back to the exact source transition and (b) later restore
the simulator to it for verification (qpos/qvel), so dreams stay grounded in real
starting conditions.
"""
from __future__ import annotations

import numpy as np

from memory import episode_store as store


def sample_start_states(dataset_id: str, split: str = "test_heldout",
                        n: int = 6, seed: int = 0) -> list[dict]:
    """Pick n (episode, t) start references from `split`, deterministically."""
    eps = store.splits(dataset_id)[split]
    rng = np.random.default_rng(seed)
    refs = []
    ep_picks = rng.choice(len(eps), size=n, replace=len(eps) < n)
    for k in ep_picks:
        ep = int(eps[int(k)])
        d = store.load_episode(dataset_id, ep)
        horizon_in_ep = d["actions"].shape[0]
        # start somewhere in the first ~60% so there is real motion ahead of it
        t = int(rng.integers(0, max(1, int(0.6 * horizon_in_ep))))
        refs.append({
            "dataset_id": dataset_id, "split": split, "episode": ep, "t": t,
            "observation": d["observation"][t].astype(np.float32),
            "qpos": d["qpos"][t].astype(np.float32),
            "qvel": d["qvel"][t].astype(np.float32),
            "desired_goal": d["desired_goal"].astype(np.float32),
        })
    return refs
