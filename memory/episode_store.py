"""Episodic memory -- read side (Component C).

Reads the on-disk dataset (per-episode NPZ + JSON manifests) and serves it two
ways: as flat (state, action, next_state) transitions for one-step training, and
as whole episodes for multi-step rollout evaluation. The split files are
authoritative, so train/val/held-out never mix here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def manifest(dataset_id: str) -> dict:
    return json.loads((ROOT / "data" / "manifests" / f"{dataset_id}.json").read_text())


def splits(dataset_id: str) -> dict:
    return json.loads(
        (ROOT / "data" / "manifests" / f"{dataset_id}_splits.json").read_text())


def _ep_path(dataset_id: str, ep_idx: int) -> Path:
    return ROOT / "data" / "raw" / dataset_id / f"ep_{ep_idx:04d}.npz"


def load_episode(dataset_id: str, ep_idx: int):
    return np.load(_ep_path(dataset_id, ep_idx))


def transitions(dataset_id: str, which: str):
    """Return flat arrays (S, A, S_next) for every episode in split `which`.

    S[i], A[i] -> S_next[i] is one transition. `which` is 'train' | 'val' | 'test_heldout'.
    """
    sp = splits(dataset_id)
    S, A, Snext = [], [], []
    for ep in sp[which]:
        d = load_episode(dataset_id, ep)
        obs = d["observation"]          # [T+1, D]
        S.append(obs[:-1])
        Snext.append(obs[1:])
        A.append(d["actions"])          # [T, A]
    return (np.concatenate(S).astype(np.float32),
            np.concatenate(A).astype(np.float32),
            np.concatenate(Snext).astype(np.float32))


def episodes(dataset_id: str, which: str):
    """Return a list of (observation[T+1,D], actions[T,A]) for multi-step eval."""
    sp = splits(dataset_id)
    out = []
    for ep in sp[which]:
        d = load_episode(dataset_id, ep)
        out.append((d["observation"].astype(np.float32), d["actions"].astype(np.float32)))
    return out
