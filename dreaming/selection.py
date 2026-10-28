"""Dream selection strategies (Component E).

Each strategy scores candidate dreams using ONLY information available before
verification (never the true outcome), then picks the top-K. The composite score
is the J(d)-style combination of normalized signals. An `oracle` (selects by the
true, post-hoc discrepancy) is provided only as an upper-bound ceiling for
evaluation -- it is not a usable strategy.
"""
from __future__ import annotations

import numpy as np

OBJ = slice(3, 6)


def uncertainty_score(std_traj: np.ndarray) -> float:
    """Mean ensemble disagreement on object position over the rollout."""
    return float(np.mean(np.linalg.norm(std_traj[:, OBJ], axis=1)))


def novelty_score(start_obs: np.ndarray, train_states_norm: np.ndarray,
                  s_mean: np.ndarray, s_std: np.ndarray) -> float:
    """Distance (normalized) from the start state to the nearest TRAIN state.
    Larger = more novel / less covered by training data (coverage signal)."""
    z = (start_obs - s_mean) / np.maximum(s_std, 1e-6)
    return float(np.min(np.linalg.norm(train_states_norm - z, axis=1)))


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return (x - x.min()) / (span + 1e-9)


def composite_score(uncertainty: np.ndarray, novelty: np.ndarray,
                    w_uncertainty: float = 1.0, w_coverage: float = 1.0) -> np.ndarray:
    """J(d) = w_u * norm(uncertainty) + w_c * norm(coverage)."""
    return w_uncertainty * _minmax(uncertainty) + w_coverage * _minmax(novelty)


def select_topk(scores, k: int) -> list:
    return list(np.argsort(np.asarray(scores))[::-1][:k])


def select_random(n: int, k: int, seed: int) -> list:
    return list(np.random.default_rng(seed).choice(n, size=k, replace=False))
