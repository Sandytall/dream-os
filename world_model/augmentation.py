"""Phase 6 -- learning from verified imagination (augmentation + retrain/eval).

Helpers to (a) turn a real-replayed or imagined rollout into (S, A, S_next)
training transitions, (b) train a FRESH world model on base-plus-extra data, and
(c) evaluate it on the FIXED held-out test set in real units.

Scientific guard: evaluation never uses the model's own predictions as ground
truth -- only the real recorded/replayed held-out data. Imagined (model-predicted)
transitions are produced ONLY for the explicit "harmful dreams" safety condition
and are tagged as such by the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.dynamics import ResidualDynamics
from world_model import training as T


def rollout_transitions(states, actions):
    """(states[H+1, D], actions[H, A]) -> (S[H, D], A[H, A], S_next[H, D])."""
    states = np.asarray(states, np.float32)
    actions = np.asarray(actions, np.float32)
    H = len(actions)
    return states[:H], actions[:H], states[1:H + 1]


def imagined_transitions(model, start_obs, actions, device):
    """One reference model's OWN open-loop rollout, as (S, A, S_next).

    Each (s, a, s') is exactly a transition the model predicts -- i.e. the model's
    hallucination. Used only for the harmful-dreams safety test.
    """
    preds = T.rollout(model, start_obs, actions, device)            # [H, D] next-states
    states = np.concatenate([np.asarray(start_obs, np.float32)[None], preds], 0)
    return rollout_transitions(states, actions)


def finite_mask(trans, bound=1e3):
    """Drop transitions with non-finite or absurd magnitudes (diverged dreams)."""
    S, A, Sn = trans
    m = (np.all(np.isfinite(S), 1) & np.all(np.isfinite(A), 1) & np.all(np.isfinite(Sn), 1)
         & (np.max(np.abs(S), 1) < bound) & (np.max(np.abs(Sn), 1) < bound))
    return (S[m], A[m], Sn[m]), int((~m).sum())


def concat_sets(base, extra):
    """Concatenate (S, A, S_next) tuples; extra may be None/empty."""
    if extra is None or len(extra[0]) == 0:
        return base
    return tuple(np.concatenate([base[i], extra[i]]).astype(np.float32) for i in range(3))


def train_fresh(train_set, val_set, device, seed, epochs=300, patience=30):
    """Fit normalization on THIS condition's own train data, then train a fresh model."""
    stats = T.compute_stats(*train_set)
    torch.manual_seed(seed)
    m = ResidualDynamics(train_set[0].shape[1], train_set[1].shape[1], stats)
    T.train(m, train_set, val_set, device, epochs=epochs, patience=patience, seed=seed)
    m.eval()
    return m


def eval_on_test(model, test_trans, test_eps, device, obj_cols, phys_cols, horizons):
    """Held-out metrics in real units (real test data only, never model targets)."""
    S, _, Snext = test_trans
    pred = T.predict_batch(model, S, test_trans[1], device)

    def rmse(cols):
        return float(np.sqrt(np.mean((pred[:, cols] - Snext[:, cols]) ** 2)))

    ms = T.evaluate_multistep(model, test_eps, device, horizons, obj_cols)
    return {
        "object_pos_rmse_m": round(rmse(obj_cols), 5),
        "physical_state_rmse": round(rmse(phys_cols), 5),
        "multistep_object_rmse_m": {h: ms[h]["model_object_rmse_m"] for h in ms},
    }
