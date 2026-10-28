"""An ensemble of world models -- the uncertainty source (Component B, uncertainty).

Several dynamics models are trained on different bootstrap resamples of the same
data with different initializations. Where they AGREE, the model is confident;
where they DISAGREE, it is uncertain. That disagreement is the honest uncertainty
signal used by uncertainty/information-gain dream selection (Phase 5).
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.dynamics import ResidualDynamics
from world_model import training as T


def train_ensemble(train_set, val_set, n_models, device,
                   seed0=0, epochs=300, patience=30) -> list:
    S, A, Sn = train_set
    stats = T.compute_stats(S, A, Sn)          # shared normalization (fair)
    d_state, d_act = S.shape[1], A.shape[1]
    rng = np.random.default_rng(seed0)
    models = []
    for i in range(n_models):
        idx = rng.integers(0, len(S), size=len(S))          # bootstrap resample
        boot = (S[idx], A[idx], Sn[idx])
        torch.manual_seed(seed0 + i)                        # different init
        m = ResidualDynamics(d_state, d_act, stats)
        T.train(m, boot, val_set, device, epochs=epochs, patience=patience, seed=seed0 + i)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def rollout_ensemble(models, start_obs, actions, device) -> np.ndarray:
    """Return predicted trajectories, one per member: array [N, H+1, D]."""
    outs = []
    for m in models:
        s = torch.as_tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)
        traj = [np.asarray(start_obs, np.float32)]
        for a in actions:
            at = torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
            s = m.predict(s, at)
            traj.append(s.squeeze(0).cpu().numpy().astype(np.float32))
        outs.append(np.asarray(traj, np.float32))
    return np.asarray(outs)


def mean_std(preds: np.ndarray):
    """preds [N,H+1,D] -> (mean[H+1,D], std[H+1,D])."""
    return preds.mean(axis=0), preds.std(axis=0)
