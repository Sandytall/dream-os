"""Learned acquisition function (Phase 7, Component E+).

Phase 6 showed that hand-built pre-verification signals (ensemble uncertainty,
coverage/novelty) do NOT identify the experiments that most improve the world
model -- only the post-hoc ORACLE does, and it is not usable live. This learns an
acquisition function: a small model, SEPARATE from the world model, that maps
pre-verification features of a candidate dream to its expected real discrepancy
(its "informativeness"), trained on PAST verified dreams.

Scientific guards: its training targets are REAL-simulator outcomes, never the
world model's own self-assessment; and it is always evaluated on candidate dreams
it was not trained on (the caller splits candidates by start state).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

OBJ = slice(3, 6)
FEATURE_NAMES = ["unc_mean", "unc_max", "novelty", "imag_disp",
                 "imag_maxstep", "act_mean_abs", "act_std"]


def candidate_features(mean_traj, std_traj, novelty, actions) -> np.ndarray:
    """Pre-verification features -- all computable WITHOUT touching the real sim."""
    finite = bool(np.all(np.isfinite(mean_traj)))
    obj_std = np.linalg.norm(std_traj[:, OBJ], axis=1)
    if finite:
        disp = float(np.linalg.norm(mean_traj[-1, OBJ] - mean_traj[0, OBJ]))
        step = np.linalg.norm(np.diff(mean_traj[:, OBJ], axis=0), axis=1)
        maxstep = float(np.max(step)) if step.size else 0.0
    else:
        disp, maxstep = 10.0, 10.0
    return np.array([
        float(np.mean(obj_std)), float(np.max(obj_std)), float(novelty),
        min(disp, 10.0), min(maxstep, 10.0),
        float(np.mean(np.abs(actions))), float(np.std(actions)),
    ], np.float32)


class _MLP(nn.Module):
    def __init__(self, d, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_predictor(X, y, device="cpu", epochs=600, seed=0, weight_decay=1e-4):
    """Standardize BOTH features and target (stats from TRAIN only) and fit a tiny MLP.

    Standardizing the target matters: informativeness here is tiny in magnitude
    (~0.01 m), so without it the MSE and its gradients are vanishingly small and the
    model collapses to predicting the mean.
    """
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ym, ysd = float(y.mean()), float(y.std() + 1e-9)
    torch.manual_seed(seed)
    m = _MLP(X.shape[1]).to(device)
    xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.as_tensor((y - ym) / ysd, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((m(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    m.eval()
    return {"model": m, "mu": mu, "sd": sd, "ym": ym, "ysd": ysd, "device": device}


@torch.no_grad()
def predict(pred, X) -> np.ndarray:
    X = np.asarray(X, np.float32)
    xt = torch.as_tensor((X - pred["mu"]) / pred["sd"], dtype=torch.float32, device=pred["device"])
    return pred["model"](xt).cpu().numpy() * pred["ysd"] + pred["ym"]
