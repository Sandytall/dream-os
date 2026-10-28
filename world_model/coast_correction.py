"""Stable momentum / velocity-decay correction (Phase 9F, E_damping_intervention).

Phase 9C/9D/9E established that the model UNDER-predicts how far a free-sliding puck coasts.
The adopted model is a residual MLP ``s_{t+1} = s_t + f(s_t, a_t)`` with NO explicit damping
parameter -- the under-travel is a *learned* velocity decay. Phase 9E's absolute-velocity input
was rejected: it did not reduce the bias and diverged from rest because the MLP amplified an
unbounded velocity feature.

This module tests the damping explanation with the SMALLEST, provably stable intervention: a
bounded momentum re-injection on the OBJECT channel only,

    obj_pred'_{k+1} = obj_pred_{k+1} + alpha * delta_{k-1}         (alpha in [0,1))

where ``delta_{k-1}`` is the object displacement of the previous step (the real finite difference
at the window start, or 0 at an episode start where the puck is at rest). The MLP never sees the
velocity, so it cannot amplify it; ignoring the (small) MLP motion the recursion is
``delta_k = alpha * delta_{k-1}``, i.e. geometric decay -> bounded for alpha<1, and exactly the
baseline at rest (delta=0) or at alpha=0. This is a different, stable form from the one Phase 9E
rejected, as §3 of the Phase 9F brief requires.

If a single calibrated alpha reduces held-out free-slide ENDPOINT error (not merely net bias)
without regressions, that is positive evidence for excess multiplicative damping. If it cannot,
the under-travel is not a constant damping offset (it points back to partial observability of the
absolute velocity).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def corrected_rollout_batch(model, S0, V0, Aw, device, alpha, obj_cols=(3, 4, 5)) -> np.ndarray:
    """Batched open-loop rollout with the bounded momentum correction on the object channel.
    S0[N,D], V0[N,len(obj)] (object velocity entering the first step), Aw[N,K,A] -> [N,K,D].
    alpha=0 reproduces the uncorrected baseline exactly."""
    oc = list(obj_cols)
    s = torch.as_tensor(S0, device=device).float()
    dprev = torch.as_tensor(V0, device=device).float()
    Aw = torch.as_tensor(Aw, device=device).float()
    out = []
    for k in range(Aw.shape[1]):
        sp = model.predict(s, Aw[:, k]).clone()
        sp[:, oc] = sp[:, oc] + alpha * dprev
        dprev = sp[:, oc] - s[:, oc]
        s = sp
        out.append(s)
    return torch.stack(out, dim=1).cpu().numpy()


@torch.no_grad()
def corrected_rollout_episode(model, obs0, v0, actions, device, alpha, obj_cols=(3, 4, 5)) -> np.ndarray:
    """Single-episode open-loop rollout with the correction; v0=0 at an at-rest episode start."""
    oc = list(obj_cols)
    s = torch.as_tensor(np.atleast_2d(obs0), device=device).float()
    dprev = torch.as_tensor(np.atleast_2d(v0), device=device).float()
    preds = []
    for a in actions:
        at = torch.as_tensor(np.atleast_2d(a), device=device).float()
        sp = model.predict(s, at).clone()
        sp[:, oc] = sp[:, oc] + alpha * dprev
        dprev = sp[:, oc] - s[:, oc]
        s = sp
        preds.append(s.squeeze(0).cpu().numpy())
    return np.asarray(preds)  # [T, D]


def endpoint_rmse(preds, reals_end, obj_cols=(3, 4, 5)) -> float:
    """Mean per-window object-position error at the final rolled-out step."""
    oc = list(obj_cols)
    return float(np.mean(np.linalg.norm(preds[:, -1, oc] - reals_end[:, oc], axis=1)))


def calibrate_alpha(model, S0, V0, Aw, reals_end, alphas, device, obj_cols=(3, 4, 5)) -> dict:
    """Pick alpha minimizing free-slide endpoint RMSE on a CALIBRATION (val) set.
    reals_end[N,D] is the real object state at the final horizon step."""
    sweep = {}
    for a in alphas:
        preds = corrected_rollout_batch(model, S0, V0, Aw, device, a, obj_cols)
        sweep[round(float(a), 3)] = round(endpoint_rmse(preds, reals_end, obj_cols), 5)
    best = min(sweep, key=sweep.get)
    return {"alpha": float(best), "val_endpoint_rmse_m": sweep[best], "sweep": sweep}
