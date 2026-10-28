"""Augmented-velocity dynamics (Phase 9E, experiment E_abs_velocity_feature).

The Phase 9C/9D failure: on free-sliding FetchSlide windows the model under-predicts
how far the puck coasts, and the failure concentrates exactly where the gripper is FAR
from the puck. Hypothesis H_velocity says the cause is representational: the only
object-velocity channel in the observation (``object_velp``, cols 14:17) is stored
RELATIVE to the gripper (verified in Phase 9A), so once the gripper leaves, the model
has no absolute momentum signal and defaults to "stopped".

This module changes exactly ONE factor versus the Phase 9D rollout-K5 subject: it adds
an ABSOLUTE object-velocity input, computed as a finite difference of consecutive object
positions. During an open-loop rollout the feature is recomputed from the model's OWN
predicted positions, so the loop stays closed and no privileged information leaks in.
Architecture, horizon, optimizer, data, and seed are otherwise identical, so any change
in the free-slide travel bias is attributable to the added velocity signal alone.

    s_aug_t   = [ obs_t (d_base) , (obj_pos_t - obj_pos_{t-1}) (d_vel) ]
    s_{t+1}   = obs_t + denorm( net([ norm(s_aug_t), norm(a_t) ]) )     # predicts d_base delta
    v_{t+1}   = obj_pos_{t+1} - obj_pos_t                               # recomputed, closes the loop
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

STAT_KEYS = ("s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std")


class AugVelDynamics(nn.Module):
    """Residual dynamics whose INPUT is the state augmented with absolute object
    velocity, but whose OUTPUT is the ordinary d_base state-change (the velocity
    feature is derived, not predicted)."""

    def __init__(self, d_base: int, d_act: int, obj_cols, stats: dict, hidden: int = 256):
        super().__init__()
        self.d_base = d_base
        self.d_act = d_act
        self.obj_cols = list(obj_cols)
        self.d_vel = len(self.obj_cols)
        self.net = nn.Sequential(
            nn.Linear(d_base + self.d_vel + d_act, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_base),
        )
        # s_mean/s_std are (d_base + d_vel); d_mean/d_std are d_base (base-state delta)
        for k in STAT_KEYS:
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))

    @staticmethod
    def _z(x, mean, std):
        return (x - mean) / std.clamp_min(1e-6)

    def pred_delta_norm(self, s_base, v, a):
        s_aug = torch.cat([s_base, v], dim=-1)
        x = torch.cat([self._z(s_aug, self.s_mean, self.s_std),
                       self._z(a, self.a_mean, self.a_std)], dim=-1)
        return self.net(x)

    def predict(self, s_base, v, a):
        """Next base state in real units."""
        delta = self.pred_delta_norm(s_base, v, a) * self.d_std.clamp_min(1e-6) + self.d_mean
        return s_base + delta

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# data preparation (object velocity = finite difference of object position)
# --------------------------------------------------------------------------- #
def _vel(obs, t, obj_cols):
    return obs[t][obj_cols] - obs[t - 1][obj_cols]


def aug_stats(eps, obj_cols) -> dict:
    """Fit normalization stats on TRAIN one-step aug transitions (t>=1 so a previous
    object position exists for the finite difference)."""
    S_aug, A, D = [], [], []
    for obs, act in eps:
        for t in range(1, len(act)):           # need obs[t-1]; target obs[t+1]
            S_aug.append(np.concatenate([obs[t], _vel(obs, t, obj_cols)]))
            A.append(act[t])
            D.append(obs[t + 1] - obs[t])
    S_aug = np.asarray(S_aug, np.float32); A = np.asarray(A, np.float32); D = np.asarray(D, np.float32)
    return {
        "s_mean": S_aug.mean(0), "s_std": S_aug.std(0),
        "a_mean": A.mean(0), "a_std": A.std(0),
        "d_mean": D.mean(0), "d_std": D.std(0),
    }


def make_aug_windows(eps, horizon, obj_cols):
    """Rollout windows (s0_base, v0, a[0:K], s[1:K+1]); windows start at t>=1 so v0
    is a real finite difference, not a guess."""
    S0, V0, Aw, St = [], [], [], []
    for obs, act in eps:
        Tn = len(act)
        for t in range(1, Tn - horizon + 1):
            S0.append(obs[t]); V0.append(_vel(obs, t, obj_cols))
            Aw.append(act[t:t + horizon]); St.append(obs[t + 1:t + horizon + 1])
    return (np.asarray(S0, np.float32), np.asarray(V0, np.float32),
            np.asarray(Aw, np.float32), np.asarray(St, np.float32))


# --------------------------------------------------------------------------- #
# training (K-step open-loop rollout loss; velocity recomputed each step)
# --------------------------------------------------------------------------- #
def train_aug_rollout(model: AugVelDynamics, train_eps, val_eps, device, base_s_std,
                      horizon=5, epochs=400, batch_size=256, lr=1e-3, patience=40,
                      seed=0, obj_cols=(3, 4, 5)) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    oc = list(obj_cols)
    base_s_std = torch.as_tensor(base_s_std, device=device).clamp_min(1e-6)

    def to_dev(x):
        return torch.as_tensor(x, device=device)

    S0, V0, Aw, St = (to_dev(x) for x in make_aug_windows(train_eps, horizon, oc))
    vS0, vV0, vAw, vSt = (to_dev(x) for x in make_aug_windows(val_eps, horizon, oc))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = S0.shape[0]
    best_val, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s, v = S0[idx], V0[idx]
            loss = 0.0
            for k in range(horizon):
                s_next = model.predict(s, v, Aw[idx, k])
                loss = loss + (((s_next - St[idx, k]) / base_s_std) ** 2).mean()
                v = s_next[:, oc] - s[:, oc]        # recompute absolute object velocity
                s = s_next
            loss = loss / horizon
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            s, v = vS0, vV0
            step_rmse = []
            for k in range(horizon):
                s_next = model.predict(s, v, vAw[:, k])
                step_rmse.append(torch.sqrt(((s_next[:, oc] - vSt[:, k][:, oc]) ** 2).mean()))
                v = s_next[:, oc] - s[:, oc]
                s = s_next
            val_rmse = float(torch.stack(step_rmse).mean().item())
        history.append({"epoch": epoch, "val_obj_rollout_rmse": val_rmse})

        if val_rmse < best_val - 1e-7:
            best_val, bad = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_val_obj_rollout_rmse": best_val,
            "horizon": horizon, "n_windows": int(n), "history": history}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def aug_rollout(model: AugVelDynamics, s0_base, v0, actions, device, obj_cols=(3, 4, 5)) -> np.ndarray:
    """Open-loop rollout; the velocity feature is recomputed from predicted positions.
    `v0` is the object velocity entering the first step (0 at an episode start where the
    puck is at rest, or a real finite difference for a mid-episode window)."""
    oc = list(obj_cols)
    s = torch.as_tensor(np.atleast_2d(s0_base), device=device).float()
    v = torch.as_tensor(np.atleast_2d(v0), device=device).float()
    preds = []
    for a in actions:
        at = torch.as_tensor(np.atleast_2d(a), device=device).float()
        s_next = model.predict(s, v, at)
        v = s_next[:, oc] - s[:, oc]
        s = s_next
        preds.append(s.squeeze(0).cpu().numpy())
    return np.asarray(preds)  # [T, d_base]


@torch.no_grad()
def aug_rollout_batch(model: AugVelDynamics, S0, V0, Aw, device, obj_cols=(3, 4, 5)) -> np.ndarray:
    """Batched open-loop rollout over windows. S0[N,d_base], V0[N,d_vel], Aw[N,K,d_act]
    -> [N, K, d_base]."""
    oc = list(obj_cols)
    s = torch.as_tensor(S0, device=device).float()
    v = torch.as_tensor(V0, device=device).float()
    Aw = torch.as_tensor(Aw, device=device).float()
    out = []
    for k in range(Aw.shape[1]):
        s_next = model.predict(s, v, Aw[:, k])
        v = s_next[:, oc] - s[:, oc]
        s = s_next
        out.append(s)
    return torch.stack(out, dim=1).cpu().numpy()
