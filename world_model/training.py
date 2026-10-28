"""Training and evaluation for the world model (Component B / learning).

Everything here reports measured numbers. The model is trained to predict the
normalized state-change; it is judged in REAL units (metres etc.) against a
persistence baseline, one-step and multi-step.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.dynamics import ResidualDynamics


# --------------------------------------------------------------------------- #
# statistics (fit on TRAIN only)
# --------------------------------------------------------------------------- #
def compute_stats(S, A, Snext) -> dict:
    delta = Snext - S
    return {
        "s_mean": S.mean(0), "s_std": S.std(0),
        "a_mean": A.mean(0), "a_std": A.std(0),
        "d_mean": delta.mean(0), "d_std": delta.std(0),
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rmse(pred, true, cols=None) -> float:
    if cols is not None:
        pred, true = pred[..., cols], true[..., cols]
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def one_step_metrics(pred_next, S, Snext, obj_cols) -> dict:
    return {
        "rmse_all": _rmse(pred_next, Snext),
        "rmse_object_pos_m": _rmse(pred_next, Snext, obj_cols),
    }


def persistence_metrics(S, Snext, obj_cols) -> dict:
    # baseline prediction = current state (nothing changes)
    return one_step_metrics(S, S, Snext, obj_cols)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(model: ResidualDynamics, train_set, val_set, device,
          epochs=300, batch_size=256, lr=1e-3, patience=30, seed=0) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    S, A, Snext = (torch.as_tensor(x, device=device) for x in train_set)
    Sv, Av, Snv = (torch.as_tensor(x, device=device) for x in val_set)

    # training target: normalized true delta
    d_true_norm = ((Snext - S) - model.d_mean) / model.d_std.clamp_min(1e-6)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = S.shape[0]
    best_val = float("inf")
    best_state = None
    bad = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model.pred_delta_norm(S[idx], A[idx])
            loss = ((pred - d_true_norm[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation one-step RMSE in real units
        model.eval()
        with torch.no_grad():
            pv = model.predict(Sv, Av)
            val_rmse = float(torch.sqrt(((pv - Snv) ** 2).mean()).item())
        history.append({"epoch": epoch, "val_rmse_all": val_rmse})

        if val_rmse < best_val - 1e-7:
            best_val, bad = val_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_val_rmse_all": best_val, "history": history}


# --------------------------------------------------------------------------- #
# multi-step rollout training (Phase 9B)
# --------------------------------------------------------------------------- #
def make_windows(eps, horizon):
    """Slice episodes into overlapping (s0, a[0:K], s[1:K+1]) rollout windows."""
    S0, Aw, St = [], [], []
    for obs, act in eps:
        Tn = len(act)
        for t in range(0, Tn - horizon + 1):
            S0.append(obs[t]); Aw.append(act[t:t + horizon]); St.append(obs[t + 1:t + horizon + 1])
    return (np.asarray(S0, np.float32), np.asarray(Aw, np.float32), np.asarray(St, np.float32))


def train_rollout(model: ResidualDynamics, train_eps, val_eps, device, horizon=5,
                  epochs=400, batch_size=256, lr=1e-3, patience=40, seed=0,
                  obj_cols=(3, 4, 5)) -> dict:
    """Same architecture, SAME one factor changed: the training objective is now a
    K-step open-loop rollout loss (back-prop through the model's own predictions)
    instead of a one-step loss. Loss at each step is normalized-state MSE (z-scored
    by s_std so all dims are comparable); early-stopping uses the mean object-position
    RMSE along the val rollout (the quantity Phase 9A showed compounding).
    """
    torch.manual_seed(seed)
    model.to(device)
    S0, Aw, St = (torch.as_tensor(x, device=device) for x in make_windows(train_eps, horizon))
    vS0, vAw, vSt = (torch.as_tensor(x, device=device) for x in make_windows(val_eps, horizon))
    s_std = model.s_std.clamp_min(1e-6)
    oc = list(obj_cols)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = S0.shape[0]
    best_val, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            s = S0[idx]
            loss = 0.0
            for k in range(horizon):
                s = model.predict(s, Aw[idx, k])
                loss = loss + (((s - St[idx, k]) / s_std) ** 2).mean()
            loss = loss / horizon
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            s = vS0
            step_rmse = []
            for k in range(horizon):
                s = model.predict(s, vAw[:, k])
                step_rmse.append(torch.sqrt(((s[:, oc] - vSt[:, k][:, oc]) ** 2).mean()))
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
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_batch(model, S, A, device) -> np.ndarray:
    S = torch.as_tensor(S, device=device)
    A = torch.as_tensor(A, device=device)
    return model.predict(S, A).cpu().numpy()


@torch.no_grad()
def rollout(model, obs0, actions, device) -> np.ndarray:
    """Open-loop: feed the model its own predictions forward through `actions`."""
    s = torch.as_tensor(obs0, device=device).unsqueeze(0)
    preds = []
    for a in actions:
        at = torch.as_tensor(a, device=device).unsqueeze(0)
        s = model.predict(s, at)
        preds.append(s.squeeze(0).cpu().numpy())
    return np.asarray(preds)  # [T, D]


def evaluate_multistep(model, eps, device, horizons, obj_cols) -> dict:
    """For each horizon h: object-position RMSE of the model's h-step-ahead
    open-loop prediction vs truth, and the persistence baseline at the same h."""
    model_err = {h: [] for h in horizons}
    base_err = {h: [] for h in horizons}
    for obs, actions in eps:
        preds = rollout(model, obs[0], actions, device)   # [T, D]
        T = preds.shape[0]
        for h in horizons:
            if h <= T:
                model_err[h].append(_rmse(preds[h - 1], obs[h], obj_cols))
                base_err[h].append(_rmse(obs[0], obs[h], obj_cols))  # persistence = obs[0]
    out = {}
    for h in horizons:
        if model_err[h]:
            out[h] = {"model_object_rmse_m": round(float(np.mean(model_err[h])), 4),
                      "baseline_object_rmse_m": round(float(np.mean(base_err[h])), 4),
                      "episodes": len(model_err[h])}
    return out
