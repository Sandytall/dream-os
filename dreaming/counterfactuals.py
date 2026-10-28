"""Dream generator (Component D): counterfactual action branches.

A "dream" is a PREDICTED trajectory: starting from a real recorded observation,
we choose an action plan (a branch) and roll the world model forward, feeding its
own predictions back in. The result is imagined, never observed -- callers tag it
`status="predicted"`. Every branch is a deterministic function of
(start_obs, goal, branch_spec, seed, model), so dreams are exactly reproducible.
"""
from __future__ import annotations

import numpy as np
import torch

from world_model.dynamics import ResidualDynamics

GRIP = slice(0, 3)
OBJ = slice(3, 6)

# The counterfactual branches generated from each start state.
BRANCHES = {
    "toward_goal": {"kind": "scripted", "angle": 0.0},   # push straight at the goal
    "angle_left":  {"kind": "scripted", "angle": +35.0},  # approach from one side
    "angle_right": {"kind": "scripted", "angle": -35.0},  # approach from the other
    "noop":        {"kind": "fixed", "value": 0.0},       # control: do nothing
    "random":      {"kind": "random"},                    # exploratory
}


def load_model(checkpoint_path, device: str = "cpu"):
    """Rebuild the world model from a checkpoint without needing the training data
    (normalization buffers are restored from the checkpoint itself)."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ds, da = ck["d_state"], ck["d_act"]
    placeholder = {"s_mean": np.zeros(ds), "s_std": np.ones(ds),
                   "a_mean": np.zeros(da), "a_std": np.ones(da),
                   "d_mean": np.zeros(ds), "d_std": np.ones(ds)}
    model = ResidualDynamics(ds, da, placeholder)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck


def _rotate_xy(vec, deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    out = vec.copy()
    out[0] = c * vec[0] - s * vec[1]
    out[1] = s * vec[0] + c * vec[1]
    return out


def scripted_action(obs, goal, angle_deg=0.0):
    """Deterministic pusher toward (optionally rotated) goal direction."""
    grip, obj = obs[GRIP], obs[OBJ]
    to_goal = goal - obj
    if angle_deg:
        to_goal = _rotate_xy(to_goal, angle_deg)
    push_dir = to_goal / (np.linalg.norm(to_goal) + 1e-8)
    behind = obj - 0.055 * push_dir
    target = behind if np.linalg.norm(grip - behind) > 0.03 else obj + 0.1 * push_dir
    a = np.zeros(4, dtype=np.float32)
    a[:3] = np.clip(12.0 * (target - grip), -1.0, 1.0)
    return a


def _preplanned_actions(spec, horizon, d_act, seed):
    """Fixed action sequence for open-loop branches; None for closed-loop scripted."""
    if spec["kind"] == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=(horizon, d_act)).astype(np.float32)
    if spec["kind"] == "fixed":
        return np.full((horizon, d_act), spec["value"], dtype=np.float32)
    return None


@torch.no_grad()
def dream(model, start_obs, goal, spec, horizon, device, seed=0):
    """Roll the world model forward under a branch. Returns (states[H+1,D], actions[H,A]).

    states[0] is the real start; states[1:] are PREDICTIONS (hats).
    """
    d_act = model.d_act
    pre = _preplanned_actions(spec, horizon, d_act, seed)
    s = torch.as_tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)
    states = [np.asarray(start_obs, np.float32)]
    actions = []
    for t in range(horizon):
        if pre is not None:
            a = pre[t]
        else:
            a = scripted_action(s.squeeze(0).cpu().numpy(), goal, spec.get("angle", 0.0))
        at = torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
        s = model.predict(s, at)
        states.append(s.squeeze(0).cpu().numpy().astype(np.float32))
        actions.append(np.asarray(a, np.float32))
    return np.asarray(states, np.float32), np.asarray(actions, np.float32)
