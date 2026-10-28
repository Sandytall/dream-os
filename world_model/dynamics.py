"""The world model (Component B): a small action-conditioned dynamics model.

ResidualDynamics predicts the CHANGE in state rather than the next state:

    s_next = s + denorm( net( [norm(s), norm(a)] ) )

The normalization statistics (means/stds, fit on TRAIN data) travel with the model
as buffers, so a saved checkpoint predicts identically after reload. Deliberately
tiny -- this must run comfortably in 6 GB of VRAM and is the honest first baseline
before any larger architecture is considered.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STAT_KEYS = ("s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std")


class ResidualDynamics(nn.Module):
    def __init__(self, d_state: int, d_act: int, stats: dict, hidden: int = 256):
        super().__init__()
        self.d_state = d_state
        self.d_act = d_act
        self.net = nn.Sequential(
            nn.Linear(d_state + d_act, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_state),
        )
        # normalization stats as buffers (move with .to(device), saved in checkpoint)
        for k in STAT_KEYS:
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))

    @staticmethod
    def _z(x, mean, std):
        return (x - mean) / std.clamp_min(1e-6)

    def pred_delta_norm(self, s, a):
        """Network output: the predicted state-change in normalized units."""
        x = torch.cat([self._z(s, self.s_mean, self.s_std),
                       self._z(a, self.a_mean, self.a_std)], dim=-1)
        return self.net(x)

    def predict(self, s, a):
        """Predicted next state in real units: s + denormalized delta."""
        delta = self.pred_delta_norm(s, a) * self.d_std.clamp_min(1e-6) + self.d_mean
        return s + delta

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
