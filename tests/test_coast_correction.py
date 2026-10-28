"""Unit tests for the Phase 9F bounded momentum correction operator.

These test the CORRECTION OPERATOR (not a trained model), using a maximally-damped stub whose
predict() returns the state unchanged (predicts no object motion). Against a real constant-coast
trajectory this isolates three required properties:
  1. alpha=0 reproduces the baseline exactly (no silent change).
  2. rest-safety: with zero incoming velocity the correction is a no-op for any alpha.
  3. with the stub predicting no motion and a no-friction real coast, alpha=1 recovers the coast
     (endpoint error -> ~0) while alpha=0 does not; and alpha<1 stays bounded (geometric decay).

Run: env -u PYTHONPATH ./.venv/bin/python tests/test_coast_correction.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from world_model import coast_correction as CC                             # noqa: E402

OBJ = [0, 1, 2]
D, A, K = 4, 2, 10


class IdentityStub:
    """Maximally over-damped 'model': predicts the object does not move at all."""
    def predict(self, s, a):
        return s.clone() if torch.is_tensor(s) else torch.as_tensor(s).clone()


def _coast_window(v0, n_windows=16, seed=0):
    """Real trajectory: object coasts at constant velocity v0 (no friction)."""
    rng = np.random.default_rng(seed)
    S0 = np.zeros((n_windows, D), np.float32)
    S0[:, OBJ] = rng.uniform(-0.1, 0.1, size=(n_windows, 3))
    V0 = np.tile(np.asarray(v0, np.float32), (n_windows, 1))
    Aw = rng.uniform(-1, 1, size=(n_windows, K, A)).astype(np.float32)
    # real end position after K constant-velocity steps
    real_end = S0.copy()
    real_end[:, OBJ] = S0[:, OBJ] + K * V0
    return S0, V0, Aw, real_end


def test_alpha_zero_is_baseline():
    m = IdentityStub()
    S0, V0, Aw, _ = _coast_window([0.02, -0.01, 0.0])
    base = CC.corrected_rollout_batch(m, S0, V0, Aw, "cpu", 0.0, OBJ)
    # identity stub + no correction -> object never moves
    assert np.allclose(base[:, -1, OBJ], S0[:, OBJ], atol=1e-6)


def test_rest_safety():
    m = IdentityStub()
    S0, V0, Aw, _ = _coast_window([0.0, 0.0, 0.0])      # zero incoming velocity
    for a in (0.0, 0.5, 0.9, 1.0):
        preds = CC.corrected_rollout_batch(m, S0, V0, Aw, "cpu", a, OBJ)
        assert np.allclose(preds[:, -1, OBJ], S0[:, OBJ], atol=1e-6), a


def test_alpha_one_recovers_constant_coast_and_decay_is_bounded():
    m = IdentityStub()
    v0 = [0.02, -0.01, 0.005]
    S0, V0, Aw, real_end = _coast_window(v0)
    err0 = CC.endpoint_rmse(CC.corrected_rollout_batch(m, S0, V0, Aw, "cpu", 0.0, OBJ), real_end, OBJ)
    err1 = CC.endpoint_rmse(CC.corrected_rollout_batch(m, S0, V0, Aw, "cpu", 1.0, OBJ), real_end, OBJ)
    print(f"  constant-coast endpoint RMSE -- alpha0 {err0:.4f}  alpha1 {err1:.6f}")
    assert err1 < 1e-5          # alpha=1 with identity stub exactly reproduces constant coast
    assert err0 > 10 * max(err1, 1e-6)

    # bounded geometric decay for alpha<1: per-step displacement shrinks
    preds = CC.corrected_rollout_batch(m, S0, V0, Aw, "cpu", 0.9, OBJ)
    pos = np.concatenate([S0[:, None, OBJ], preds[:, :, OBJ]], axis=1)
    disp = np.linalg.norm(np.diff(pos, axis=1), axis=2)     # [N,K]
    assert np.all(disp[:, -1] < disp[:, 0] + 1e-8)          # last step displacement <= first (decaying)
    assert np.isfinite(preds).all()


if __name__ == "__main__":
    test_alpha_zero_is_baseline()
    test_rest_safety()
    test_alpha_one_recovers_constant_coast_and_decay_is_bounded()
    print("ok")
