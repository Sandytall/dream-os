"""Unit/integration tests for the world model, incl. Phase 9B rollout training.

Run: env -u PYTHONPATH ./.venv/bin/python -m pytest tests/test_world_model.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from world_model.dynamics import ResidualDynamics          # noqa: E402
from world_model import training as T                        # noqa: E402


def _linear_episodes(n_eps=40, horizon=20, d=5, da=2, seed=0):
    """A fully learnable deterministic system: s_{t+1} = s_t + 0.5 * B @ a_t.
    Multi-step prediction is exactly representable, so a good trainer should drive
    long-horizon rollout error close to zero."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, da)).astype(np.float32)
    eps = []
    for _ in range(n_eps):
        s = rng.standard_normal(d).astype(np.float32)
        obs = [s.copy()]; acts = []
        for _t in range(horizon):
            a = rng.uniform(-1, 1, size=da).astype(np.float32)
            s = s + 0.5 * (B @ a)
            obs.append(s.copy()); acts.append(a)
        eps.append((np.asarray(obs, np.float32), np.asarray(acts, np.float32)))
    return eps, B


def test_make_windows_shapes():
    eps, _ = _linear_episodes(n_eps=3, horizon=10, d=4, da=2)
    S0, Aw, St = T.make_windows(eps, horizon=5)
    assert S0.shape == (3 * (10 - 5 + 1), 4)
    assert Aw.shape == (S0.shape[0], 5, 2)
    assert St.shape == (S0.shape[0], 5, 4)
    # window target is the true next-state sequence
    assert np.allclose(St[0, 0], eps[0][0][1])


def test_rollout_training_learns_multistep_linear():
    eps, _ = _linear_episodes(n_eps=60, horizon=20, d=5, da=2, seed=1)
    train_eps, val_eps = eps[:50], eps[50:]
    S = np.concatenate([o[:-1] for o, _ in train_eps]).astype(np.float32)
    A = np.concatenate([a for _, a in train_eps]).astype(np.float32)
    Sn = np.concatenate([o[1:] for o, _ in train_eps]).astype(np.float32)
    stats = T.compute_stats(S, A, Sn)
    obj = tuple(range(5))

    model = ResidualDynamics(5, 2, stats, hidden=64)
    info = T.train_rollout(model, train_eps, val_eps, "cpu", horizon=8,
                           epochs=250, patience=40, seed=0, obj_cols=obj)
    ms = T.evaluate_multistep(model, val_eps, "cpu", [1, 10, 20], obj)
    assert np.isfinite(info["best_val_obj_rollout_rmse"])
    # one-step is accurate on a learnable system
    assert ms[1]["model_object_rmse_m"] < 0.05, ms
    # long-horizon open-loop prediction is far better than persistence (>=5x)
    assert ms[20]["model_object_rmse_m"] < 0.2 * ms[20]["baseline_object_rmse_m"], ms


if __name__ == "__main__":
    test_make_windows_shapes()
    test_rollout_training_learns_multistep_linear()
    print("ok")
