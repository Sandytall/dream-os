"""Mechanism test for the Phase 9E absolute-velocity augmentation.

Builds a synthetic COASTING system where the observation exposes object POSITION but
NOT absolute velocity -- the same representational gap Phase 9D attributes to
FetchSlide's gripper-relative velp. A position-only Markov model f(s_t, a_t) cannot
represent momentum, so in open-loop rollout it under-coasts; the augmented model, fed a
finite-difference absolute velocity recomputed from its own predictions, should coast
correctly. If the augmentation is sound, it beats the position-only model by a clear
margin on multi-step coasting error. If this test fails, the Phase 9E experiment is not
trustworthy.

Run: env -u PYTHONPATH ./.venv/bin/python tests/test_aug_velocity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from world_model.dynamics import ResidualDynamics                         # noqa: E402
from world_model import training as T                                     # noqa: E402
from world_model import aug_velocity as AV                                # noqa: E402

OBJ = [0, 1, 2]
DBASE, DACT = 4, 3
MU, C = 0.10, 0.30
T_STEPS, T_PUSH = 16, 4


def make_episode(rng):
    p = rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
    v = np.zeros(3, np.float32)
    obs = [np.concatenate([p, [0.0]]).astype(np.float32)]
    acts = []
    for t in range(T_STEPS):
        a = rng.uniform(-1, 1, size=3).astype(np.float32) if t < T_PUSH else np.zeros(3, np.float32)
        v = v * (1 - MU) + C * a          # momentum + impulse - friction
        p = p + v                          # coasts after the push stops
        obs.append(np.concatenate([p, [0.0]]).astype(np.float32))
        acts.append(a)
    return np.asarray(obs, np.float32), np.asarray(acts, np.float32)


def make_set(n, seed):
    rng = np.random.default_rng(seed)
    return [make_episode(rng) for _ in range(n)]


def flat(eps):
    S, A, Sn = [], [], []
    for obs, act in eps:
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(act)
    return (np.concatenate(S), np.concatenate(A), np.concatenate(Sn))


def multistep_obj_rmse(roll_fn, eps, h):
    errs, base = [], []
    for obs, act in eps:
        preds = roll_fn(obs, act)                 # [T, DBASE]
        errs.append(np.sqrt(np.mean((preds[h - 1][OBJ] - obs[h][OBJ]) ** 2)))
        base.append(np.sqrt(np.mean((obs[0][OBJ] - obs[h][OBJ]) ** 2)))
    return float(np.mean(errs)), float(np.mean(base))


def test_aug_velocity_coasts_better_than_position_only():
    device = "cpu"
    train_eps, val_eps, test_eps = make_set(80, 1), make_set(16, 2), make_set(24, 3)
    K, H = 5, 10

    # position-only baseline (rollout-K5), identical objective
    stats = T.compute_stats(*flat(train_eps))
    torch.manual_seed(0)
    base = ResidualDynamics(DBASE, DACT, stats)
    T.train_rollout(base, train_eps, val_eps, device, horizon=K, epochs=200,
                    patience=25, seed=0, obj_cols=tuple(OBJ))
    base.eval()

    # augmented-velocity model, SAME objective + one added input
    astats = AV.aug_stats(train_eps, OBJ)
    torch.manual_seed(0)
    aug = AV.AugVelDynamics(DBASE, DACT, OBJ, astats)
    base_s_std = flat(train_eps)[0].std(0)
    AV.train_aug_rollout(aug, train_eps, val_eps, device, base_s_std, horizon=K,
                         epochs=200, patience=25, seed=0, obj_cols=tuple(OBJ))
    aug.eval()

    base_err, persist = multistep_obj_rmse(
        lambda obs, act: T.rollout(base, obs[0], act, device), test_eps, H)
    aug_err, _ = multistep_obj_rmse(
        lambda obs, act: AV.aug_rollout(aug, obs[0], np.zeros(3), act, device, OBJ), test_eps, H)

    print(f"  h{H} coasting object RMSE -- persistence {persist:.4f}  "
          f"position-only {base_err:.4f}  aug-velocity {aug_err:.4f}")
    # the augmentation must coast; position-only cannot represent momentum
    assert aug_err < 0.5 * base_err, (aug_err, base_err)
    assert aug_err < 0.5 * persist, (aug_err, persist)


if __name__ == "__main__":
    test_aug_velocity_coasts_better_than_position_only()
    print("ok")
