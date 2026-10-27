"""Phase 1 -- dataset integrity + replay verification (the Phase-1 gate).

Two things must hold before Phase 2:
  1. Integrity: every stored array is finite, shapes are consistent, actions are
     within bounds.
  2. Replay: re-running a sampled episode from its seed with the SAVED actions
     reproduces the stored observation trajectory to tolerance -- i.e. the data is
     reproducible, not a one-off artifact.

Reports measured numbers only; on failure it reports the failure, it does not hide it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env, obs_parts  # noqa: E402


def _manifest_path(dataset_id: str) -> Path:
    return ROOT / "data" / "manifests" / f"{dataset_id}.json"


def verify_dataset(dataset_id: str, n_check: int = 8, tol: float = 1e-4,
                   sample_seed: int = 0) -> dict:
    manifest = json.loads(_manifest_path(dataset_id).read_text())
    raw_dir = ROOT / "data" / "raw" / dataset_id
    eps = manifest["episode_list"]
    seed_of = {e["episode"]: e["seed"] for e in eps}

    # ---- integrity over ALL episodes ----
    all_finite = True
    bounds_ok = True
    shapes_ok = True
    qpos_dims = set()
    final_dists = []
    for e in eps:
        d = np.load(raw_dir / e["file"])
        obs, act = d["observation"], d["actions"]
        all_finite &= bool(np.isfinite(obs).all() and np.isfinite(act).all()
                           and np.isfinite(d["qpos"]).all())
        bounds_ok &= bool((act >= -1 - 1e-6).all() and (act <= 1 + 1e-6).all())
        shapes_ok &= bool(obs.shape[0] == act.shape[0] + 1 == d["qpos"].shape[0])
        qpos_dims.add(int(d["qpos"].shape[1]))
        final_dists.append(float(np.linalg.norm(d["achieved_goal"][-1] - d["desired_goal"])))

    # ---- replay a sample from seed + saved actions ----
    env = make_env(manifest["env"])
    rng = np.random.default_rng(sample_seed)
    pick = sorted(rng.choice(len(eps), size=min(n_check, len(eps)), replace=False).tolist())
    max_replay_err = 0.0
    per_episode = []
    for i in pick:
        d = np.load(raw_dir / eps[i]["file"])
        obs_saved, actions = d["observation"], d["actions"]
        obs, _ = env.reset(seed=int(seed_of[i]))
        traj = [obs_parts(obs)[0]]
        for a in actions:
            obs, *_ = env.step(a)
            traj.append(obs_parts(obs)[0])
        err = float(np.max(np.abs(np.asarray(traj) - obs_saved)))
        max_replay_err = max(max_replay_err, err)
        per_episode.append({"episode": i, "replay_max_abs_err": err})
    env.close()

    replay_ok = max_replay_err <= tol
    integrity_ok = all_finite and bounds_ok and shapes_ok and len(qpos_dims) == 1

    result = {
        "dataset_id": dataset_id,
        "episodes_checked_integrity": len(eps),
        "episodes_checked_replay": len(pick),
        "all_finite": all_finite,
        "action_bounds_ok": bounds_ok,
        "shapes_consistent": shapes_ok,
        "qpos_dim": sorted(qpos_dims),
        "max_replay_abs_err": max_replay_err,
        "replay_tolerance": tol,
        "replay_ok": replay_ok,
        "integrity_ok": integrity_ok,
        "success_rate": manifest["success_rate"],
        "mean_final_object_goal_dist_m": round(float(np.mean(final_dists)), 4),
        "min_final_object_goal_dist_m": round(float(np.min(final_dists)), 4),
        "replay_samples": per_episode,
        "passed": bool(replay_ok and integrity_ok),
    }
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("--n-check", type=int, default=8)
    args = ap.parse_args()
    r = verify_dataset(args.dataset_id, n_check=args.n_check)
    print(json.dumps(r, indent=2))
    print("VERIFY:", "PASS" if r["passed"] else "FAIL")
    raise SystemExit(0 if r["passed"] else 1)
