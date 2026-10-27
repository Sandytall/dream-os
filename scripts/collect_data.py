"""Phase 1 -- Experience Collector (Component A).

Collects reproducible trajectories from the selected benchmark and stores each
episode as a compressed NPZ plus a dataset manifest carrying full provenance
(seeds, versions, git commit, policy). It records exactly what happened -- it
does NOT filter for successes.

Each stored episode keeps both:
  * the learning observation (and achieved/desired goal), and
  * the raw MuJoCo state (qpos, qvel) at every step, for later exact replay.

Usage:
    python scripts/collect_data.py --episodes 60 --policy scripted
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env, get_sim_state, obs_parts  # noqa: E402


# --------------------------------------------------------------------------- #
# provenance helpers
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uninitialised"


def _versions() -> dict:
    import gymnasium
    import mujoco
    import torch
    try:
        import gymnasium_robotics
        gr = gymnasium_robotics.__version__
    except Exception:
        gr = None
    return {
        "gymnasium": gymnasium.__version__,
        "gymnasium_robotics": gr,
        "mujoco": mujoco.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def scripted_push_action(obs, rng, noise: float = 0.08) -> np.ndarray:
    """Heuristic FetchPush controller -- NOT optimal, just informative.

    Stage the gripper just behind the block along the block->goal line, then push
    through it toward the goal. Gaussian noise adds exploration/state coverage.
    Returns a 4-D action clipped to [-1, 1].
    """
    o, ag, dg = obs_parts(obs)
    grip = o[0:3]
    obj = ag if ag is not None else o[3:6]
    goal = dg if dg is not None else obj
    to_goal = goal - obj
    push_dir = to_goal / (np.linalg.norm(to_goal) + 1e-8)
    behind = obj - 0.055 * push_dir
    target = behind if np.linalg.norm(grip - behind) > 0.03 else goal
    a = np.zeros(4, dtype=np.float32)
    a[:3] = np.clip(12.0 * (target - grip), -1.0, 1.0)
    a[:3] = np.clip(a[:3] + rng.normal(0.0, noise, size=3), -1.0, 1.0)
    return a


def random_action(action_space, rng) -> np.ndarray:
    return rng.uniform(action_space.low, action_space.high).astype(np.float32)


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def collect_episode(env, seed: int, policy: str, rng, max_steps: int) -> dict:
    obs, info = env.reset(seed=seed)
    o0, ag0, dg0 = obs_parts(obs)
    qpos0, qvel0 = get_sim_state(env)

    observation = [o0]
    achieved = [ag0 if ag0 is not None else np.zeros(3, np.float32)]
    qpos, qvel = [qpos0], [qvel0]
    actions, rewards, terms, truncs, succs = [], [], [], [], []

    steps = 0
    while True:
        if policy == "scripted":
            a = scripted_push_action(obs, rng)
        else:
            a = random_action(env.action_space, rng)
        obs, r, term, trunc, info = env.step(a)
        o, ag, _ = obs_parts(obs)
        qp, qv = get_sim_state(env)

        observation.append(o)
        achieved.append(ag if ag is not None else np.zeros(3, np.float32))
        qpos.append(qp)
        qvel.append(qv)
        actions.append(a)
        rewards.append(float(r))
        terms.append(bool(term))
        truncs.append(bool(trunc))
        succs.append(bool(info.get("is_success", 0.0)))

        steps += 1
        if term or trunc or steps >= max_steps:
            break

    return {
        "seed": seed,
        "length": steps,
        "desired_goal": dg0 if dg0 is not None else np.zeros(3, np.float32),
        "observation": np.asarray(observation, np.float32),   # [T+1, obs]
        "achieved_goal": np.asarray(achieved, np.float32),     # [T+1, 3]
        "qpos": np.asarray(qpos, np.float32),                  # [T+1, nq]
        "qvel": np.asarray(qvel, np.float32),                  # [T+1, nv]
        "actions": np.asarray(actions, np.float32),            # [T, act]
        "rewards": np.asarray(rewards, np.float32),            # [T]
        "terminated": np.asarray(terms, bool),
        "truncated": np.asarray(truncs, bool),
        "is_success": np.asarray(succs, bool),
    }


def run_collection(env_id="FetchPush-v4", episodes=60, max_steps=50,
                   policy="scripted", base_seed=0, name=None) -> Path:
    stamp = datetime.now(timezone.utc)
    dataset_id = name or f"{env_id}_{policy}_{stamp.strftime('%Y%m%dT%H%M%S')}"
    raw_dir = ROOT / "data" / "raw" / dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id)
    ep_meta = []
    for i in range(episodes):
        seed = base_seed + i
        # deterministic, seed-specific RNG for action noise -> fully reproducible
        rng = np.random.default_rng(1000 + seed)
        ep = collect_episode(env, seed, policy, rng, max_steps)
        arrays = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
        arrays["desired_goal"] = ep["desired_goal"]
        np.savez_compressed(raw_dir / f"ep_{i:04d}.npz", **arrays)
        ep_meta.append({
            "episode": i, "file": f"ep_{i:04d}.npz", "seed": seed,
            "length": ep["length"], "success": bool(ep["is_success"].any()),
            "return": round(float(ep["rewards"].sum()), 3),
        })
    env.close()

    manifest = {
        "dataset_id": dataset_id,
        "env": env_id,
        "policy": policy,
        "timestamp_utc": stamp.isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "base_seed": base_seed,
        "action_rng": "numpy.default_rng(1000 + seed)",
        "episodes": episodes,
        "max_steps": max_steps,
        "n_transitions": int(sum(e["length"] for e in ep_meta)),
        "success_rate": round(float(np.mean([e["success"] for e in ep_meta])), 4),
        "obs_dim": int(np.load(raw_dir / "ep_0000.npz")["observation"].shape[1]),
        "episode_list": ep_meta,
    }
    man_dir = ROOT / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    man_path = man_dir / f"{dataset_id}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"[collect] {dataset_id}: {episodes} eps, {manifest['n_transitions']} "
          f"transitions, success_rate={manifest['success_rate']}")
    print(f"[collect] raw -> {raw_dir}")
    print(f"[collect] manifest -> {man_path}")
    return man_path


def main() -> int:
    ap = argparse.ArgumentParser(description="DREAM-OS Phase 1 experience collector")
    ap.add_argument("--env", default="FetchPush-v4")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    run_collection(args.env, args.episodes, args.max_steps, args.policy,
                   args.base_seed, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
