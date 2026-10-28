"""Phase 8 -- collect a harder-benchmark dataset (FetchSlide) for the headroom study.

Reuses the Phase 1 collector's storage format and policies verbatim (so
episode_store / retrieval / replay all work unchanged) but mixes scripted-strike
and random episodes for better state-action coverage of the sliding dynamics, and
writes scenario-level train/val/test splits. Phase 1 code is untouched.

Each episode is a distinct scenario (its own seed = its own puck/goal placement),
so splitting by episode is splitting by scenario -- the generalization split the
Phase 8 prompt asks for.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env                       # noqa: E402
from scripts.collect_data import collect_episode, _git_commit, _versions  # noqa: E402

ENV_ID = "FetchSlide-v4"
DATASET_ID = "fetchslide_phase8"
MAX_STEPS = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", type=int, default=140)
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    raw_dir = ROOT / "data" / "raw" / DATASET_ID
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(ENV_ID)

    # policy per episode: scripted block first, then random block (distinct seeds)
    plan = [("scripted", 100 + i) for i in range(args.scripted)] + \
           [("random", 5000 + i) for i in range(args.random)]
    ep_meta = []
    for i, (policy, seed) in enumerate(plan):
        rng = np.random.default_rng(1000 + seed)
        ep = collect_episode(env, seed, policy, rng, MAX_STEPS)
        arrays = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
        arrays["desired_goal"] = ep["desired_goal"]
        np.savez_compressed(raw_dir / f"ep_{i:04d}.npz", **arrays)
        ep_meta.append({"episode": i, "file": f"ep_{i:04d}.npz", "seed": seed,
                        "policy": policy, "length": ep["length"],
                        "success": bool(ep["is_success"].any()),
                        "return": round(float(ep["rewards"].sum()), 3)})
    env.close()

    stamp = datetime.now(timezone.utc)
    manifest = {
        "dataset_id": DATASET_ID, "env": ENV_ID, "policy": "mixed(scripted+random)",
        "timestamp_utc": stamp.isoformat(timespec="seconds"), "git_commit": _git_commit(),
        "versions": _versions(), "max_steps": MAX_STEPS,
        "episodes": len(ep_meta), "action_rng": "numpy.default_rng(1000 + seed)",
        "n_transitions": int(sum(e["length"] for e in ep_meta)),
        "success_rate": round(float(np.mean([e["success"] for e in ep_meta])), 4),
        "n_scripted": args.scripted, "n_random": args.random,
        "obs_dim": int(np.load(raw_dir / "ep_0000.npz")["observation"].shape[1]),
        "episode_list": ep_meta,
    }
    (ROOT / "data" / "manifests" / f"{DATASET_ID}.json").write_text(json.dumps(manifest, indent=2))

    # ---- scenario-level splits (deterministic shuffle of episode ids) ----
    rng = np.random.default_rng(0)
    ids = np.arange(len(ep_meta)); rng.shuffle(ids)
    n = len(ids); n_test = int(args.test_frac * n); n_val = int(args.val_frac * n)
    splits = {"test_heldout": sorted(int(x) for x in ids[:n_test]),
              "val": sorted(int(x) for x in ids[n_test:n_test + n_val]),
              "train": sorted(int(x) for x in ids[n_test + n_val:])}
    (ROOT / "data" / "manifests" / f"{DATASET_ID}_splits.json").write_text(json.dumps(splits, indent=2))

    print(f"[phase8-collect] {DATASET_ID}: {len(ep_meta)} eps "
          f"({args.scripted} scripted + {args.random} random), "
          f"{manifest['n_transitions']} transitions, success_rate={manifest['success_rate']}")
    print(f"[phase8-collect] splits: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test_heldout'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
