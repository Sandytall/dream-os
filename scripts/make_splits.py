"""Phase 1 -- train / validation / held-out splits.

Splits are made over WHOLE EPISODES (never individual transitions), so no episode
appears in two splits and there is no train/test leakage through near-identical
consecutive states. A disjoint held-out SEED range is also declared and reserved
for Phase 9 generalization, so those initial conditions can never be trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_splits(dataset_id: str, fractions=(0.70, 0.15, 0.15),
                split_seed: int = 0, heldout_seed_start: int = 10_000) -> dict:
    man_path = ROOT / "data" / "manifests" / f"{dataset_id}.json"
    manifest = json.loads(man_path.read_text())
    eps = manifest["episode_list"]
    idx = np.array([e["episode"] for e in eps])
    seeds_used = sorted(e["seed"] for e in eps)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(idx))
    n = len(idx)
    n_tr = int(fractions[0] * n)
    n_va = int(fractions[1] * n)
    train = sorted(idx[perm[:n_tr]].tolist())
    val = sorted(idx[perm[n_tr:n_tr + n_va]].tolist())
    test = sorted(idx[perm[n_tr + n_va:]].tolist())

    splits = {
        "dataset_id": dataset_id,
        "split_seed": split_seed,
        "fractions": list(fractions),
        "split_unit": "episode",
        "no_leakage": "splits are by whole episode; no transition is shared across splits",
        "train": train,
        "val": val,
        "test_heldout": test,
        "counts": {"train": len(train), "val": len(val), "test_heldout": len(test)},
        "train_val_test_seed_range": [min(seeds_used), max(seeds_used)],
        "heldout_generalization_seed_range": [heldout_seed_start, heldout_seed_start + 1000],
        "heldout_generalization_note": "reserved disjoint seeds for Phase 9; not collected yet",
    }
    out = ROOT / "data" / "manifests" / f"{dataset_id}_splits.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"[splits] train/val/test = {splits['counts']['train']}/"
          f"{splits['counts']['val']}/{splits['counts']['test_heldout']} -> {out}")
    return splits


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    args = ap.parse_args()
    make_splits(args.dataset_id)
