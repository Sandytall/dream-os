"""Dream store + provenance (Component C, dream side).

Dreams are PREDICTED trajectories and are stored separately from observed/verified
data, each with a complete provenance trail so it can be traced and regenerated:
  * start_ref  -- the exact recorded transition it began from,
  * model      -- checkpoint path + content hash that produced it,
  * branch     -- the counterfactual action plan + seed,
  * status     -- "predicted" (never "observed"/"verified" at this stage).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path, n: int = 16) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:n]


def save_batch(batch_id: str, dreams: list[dict]) -> Path:
    """dreams: list of dicts with arrays `predicted_states`,`actions` + metadata."""
    ddir = ROOT / "data" / "dreams" / batch_id
    ddir.mkdir(parents=True, exist_ok=True)
    meta = []
    for dr in dreams:
        np.savez_compressed(
            ddir / f"{dr['dream_id']}.npz",
            predicted_states=dr["predicted_states"],
            actions=dr["actions"])
        meta.append({k: v for k, v in dr.items()
                     if k not in ("predicted_states", "actions")})
    man = ROOT / "data" / "manifests" / f"dreams_{batch_id}.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(
        {"batch_id": batch_id, "n_dreams": len(meta),
         "status": "predicted", "dreams": meta}, indent=2))
    return man


def load_dream_arrays(batch_id: str, dream_id: str):
    return np.load(ROOT / "data" / "dreams" / batch_id / f"{dream_id}.npz")
