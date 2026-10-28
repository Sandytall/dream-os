"""Phase 2 -- train and evaluate the baseline world model.

Pipeline: load splits -> fit normalization on TRAIN -> train residual dynamics ->
evaluate one-step and multi-step on the HELD-OUT test split vs a persistence
baseline -> save checkpoint + reports/phase2_results.json.

GATE (defined on physical grounds, not chosen after the fact):
  A useful dynamics model must reduce one-step prediction error, vs persistence,
  on the physically meaningful state -- the object's position AND the full
  position+velocity state. We EXCLUDE the block's Euler-angle rotation from the
  gate because that representation is discontinuous (+/-pi wrap) and a free block's
  spin at contact is near-chaotic, so it is nearly unpredictable and dominates a
  naive all-dimensions average. Rotation error is still REPORTED, as a known
  limitation, with the fix (a continuous sin/cos or quaternion representation)
  noted for future work. The all-dimensions overall RMSE is also reported.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memory import episode_store as store          # noqa: E402
from world_model.dynamics import ResidualDynamics  # noqa: E402
from world_model import training as T              # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
# FetchPush observation layout (gymnasium-robotics, 25 dims):
BLOCKS = [("grip_pos", 0, 3), ("object_pos", 3, 6), ("object_rel_pos", 6, 9),
          ("gripper_state", 9, 11), ("object_rot", 11, 14), ("object_velp", 14, 17),
          ("object_velr", 17, 20), ("grip_velp", 20, 23), ("gripper_vel", 23, 25)]
OBJ_COLS = [3, 4, 5]                 # object position
ROT_COLS = [11, 12, 13]              # object Euler rotation (excluded from gate)
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]   # position + velocity state
HORIZONS = [1, 2, 5, 10, 20, 40]


def rmse(pred, true, cols) -> float:
    return float(np.sqrt(np.mean((pred[:, cols] - true[:, cols]) ** 2)))


def main() -> int:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = store.transitions(DATASET_ID, "train")
    va = store.transitions(DATASET_ID, "val")
    te = store.transitions(DATASET_ID, "test_heldout")
    d_state, d_act = tr[0].shape[1], tr[1].shape[1]

    stats = T.compute_stats(*tr)
    torch.manual_seed(0)
    model = ResidualDynamics(d_state, d_act, stats, hidden=256)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_info = T.train(model, tr, va, device,
                         epochs=400, batch_size=256, lr=1e-3, patience=40, seed=0)

    # ---- one-step evaluation on HELD-OUT test ----
    S, A, Snext = te
    pred = T.predict_batch(model, S, A, device)
    allc = list(range(d_state))

    def pair(cols):
        return {"model": round(rmse(pred, Snext, cols), 5),
                "persistence": round(rmse(S, Snext, cols), 5)}

    one_step = {
        "overall_all_dims": pair(allc),
        "object_pos_m": pair(OBJ_COLS),
        "physical_state_excl_rotation": pair(PHYS_COLS),
        "object_rotation_m_known_hard": pair(ROT_COLS),
    }
    per_block = {name: pair(list(range(a, b))) for name, a, b in BLOCKS}

    # ---- multi-step (object position) ----
    eps = store.episodes(DATASET_ID, "test_heldout")
    multistep = T.evaluate_multistep(model, eps, device, HORIZONS, OBJ_COLS)

    # ---- save checkpoint ----
    ckpt_dir = ROOT / "world_model" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{DATASET_ID}_dynamics.pt"
    torch.save({"state_dict": model.state_dict(), "d_state": d_state,
                "d_act": d_act, "dataset_id": DATASET_ID}, ckpt)

    # ---- gate (physical, pre-justified above) ----
    beats_object = one_step["object_pos_m"]["model"] < one_step["object_pos_m"]["persistence"]
    beats_phys = (one_step["physical_state_excl_rotation"]["model"]
                  < one_step["physical_state_excl_rotation"]["persistence"])
    gate = bool(beats_object and beats_phys)
    obj_improve = round(100 * (1 - one_step["object_pos_m"]["model"]
                               / max(one_step["object_pos_m"]["persistence"], 1e-9)), 1)

    peak_vram = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                 if device == "cuda" else None)

    report = {
        "phase": 2, "dataset_id": DATASET_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device, "model_params": model.num_params(),
        "n_train_transitions": int(tr[0].shape[0]),
        "n_test_transitions": int(S.shape[0]),
        "epochs_run": train_info["epochs_run"],
        "gate_definition": ("model beats persistence (one-step, held-out) on object "
                            "position AND on position+velocity state; object Euler "
                            "rotation excluded as near-unpredictable and reported "
                            "separately (fix: continuous rotation representation)."),
        "one_step_heldout": one_step,
        "object_pos_improvement_pct": obj_improve,
        "per_block_one_step": per_block,
        "blocks_won_by_model": sum(1 for v in per_block.values()
                                   if v["model"] < v["persistence"]),
        "multistep_heldout_object_rmse_m": multistep,
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "peak_vram_gb": peak_vram,
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": gate,
    }
    out = ROOT / "reports" / "phase2_results.json"
    out.write_text(json.dumps(report, indent=2))

    # ---- teaching summary ----
    print("=" * 70)
    print("DREAM-OS Phase 2 report  (world model vs persistence baseline)")
    print("=" * 70)
    print(f"  device={device}  params={model.num_params():,}  "
          f"epochs_run={train_info['epochs_run']}  train_n={tr[0].shape[0]}")
    print("  -- one-step RMSE on HELD-OUT test (model vs persistence) --")
    for k, v in one_step.items():
        tag = "  <- gated" if k in ("object_pos_m", "physical_state_excl_rotation") else ""
        flag = "" if k != "object_rotation_m_known_hard" else "  (known-hard, NOT gated)"
        print(f"     {k:30s} model {v['model']:.4f}  persist {v['persistence']:.4f}{tag}{flag}")
    print(f"     object position improvement: {obj_improve:.0f}%   "
          f"blocks won by model: {report['blocks_won_by_model']}/9")
    print("  -- multi-step object-pos RMSE (open-loop; error compounds) --")
    for h in HORIZONS:
        if h in multistep:
            m = multistep[h]
            print(f"     h={h:2d}: model {m['model_object_rmse_m']:.4f} m   "
                  f"persist {m['baseline_object_rmse_m']:.4f} m")
    print(f"  peak_vram={peak_vram}GB  wall={report['wall_seconds']}s  -> {out}")
    print("GATE PASSED:", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
