"""Phase 3 orchestrator: retrieve start states -> generate counterfactual dreams
-> store with provenance -> verify reproducibility + traceability (the gate).

No MuJoCo here: dreams are the world model's PREDICTIONS. Phase 4 checks them
against real simulator physics.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memory import retrieval                     # noqa: E402
from dreaming import counterfactuals as cf       # noqa: E402
from dreaming import dream_store as ds            # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
CKPT = ROOT / "world_model" / "checkpoints" / f"{DATASET_ID}_dynamics.pt"
BATCH_ID = "phase3_fetchpush"
N_STARTS = 6
HORIZON = 20
OBJ = slice(3, 6)
# Real FetchPush states live well within ~2 in every dim; a predicted trajectory
# that exceeds this bound has diverged (out-of-distribution) and is flagged, not
# trusted. (§8: mark OOD dreams explicitly.)
PLAUSIBLE_BOUND = 10.0


def _xy_disp(states):
    return float(np.linalg.norm((states[-1][OBJ] - states[0][OBJ])[:2]))


def build_dreams(model, device, model_sha):
    starts = retrieval.sample_start_states(DATASET_ID, "test_heldout", n=N_STARTS, seed=0)
    dreams = []
    for si, ref in enumerate(starts):
        goal = ref["desired_goal"]
        for bname, spec in cf.BRANCHES.items():
            seed = 100 + si                      # deterministic per (start, branch)
            states, actions = cf.dream(model, ref["observation"], goal, spec,
                                       HORIZON, device, seed=seed)
            max_abs = float(np.max(np.abs(states)))
            plausible = bool(np.isfinite(max_abs) and max_abs < PLAUSIBLE_BOUND)
            dreams.append({
                "dream_id": f"d{si:02d}_{bname}",
                "status": "predicted",
                "plausible": plausible,          # False => diverged / out-of-distribution
                "max_abs_predicted_state": round(max_abs, 4) if np.isfinite(max_abs) else None,
                "start_ref": {k: ref[k] for k in ("dataset_id", "split", "episode", "t")},
                "start_obs_sha": hashlib.sha256(ref["observation"].tobytes()).hexdigest()[:16],
                "branch": {"name": bname, **spec, "seed": seed},
                "horizon": HORIZON,
                "model": {"checkpoint": str(CKPT.relative_to(ROOT)), "sha256_16": model_sha},
                "predicted_object_xy_disp_m": round(_xy_disp(states), 4) if plausible else None,
                "predicted_states": states,
                "actions": actions,
            })
    return dreams, starts


def main() -> int:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = cf.load_model(CKPT, device=device)
    model_sha = ds.sha256_file(CKPT)

    dreams, starts = build_dreams(model, device, model_sha)
    manifest = ds.save_batch(BATCH_ID, dreams)

    # ---- reproducibility: regenerate each dream, compare to stored ----
    max_repro_err = 0.0
    for dr in dreams:
        si = int(dr["dream_id"][1:3])
        ref = starts[si]
        spec = cf.BRANCHES[dr["branch"]["name"]]
        states2, _ = cf.dream(model, ref["observation"], ref["desired_goal"],
                              spec, HORIZON, device, seed=dr["branch"]["seed"])
        stored = ds.load_dream_arrays(BATCH_ID, dr["dream_id"])["predicted_states"]
        max_repro_err = max(max_repro_err, float(np.max(np.abs(states2 - stored))))

    # ---- traceability: every dream fully provenanced ----
    needed = ("start_ref", "model", "branch", "status", "horizon")
    man = json.loads(manifest.read_text())
    traceable = all(all(k in d for k in needed)
                    and d["model"].get("sha256_16") and "episode" in d["start_ref"]
                    for d in man["dreams"])

    tol = 1e-5
    gate = bool(max_repro_err <= tol and traceable)

    # ---- plausible vs diverged, and do the counterfactuals differ? ----
    plausible = [d for d in man["dreams"] if d["plausible"]]
    diverged = [d for d in man["dreams"] if not d["plausible"]]

    def branch_mean_disp(b):
        vals = [d["predicted_object_xy_disp_m"] for d in plausible
                if d["branch"]["name"] == b and d["predicted_object_xy_disp_m"] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    branch_means = {b: branch_mean_disp(b) for b in cf.BRANCHES}
    diverged_by_branch = {b: sum(1 for d in diverged if d["branch"]["name"] == b)
                          for b in cf.BRANCHES}

    report = {
        "phase": 3, "batch_id": BATCH_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": device, "n_starts": N_STARTS, "branches": list(cf.BRANCHES),
        "horizon": HORIZON, "n_dreams": len(dreams),
        "model_checkpoint_sha256_16": model_sha,
        "reproducibility_max_abs_err": max_repro_err,
        "reproducibility_tol": tol,
        "all_dreams_traceable": traceable,
        "n_plausible": len(plausible),
        "n_diverged_out_of_distribution": len(diverged),
        "diverged_by_branch": diverged_by_branch,
        "predicted_obj_xy_disp_by_branch_m_plausible_only": branch_means,
        "manifest": str(manifest.relative_to(ROOT)),
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": gate,
    }
    out = ROOT / "reports" / "phase3_results.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 66)
    print("DREAM-OS Phase 3 report  (counterfactual dream generation)")
    print("=" * 66)
    print(f"  device={device}  model_sha={model_sha}  "
          f"starts={N_STARTS} x branches={len(cf.BRANCHES)} = {len(dreams)} dreams, "
          f"horizon={HORIZON}")
    print(f"  reproducibility max_abs_err = {max_repro_err:.2e}  (tol {tol})  "
          f"-> {'identical' if max_repro_err == 0 else 'within tol' if gate else 'FAIL'}")
    print(f"  all dreams traceable (start_ref + model_sha + branch) : {traceable}")
    print(f"  plausible dreams: {len(plausible)}/{len(dreams)}   "
          f"diverged (out-of-distribution, flagged): {len(diverged)}")
    print(f"  diverged by branch: {diverged_by_branch}")
    print("  predicted object xy-displacement by branch (plausible dreams only):")
    for b, v in branch_means.items():
        vs = f"{v:.3f} m" if v is not None else "n/a (all diverged)"
        print(f"     {b:12s} mean {vs}")
    print(f"  wall={report['wall_seconds']}s  -> {out}")
    print("GATE PASSED:", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
