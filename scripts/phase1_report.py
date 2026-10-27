"""Phase 1 orchestrator: collect -> verify (gate) -> split -> report.

Runs a small reproducible FetchPush-v4 collection, verifies integrity + replay,
builds train/val/held-out splits, and writes reports/phase1_results.json with
provenance and resource use. Exit code reflects the Phase-1 gate.
"""
from __future__ import annotations

import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import collect_data
import verify_dataset
import make_splits

DATASET_ID = "fetchpush_scripted_phase1"


def main() -> int:
    t0 = time.time()

    collect_data.run_collection(
        env_id="FetchPush-v4", episodes=60, max_steps=50,
        policy="scripted", base_seed=0, name=DATASET_ID)

    verification = verify_dataset.verify_dataset(DATASET_ID, n_check=10)
    splits = make_splits.make_splits(DATASET_ID)

    report = {
        "phase": 1,
        "dataset_id": DATASET_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "verification": verification,
        "splits": splits["counts"],
        "wall_seconds": round(time.time() - t0, 2),
        "peak_ram_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2),
        "gate_passed": bool(verification["passed"]
                            and sum(splits["counts"].values()) == 60),
    }
    out = ROOT / "reports" / "phase1_results.json"
    out.write_text(json.dumps(report, indent=2))

    v = verification
    print("=" * 60)
    print("DREAM-OS Phase 1 report")
    print("=" * 60)
    print(f"  dataset        : {DATASET_ID}")
    print(f"  success_rate   : {v['success_rate']}  "
          f"(mean final obj->goal dist {v['mean_final_object_goal_dist_m']} m, "
          f"min {v['min_final_object_goal_dist_m']} m)")
    print(f"  integrity_ok   : {v['integrity_ok']}  "
          f"(finite={v['all_finite']}, bounds={v['action_bounds_ok']}, "
          f"shapes={v['shapes_consistent']}, qpos_dim={v['qpos_dim']})")
    print(f"  replay_ok      : {v['replay_ok']}  "
          f"(max_abs_err={v['max_replay_abs_err']:.2e} <= tol {v['replay_tolerance']})")
    print(f"  splits         : {report['splits']}")
    print(f"  wall={report['wall_seconds']}s  peak_ram={report['peak_ram_gb']}GB")
    print(f"  -> {out}")
    print("GATE PASSED:", report["gate_passed"])
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
