"""Phase 4 orchestrator: verify dreams against the reference MuJoCo simulator.

For every stored dream: reconstruct its real start, replay the dream's actions in
real physics, compare predicted vs real, classify supported/contradicted/unresolved
with explicit tolerances, and RETAIN both the prediction and the replayed outcome.

Gate (verifier correctness, not model quality):
  * starts reconstruct exactly (foundation is sound),
  * every dream gets a classification,
  * known-garbage (diverged, out-of-distribution) dreams are correctly contradicted.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env           # noqa: E402
from dreaming import dream_store as ds                  # noqa: E402
from verification import replay as vr                   # noqa: E402
from verification import metrics as vm                  # noqa: E402

BATCH_ID = "phase3_fetchpush"
OBJ = slice(3, 6)


def main() -> int:
    t0 = time.time()
    man = json.loads((ROOT / "data" / "manifests" / f"dreams_{BATCH_ID}.json").read_text())
    dreams = man["dreams"]

    env = make_env("FetchPush-v4")
    ver_dir = ROOT / "data" / "verification" / BATCH_ID
    ver_dir.mkdir(parents=True, exist_ok=True)

    records = []
    max_recon_err = 0.0
    for d in dreams:
        arr = ds.load_dream_arrays(BATCH_ID, d["dream_id"])
        pred = arr["predicted_states"]
        actions = arr["actions"]
        ref = d["start_ref"]
        real, recon_err = vr.reconstruct_and_replay(
            env, ref["dataset_id"], ref["episode"], ref["t"], actions)
        max_recon_err = max(max_recon_err, recon_err)

        per_step = vm.object_discrepancy(pred, real)
        label, stats = vm.classify(per_step, d["plausible"], recon_err)

        # retain the replayed real trajectory (ground-truth record)
        np.savez_compressed(ver_dir / f"{d['dream_id']}.npz", real_states=real)

        pf = pred[-1][OBJ]
        rf = real[-1][OBJ]
        records.append({
            "dream_id": d["dream_id"], "branch": d["branch"]["name"],
            "plausible": d["plausible"], "classification": label,
            "reconstruction_err_m": round(recon_err, 6),
            "predicted_object_final": [round(float(x), 3) for x in pf],
            "real_object_final": [round(float(x), 3) for x in rf],
            **stats,
        })
    env.close()

    # ---- aggregate ----
    labels = [r["classification"] for r in records]
    counts = {k: labels.count(k) for k in ("supported", "contradicted", "unresolved")}
    by_branch = {}
    for r in records:
        by_branch.setdefault(r["branch"], {"supported": 0, "contradicted": 0, "unresolved": 0})
        by_branch[r["branch"]][r["classification"]] += 1
    # mean object discrepancy among plausible dreams that had a numeric comparison
    plaus_disc = [r["mean_obj_disc_m"] for r in records
                  if r["plausible"] and "mean_obj_disc_m" in r]
    diverged = [r for r in records if not r["plausible"]]
    diverged_all_contradicted = all(r["classification"] == "contradicted" for r in diverged)

    # ---- save verification records (ground-truth, separate from predictions) ----
    ver_man = ROOT / "data" / "manifests" / f"verification_{BATCH_ID}.json"
    ver_man.write_text(json.dumps({
        "batch_id": BATCH_ID, "kind": "verification_records",
        "simulator": "MuJoCo via FetchPush-v4",
        "tolerances_m": {"support": vm.SUPPORT_TOL_M, "contradict": vm.CONTRADICT_TOL_M,
                         "reconstruction": vm.RECON_TOL_M},
        "records": records}, indent=2))

    gate = bool(max_recon_err < vm.RECON_TOL_M
                and len(records) == len(dreams)
                and diverged_all_contradicted)

    report = {
        "phase": 4, "batch_id": BATCH_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_dreams_verified": len(records),
        "max_reconstruction_err_m": round(max_recon_err, 8),
        "classification_counts": counts,
        "classification_by_branch": by_branch,
        "verification_agreement_rate": round(counts["supported"] / len(records), 3),
        "mean_object_discrepancy_plausible_m": (round(float(np.mean(plaus_disc)), 4)
                                                if plaus_disc else None),
        "diverged_all_contradicted": diverged_all_contradicted,
        "tolerances_m": {"support": vm.SUPPORT_TOL_M, "contradict": vm.CONTRADICT_TOL_M},
        "records_manifest": str(ver_man.relative_to(ROOT)),
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": gate,
    }
    (ROOT / "reports" / "phase4_results.json").write_text(json.dumps(report, indent=2))

    print("=" * 68)
    print("DREAM-OS Phase 4 report  (verify dreams vs MuJoCo)")
    print("=" * 68)
    print(f"  dreams verified: {len(records)}   "
          f"start reconstruction max err: {max_recon_err:.2e} m "
          f"({'EXACT/sound' if max_recon_err < vm.RECON_TOL_M else 'FAILED'})")
    print(f"  classification: {counts}   "
          f"agreement(supported/total)={report['verification_agreement_rate']}")
    print("  by branch (supported / contradicted / unresolved):")
    for b, c in by_branch.items():
        print(f"     {b:12s} {c['supported']} / {c['contradicted']} / {c['unresolved']}")
    print(f"  mean object discrepancy (plausible dreams): "
          f"{report['mean_object_discrepancy_plausible_m']} m")
    print(f"  diverged (random) dreams all correctly contradicted: "
          f"{diverged_all_contradicted}")
    print(f"  tolerances: supported<={vm.SUPPORT_TOL_M} m, contradicted>={vm.CONTRADICT_TOL_M} m")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase4_results.json")
    print("GATE PASSED:", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
