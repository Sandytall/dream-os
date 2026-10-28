"""Phase 9A -- characterize the CURRENT world model before changing anything.

The Phase 9 prompt (section 4) requires measuring where the existing model
succeeds, where it fails, and how its error grows with the prediction horizon --
BEFORE selecting a new architecture. This script does exactly that on the
verified FetchSlide benchmark, using the existing (one-step-loss-trained)
ResidualDynamics unchanged. It reports:

  * one-step error broken out by physical quantity (object pos / rotation /
    relative velocity / gripper / full physical state), vs a persistence baseline;
  * multi-step open-loop object error at several horizons and its growth;
  * the train-vs-held-out generalization gap;
  * stratified one-step object error by observable regime -- action magnitude,
    measured object displacement (free-sliding vs near-static), and
    object-gripper distance (contact proximity);
  * whether ensemble disagreement (uncertainty) and nearest-train-neighbour
    distance (out-of-distribution) actually RANK held-out error -- the three
    concepts kept separate per section 6;
  * runtime and peak VRAM.

Everything is computed from the real held-out split (never the model's own
predictions as ground truth) and written to reports/phase9a_baseline.json.

Obs layout note (verified from data, not assumed): standard 25-D Fetch vector.
object_velp cols 14:17 are stored RELATIVE to the gripper, so this script uses
the measured per-step object displacement ||d(object_pos)|| as the object-speed
regime variable rather than those columns.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memory import episode_store as store                       # noqa: E402
from world_model import training as T, augmentation as aug, ensemble as ens  # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ_COLS = [3, 4, 5]
RELV_COLS = [14, 15, 16]          # object_velp (gripper-relative) -- reported, not a regime key
RELPOS_COLS = [6, 7, 8]           # object position relative to gripper (contact proximity)
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 15, 20]
N_SEEDS = 3
EPOCHS, PATIENCE = 400, 40


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9a] CUDA probe failed ({exc}); using CPU.")
    return "cpu"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def transitions_from_eps(ep_ids):
    S, A, Sn = [], [], []
    for ep in ep_ids:
        d = store.load_episode(DATASET_ID, ep)
        obs = d["observation"]
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(d["actions"])
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


def rmse(a, b, cols):
    return float(np.sqrt(np.mean((a[:, cols] - b[:, cols]) ** 2)))


def mean_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 5), round(float(np.std(xs)), 5)) if xs else (None, None)


def verify_obs_layout(train_eps):
    """Data-driven check that cols 14:17 are object velocity relative to the gripper:
    correlate each with the measured per-step object displacement."""
    dpos, cols = [], []
    for ep in train_eps:
        obs = store.load_episode(DATASET_ID, ep)["observation"]
        dpos.append(obs[1:, OBJ_COLS] - obs[:-1, OBJ_COLS]); cols.append(obs[:-1])
    dpos = np.concatenate(dpos); cols = np.concatenate(cols)
    corr = {}
    for c in RELV_COLS:
        corr[c] = [round(float(np.corrcoef(cols[:, c], dpos[:, k])[0, 1]), 2) for k in range(3)]
    return corr


def stratify(keys, errs, edges, labels):
    """Mean object error + count within each [edge) bin of a regime variable."""
    out = {}
    idx = np.digitize(keys, edges)
    for b, lab in enumerate(labels):
        m = idx == b
        out[lab] = {"n": int(m.sum()),
                    "mean_obj_err_m": round(float(np.mean(errs[m])), 5) if m.any() else None,
                    "frac": round(float(m.mean()), 3)}
    return out


def main() -> int:
    t0 = time.time()
    device = pick_device()
    train_eps = list(store.splits(DATASET_ID)["train"])
    val = transitions_from_eps(store.splits(DATASET_ID)["val"])
    train_trans = transitions_from_eps(train_eps)
    test_trans = transitions_from_eps(store.splits(DATASET_ID)["test_heldout"])
    test_eps = store.episodes(DATASET_ID, "test_heldout")
    S, A, Sn = test_trans

    # ---- persistence baseline (predict no change) on held-out ----
    persistence = {
        "object_pos_rmse_m": round(rmse(S, Sn, OBJ_COLS), 5),
        "object_rot_rmse": round(rmse(S, Sn, ROT_COLS), 5),
        "object_relvel_rmse": round(rmse(S, Sn, RELV_COLS), 5),
        "grip_pos_rmse_m": round(rmse(S, Sn, [0, 1, 2]), 5),
        "physical_state_rmse": round(rmse(S, Sn, PHYS_COLS), 5),
    }

    # ---- train N_SEEDS one-step models; aggregate error + horizon + gen gap ----
    peak_vram = None
    seed_metrics, train_secs = [], []
    multistep_runs, persist_multi = [], None
    for s in range(N_SEEDS):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        m = aug.train_fresh(train_trans, val, device, seed=s, epochs=EPOCHS, patience=PATIENCE)
        train_secs.append(time.time() - t1)
        if device == "cuda":
            peak_vram = max(peak_vram or 0, round(torch.cuda.max_memory_allocated() / 1e9, 3))
        pred = T.predict_batch(m, S, A, device)
        pred_tr = T.predict_batch(m, train_trans[0], train_trans[1], device)
        seed_metrics.append({
            "object_pos_rmse_m": rmse(pred, Sn, OBJ_COLS),
            "object_rot_rmse": rmse(pred, Sn, ROT_COLS),
            "object_relvel_rmse": rmse(pred, Sn, RELV_COLS),
            "grip_pos_rmse_m": rmse(pred, Sn, [0, 1, 2]),
            "physical_state_rmse": rmse(pred, Sn, PHYS_COLS),
            "train_object_pos_rmse_m": rmse(pred_tr, train_trans[2], OBJ_COLS),
        })
        ms = T.evaluate_multistep(m, test_eps, device, HORIZONS, OBJ_COLS)
        multistep_runs.append({h: ms[h]["model_object_rmse_m"] for h in ms})
        if persist_multi is None:
            persist_multi = {h: ms[h]["baseline_object_rmse_m"] for h in ms}

    agg = {k: mean_std([sm[k] for sm in seed_metrics]) for k in seed_metrics[0]}
    one_step = {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()}
    multistep = {int(h): {"model_mean": round(float(np.mean([r[h] for r in multistep_runs if h in r])), 4),
                          "persistence": persist_multi[h]} for h in HORIZONS if persist_multi and h in persist_multi}
    gen_gap = round(one_step["object_pos_rmse_m"]["mean"] - one_step["train_object_pos_rmse_m"]["mean"], 5)

    # ---- stratified one-step object error (seed-0 model) by observable regime ----
    m0 = aug.train_fresh(train_trans, val, device, seed=0, epochs=EPOCHS, patience=PATIENCE)
    pred0 = T.predict_batch(m0, S, A, device)
    obj_err = np.linalg.norm(pred0[:, OBJ_COLS] - Sn[:, OBJ_COLS], axis=1)      # per-transition L2
    act_mag = np.linalg.norm(A, axis=1)
    obj_disp = np.linalg.norm(Sn[:, OBJ_COLS] - S[:, OBJ_COLS], axis=1)         # measured object speed proxy
    contact = np.linalg.norm(S[:, RELPOS_COLS], axis=1)                         # object-gripper distance
    strat = {
        "by_action_magnitude": stratify(act_mag, obj_err, [1.0, 2.0],
                                        ["low(<1)", "mid(1-2)", "high(>2)"]),
        "by_object_displacement_m": stratify(obj_disp, obj_err, [0.005, 0.02],
                                              ["near_static(<0.005)", "sliding(0.005-0.02)", "fast(>0.02)"]),
        "by_contact_distance_m": stratify(contact, obj_err, [0.05, 0.10],
                                          ["near(<0.05)", "mid(0.05-0.10)", "far(>0.10)"]),
    }

    # ---- uncertainty & OOD vs error (section 6: three distinct concepts) ----
    members = ens.train_ensemble(train_trans, val, n_models=5, device=device,
                                 seed0=100, epochs=EPOCHS, patience=PATIENCE)
    per_member = np.stack([T.predict_batch(mm, S, A, device)[:, OBJ_COLS] for mm in members])  # [M,N,3]
    ens_mean = per_member.mean(0)
    disagreement = np.linalg.norm(per_member.std(0), axis=1)                    # ensemble uncertainty
    ens_err = np.linalg.norm(ens_mean - Sn[:, OBJ_COLS], axis=1)                # ensemble one-step error
    # OOD: normalized nearest-train-neighbour distance of each test state
    s_mean, s_std = train_trans[0].mean(0), train_trans[0].std(0)
    tree = cKDTree((train_trans[0] - s_mean) / np.maximum(s_std, 1e-6))
    ood, _ = tree.query((S - s_mean) / np.maximum(s_std, 1e-6), k=1)
    calib = {
        "spearman_uncertainty_vs_error": round(float(spearmanr(disagreement, ens_err).statistic), 3),
        "spearman_ood_vs_error": round(float(spearmanr(ood, ens_err).statistic), 3),
        "spearman_uncertainty_vs_ood": round(float(spearmanr(disagreement, ood).statistic), 3),
        "error_in_high_vs_low_uncertainty_tercile_m": None,
        "n_test_transitions": int(len(ens_err)),
    }
    hi = disagreement >= np.quantile(disagreement, 2 / 3)
    lo = disagreement <= np.quantile(disagreement, 1 / 3)
    calib["error_in_high_vs_low_uncertainty_tercile_m"] = [round(float(ens_err[hi].mean()), 5),
                                                           round(float(ens_err[lo].mean()), 5)]

    report = {
        "phase": "9A", "stage": "baseline_characterization", "dataset_id": DATASET_ID,
        "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS,
                       "train_episodes": len(train_eps), "epochs": EPOCHS, "patience": PATIENCE,
                       "model": "ResidualDynamics (one-step loss, unchanged)",
                       "num_params": int(m0.num_params()), "peak_vram_gb": peak_vram},
        "obs_layout_check": {"note": "corr of cols 14:17 (object_velp) with measured object displacement (x,y,z)",
                             "corr": verify_obs_layout(train_eps)},
        "persistence_baseline": persistence,
        "one_step_held_out": one_step,
        "generalization_gap_object_m": gen_gap,
        "multistep_object_rmse_m": multistep,
        "error_growth_ratio_h20_over_h1": round(multistep[20]["model_mean"] / multistep[1]["model_mean"], 1)
        if 20 in multistep and 1 in multistep else None,
        "stratified_one_step_object_err": strat,
        "uncertainty_ood_calibration": calib,
        "runtime": {"train_seconds_mean": round(float(np.mean(train_secs)), 2),
                    "wall_seconds": round(time.time() - t0, 2)},
    }
    (ROOT / "reports" / "phase9a_baseline.json").write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print("=" * 74)
    print("DREAM-OS Phase 9A -- baseline characterization (FetchSlide-v4)")
    print("=" * 74)
    print(f"  model: {report['provenance']['num_params']} params, one-step loss, "
          f"{N_SEEDS} seeds, {len(train_eps)} train eps")
    print(f"  one-step object RMSE   : {one_step['object_pos_rmse_m']['mean']:.5f} "
          f"+/- {one_step['object_pos_rmse_m']['std']:.5f} m   (persistence {persistence['object_pos_rmse_m']})")
    print(f"  one-step physical RMSE : {one_step['physical_state_rmse']['mean']:.5f}   "
          f"(persistence {persistence['physical_state_rmse']})")
    print(f"  train->test gen gap    : {gen_gap:+.5f} m")
    print("  multi-step object RMSE vs persistence:")
    for h in HORIZONS:
        if h in multistep:
            print(f"      h={h:2d}:  model {multistep[h]['model_mean']:.4f}   "
                  f"persistence {multistep[h]['persistence']:.4f}")
    print(f"  error growth  h20/h1   : {report['error_growth_ratio_h20_over_h1']}x")
    print("  stratified one-step object error (seed 0):")
    for group, bins in strat.items():
        parts = "  ".join(f"{k}:{v['mean_obj_err_m']}(n={v['n']})" for k, v in bins.items())
        print(f"      {group}: {parts}")
    print(f"  uncertainty->error Spearman: {calib['spearman_uncertainty_vs_error']}   "
          f"OOD->error Spearman: {calib['spearman_ood_vs_error']}")
    print(f"  err hi/lo uncertainty tercile: {calib['error_in_high_vs_low_uncertainty_tercile_m']} m")
    print(f"  peak_vram={peak_vram}GB  wall={report['runtime']['wall_seconds']}s  "
          f"-> reports/phase9a_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
