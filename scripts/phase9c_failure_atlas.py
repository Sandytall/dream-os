"""Phase 9C -- failure atlas (section 6): characterize WHERE the model fails,
using only observable conditions and recorded data.

Subject: the adopted multi-step model (rollout K=5, Phase 9B). For comparison the
one-step baseline is stratified on the same windows, to show whether rollout
training fixed the hard regimes or merely scaled error down.

For every length-H diagnostic window of each held-out episode we measure, from the
REAL trajectory only:
  * object travel, peak per-step object speed  (how much sliding happened)
  * gripper-object distance over the window     (contact proximity)
  * a motion regime label: quasi-static / contact-driven / free-sliding(coasting)
  * action magnitude
  * training-data coverage = nearest-train-neighbour distance of the start (OOD)
  * ensemble disagreement of the rollout model (uncertainty)
and the realized H-step object-position error of each model.

Outputs a failure atlas (reports/phase9c_failure_atlas.json) with: error by regime,
a contact x travel failure map, uncertainty/OOD calibration (kept distinct per
section 6), a coverage-limited-region check, and the SIGNED coasting-distance bias
(does the model imagine the puck sliding farther than reality -> under-estimated
friction?). Every number traces to these windows.
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
from world_model.dynamics import ResidualDynamics               # noqa: E402
from world_model import training as T                            # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ = [3, 4, 5]
RELPOS = [6, 7, 8]
H_DIAG = 10
ROLLOUT_K = 5
N_ENSEMBLE = 5
EPOCHS, PATIENCE = 400, 40


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9c] CUDA probe failed ({exc}); using CPU.")
    return "cpu"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def flat(ep_ids):
    S, A, Sn = [], [], []
    for ep in ep_ids:
        d = store.load_episode(DATASET_ID, ep)
        obs = d["observation"]
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(d["actions"])
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


def windows(eps, H):
    """Per-window (real states[H+1,D], actions[H,A]) from each episode."""
    out = []
    for obs, act in eps:
        Tn = len(act)
        for t in range(0, Tn - H + 1):
            out.append((obs[t:t + H + 1], act[t:t + H]))
    return out


@torch.no_grad()
def rollout_windows(model, S0, Aw, device):
    """Batched open-loop rollout of every window. Returns preds [Nw, H, D]."""
    s = torch.as_tensor(S0, device=device)
    Aw = torch.as_tensor(Aw, device=device)
    out = []
    for k in range(Aw.shape[1]):
        s = model.predict(s, Aw[:, k])
        out.append(s)
    return torch.stack(out, dim=1).cpu().numpy()


def classify(real):
    """Label a window's motion regime from the real object trajectory."""
    obj = real[:, OBJ]
    c = np.linalg.norm(real[:, RELPOS], axis=1)                 # gripper-object dist [H+1]
    travel = float(np.linalg.norm(obj[-1] - obj[0]))
    d = np.linalg.norm(np.diff(obj, axis=0), axis=1)            # per-step object disp [H]
    half = len(d) // 2
    move_late, contact_late = float(d[half:].mean()), float(c[half:].mean())
    if travel < 0.01:
        cls = "quasi_static"
    elif contact_late > 0.08 and move_late > 0.004:
        cls = "free_sliding"
    else:
        cls = "contact_driven"
    return {"cls": cls, "travel": travel, "peak_speed": float(d.max()),
            "min_contact": float(c.min()), "mean_contact": float(c.mean())}


def strat(keys, errs, edges, labels):
    out = {}
    idx = np.digitize(keys, edges)
    for b, lab in enumerate(labels):
        m = idx == b
        out[lab] = {"n": int(m.sum()), "frac": round(float(m.mean()), 3),
                    "mean_err_m": round(float(np.mean(errs[m])), 5) if m.any() else None}
    return out


def main() -> int:
    t0 = time.time()
    device = pick_device()
    sp = store.splits(DATASET_ID)
    train_eps, val_eps = sp["train"], sp["val"]
    train_set, val_set = flat(train_eps), flat(val_eps)
    train_ep_list, val_ep_list = store.episodes(DATASET_ID, "train"), store.episodes(DATASET_ID, "val")
    test_eps = store.episodes(DATASET_ID, "test_heldout")
    stats = T.compute_stats(*train_set)

    # ---- build diagnostic windows + per-window regime descriptors ----
    wins = windows(test_eps, H_DIAG)
    S0 = np.asarray([w[0][0] for w in wins], np.float32)
    Aw = np.asarray([w[1] for w in wins], np.float32)
    real_obj_H = np.asarray([w[0][H_DIAG][OBJ] for w in wins], np.float32)     # truth at step H
    desc = [classify(w[0]) for w in wins]
    travel = np.asarray([d["travel"] for d in desc])
    peak = np.asarray([d["peak_speed"] for d in desc])
    min_contact = np.asarray([d["min_contact"] for d in desc])
    act_mag = np.linalg.norm(Aw, axis=2).mean(axis=1)
    cls = np.asarray([d["cls"] for d in desc])

    # ---- point models: one-step baseline vs rollout K5 (seed 0) ----
    torch.manual_seed(0)
    base = ResidualDynamics(25, 4, stats)
    T.train(base, train_set, val_set, device, epochs=EPOCHS, patience=PATIENCE, seed=0)
    base.eval()
    torch.manual_seed(0)
    roll = ResidualDynamics(25, 4, stats)
    T.train_rollout(roll, train_ep_list, val_ep_list, device, horizon=ROLLOUT_K,
                    epochs=EPOCHS, patience=PATIENCE, seed=0, obj_cols=OBJ)
    roll.eval()

    base_predH = rollout_windows(base, S0, Aw, device)[:, -1, :][:, OBJ]
    roll_predH = rollout_windows(roll, S0, Aw, device)[:, -1, :][:, OBJ]
    base_err = np.linalg.norm(base_predH - real_obj_H, axis=1)
    roll_err = np.linalg.norm(roll_predH - real_obj_H, axis=1)

    # ---- rollout ensemble for uncertainty on the adopted model ----
    members = []
    for i in range(N_ENSEMBLE):
        rng = np.random.default_rng(200 + i)
        idx = rng.integers(0, len(train_ep_list), size=len(train_ep_list))        # bootstrap episodes
        boot = [train_ep_list[j] for j in idx]
        torch.manual_seed(200 + i)
        m = ResidualDynamics(25, 4, stats)
        T.train_rollout(m, boot, val_ep_list, device, horizon=ROLLOUT_K,
                        epochs=EPOCHS, patience=PATIENCE, seed=200 + i, obj_cols=OBJ)
        m.eval(); members.append(m)
    member_predH = np.stack([rollout_windows(m, S0, Aw, device)[:, -1, :][:, OBJ] for m in members])
    ens_mean = member_predH.mean(0)
    disagreement = np.linalg.norm(member_predH.std(0), axis=1)
    ens_err = np.linalg.norm(ens_mean - real_obj_H, axis=1)

    # ---- training-data coverage (OOD) of each window start ----
    s_mean, s_std = train_set[0].mean(0), train_set[0].std(0)
    tree = cKDTree((train_set[0] - s_mean) / np.maximum(s_std, 1e-6))
    ood, _ = tree.query((S0 - s_mean) / np.maximum(s_std, 1e-6), k=1)

    # ---- error by motion regime (rollout vs baseline) ----
    by_regime = {}
    for c in ["quasi_static", "contact_driven", "free_sliding"]:
        m = cls == c
        by_regime[c] = {"n": int(m.sum()), "frac": round(float(m.mean()), 3),
                        "rollout_err_m": round(float(roll_err[m].mean()), 5) if m.any() else None,
                        "baseline_err_m": round(float(base_err[m].mean()), 5) if m.any() else None}

    # ---- contact x travel failure map (rollout model) ----
    c_edges, c_labs = [0.05, 0.10], ["near(<0.05)", "mid", "far(>0.10)"]
    t_edges, t_labs = [0.02, 0.08], ["low(<0.02)", "mid", "high(>0.08)"]
    ci, ti = np.digitize(min_contact, c_edges), np.digitize(travel, t_edges)
    fmap = {}
    for a, cl in enumerate(c_labs):
        for b, tl in enumerate(t_labs):
            m = (ci == a) & (ti == b)
            fmap[f"contact={cl},travel={tl}"] = {
                "n": int(m.sum()),
                "rollout_err_m": round(float(roll_err[m].mean()), 5) if m.any() else None}

    # ---- signed coasting bias on free-sliding windows (feeds Phase 9D) ----
    fs = cls == "free_sliding"
    real_travel = np.linalg.norm(real_obj_H - S0[:, OBJ], axis=1)
    roll_travel = np.linalg.norm(roll_predH - S0[:, OBJ], axis=1)
    base_travel = np.linalg.norm(base_predH - S0[:, OBJ], axis=1)
    coasting = {
        "n_free_sliding": int(fs.sum()),
        "real_mean_travel_m": round(float(real_travel[fs].mean()), 4),
        "rollout_mean_predicted_travel_m": round(float(roll_travel[fs].mean()), 4),
        "baseline_mean_predicted_travel_m": round(float(base_travel[fs].mean()), 4),
        "rollout_signed_travel_bias_m": round(float((roll_travel[fs] - real_travel[fs]).mean()), 4),
        "baseline_signed_travel_bias_m": round(float((base_travel[fs] - real_travel[fs]).mean()), 4),
        "interpretation": "positive bias = model imagines the puck sliding FARTHER than reality (under-estimated friction/deceleration)",
    }

    # ---- calibration: do uncertainty & OOD rank realized error? ----
    def terciles(key, err):
        hi = key >= np.quantile(key, 2 / 3); lo = key <= np.quantile(key, 1 / 3)
        return [round(float(err[hi].mean()), 5), round(float(err[lo].mean()), 5)]
    calib = {
        "spearman_uncertainty_vs_error": round(float(spearmanr(disagreement, ens_err).statistic), 3),
        "spearman_ood_vs_error": round(float(spearmanr(ood, ens_err).statistic), 3),
        "spearman_uncertainty_vs_ood": round(float(spearmanr(disagreement, ood).statistic), 3),
        "err_hi_lo_uncertainty_tercile_m": terciles(disagreement, ens_err),
        "err_hi_lo_ood_tercile_m": terciles(ood, ens_err),
        "coverage_limited_note": "high-OOD, high-error windows = where training coverage is insufficient",
    }

    report = {
        "phase": "9C", "stage": "failure_atlas", "dataset_id": DATASET_ID, "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "diagnostic_horizon": H_DIAG,
                       "rollout_K": ROLLOUT_K, "ensemble": N_ENSEMBLE, "n_windows": len(wins),
                       "subject": "rollout K5 (adopted) vs one-step baseline"},
        "error_by_motion_regime": by_regime,
        "contact_x_travel_failure_map_rollout": fmap,
        "error_by_action_magnitude_rollout": strat(act_mag, roll_err, [1.0, 1.4],
                                                   ["low(<1)", "mid(1-1.4)", "high(>1.4)"]),
        "error_by_peak_object_speed_rollout": strat(peak, roll_err, [0.01, 0.03],
                                                   ["slow(<0.01)", "mid", "fast(>0.03)"]),
        "coasting_distance_bias_free_sliding": coasting,
        "uncertainty_ood_calibration": calib,
        "wall_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "reports" / "phase9c_failure_atlas.json").write_text(json.dumps(report, indent=2))

    print("=" * 74)
    print("DREAM-OS Phase 9C -- failure atlas (FetchSlide-v4, H=%d)" % H_DIAG)
    print("=" * 74)
    print(f"  {len(wins)} diagnostic windows; subject = rollout K{ROLLOUT_K} vs one-step baseline")
    print("  error by motion regime  (rollout / baseline, object RMSE m):")
    for c, v in by_regime.items():
        print(f"      {c:14s} n={v['n']:4d} ({v['frac']*100:4.0f}%)  "
              f"rollout {v['rollout_err_m']}   baseline {v['baseline_err_m']}")
    print("  signed coasting travel bias on free-sliding windows:")
    print(f"      real travel {coasting['real_mean_travel_m']}  |  "
          f"rollout predicts {coasting['rollout_mean_predicted_travel_m']} "
          f"(bias {coasting['rollout_signed_travel_bias_m']:+.4f})  |  "
          f"baseline {coasting['baseline_mean_predicted_travel_m']} "
          f"(bias {coasting['baseline_signed_travel_bias_m']:+.4f})")
    print("  calibration (adopted rollout ensemble):")
    print(f"      uncertainty->error Spearman {calib['spearman_uncertainty_vs_error']}  "
          f"(hi/lo tercile {calib['err_hi_lo_uncertainty_tercile_m']} m)")
    print(f"      OOD->error Spearman         {calib['spearman_ood_vs_error']}  "
          f"(hi/lo tercile {calib['err_hi_lo_ood_tercile_m']} m)")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase9c_failure_atlas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
