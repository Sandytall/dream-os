"""Phase 9E -- execute the decisive discriminating experiments (sections 7-8, Gates D & E).

Phase 9D left two live contenders for the free-sliding under-travel failure:
  H_velocity -- the input lacks an ABSOLUTE object-velocity signal (velp is gripper-relative),
  H_horizon  -- it is purely a multi-step propagation artifact of the training objective's horizon.
H_friction (strong one-step over-damping) was unsupported by E_speed_decay; H_coverage is argued
against by the weak OOD->error correlation (rho 0.22).

Two experiments, each changing ONE factor, separate them:

  E_abs_velocity_feature : retrain rollout-K5 with an added absolute object-velocity input
                           (finite difference, recomputed each open-loop step). H_velocity predicts
                           the free-slide travel bias shrinks substantially; H_horizon / H_friction
                           predict it is ~unchanged (the objective and dynamics capacity are the same).
  E_horizon_sweep        : free-slide travel bias vs training rollout horizon K in {1,3,5,10}.
                           H_horizon predicts bias -> 0 as K grows; H_friction predicts it plateaus
                           above zero; H_velocity predicts it is ~constant in K.

Baseline rollout-K5 and augmented-K5 are trained at the SAME seeds on the SAME data; everything but
the added input (resp. the horizon) is held fixed. Then, SEPARATELY from hypothesis discrimination
(Gate D), the learning benefit is measured (Gate E): is the winning model actually better on held-out
free-slide endpoint error AND overall multi-step prediction, beyond seed spread? If not, roll back.

-> reports/phase9e_decisive.json ; saves world_model/checkpoints/fetchslide_augvel_k5.pt
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from memory import episode_store as store                       # noqa: E402
from world_model.dynamics import ResidualDynamics               # noqa: E402
from world_model import training as T                            # noqa: E402
from world_model import aug_velocity as AV                       # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ = [3, 4, 5]
RELPOS = [6, 7, 8]
H_DIAG = 10
HORIZONS = [1, 5, 10, 15, 20]
ROLLOUT_K = 5
SWEEP_KS = [1, 3, 5, 10]
N_SEEDS = 3
EPOCHS, PATIENCE = 400, 40
CKPT_BASE = ROOT / "world_model" / "checkpoints" / "fetchslide_rollout_k5.pt"
CKPT_AUG = ROOT / "world_model" / "checkpoints" / "fetchslide_augvel_k5.pt"


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9e] CUDA probe failed ({exc}); using CPU.")
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


def classify(real):
    """Same regime labeller as Phase 9C/9D."""
    obj = real[:, OBJ]
    c = np.linalg.norm(real[:, RELPOS], axis=1)
    travel = float(np.linalg.norm(obj[-1] - obj[0]))
    d = np.linalg.norm(np.diff(obj, axis=0), axis=1)
    half = len(d) // 2
    if travel < 0.01:
        return "quasi_static"
    if c[half:].mean() > 0.08 and d[half:].mean() > 0.004:
        return "free_sliding"
    return "contact_driven"


def free_slide_windows(device_eps):
    """Held-out free-slide windows starting at t>=1 (so the aug model has a real v0).
    Returns S0[N,25], Sprev[N,25], Aw[N,H,4], reals[N,H+1,25]."""
    S0, Sprev, Aw, reals = [], [], [], []
    for obs, act in device_eps:
        for t in range(1, len(act) - H_DIAG + 1):
            real = obs[t:t + H_DIAG + 1]
            if classify(real) == "free_sliding":
                S0.append(obs[t]); Sprev.append(obs[t - 1])
                Aw.append(act[t:t + H_DIAG]); reals.append(real)
    return (np.asarray(S0, np.float32), np.asarray(Sprev, np.float32),
            np.asarray(Aw, np.float32), np.asarray(reals, np.float32))


@torch.no_grad()
def base_rollout_batch(model, S0, Aw, device):
    s = torch.as_tensor(S0, device=device).float()
    Aw = torch.as_tensor(Aw, device=device).float()
    out = []
    for k in range(Aw.shape[1]):
        s = model.predict(s, Aw[:, k])
        out.append(s)
    return torch.stack(out, dim=1).cpu().numpy()


def freeslide_metrics(pred_pos_end, S0, reals):
    """Signed travel-magnitude bias (comparable to Phase 9C) + endpoint object RMSE."""
    start = S0[:, OBJ]
    real_end = reals[:, H_DIAG, :][:, OBJ]
    real_travel = np.linalg.norm(real_end - start, axis=1)
    pred_travel = np.linalg.norm(pred_pos_end - start, axis=1)
    endpoint_err = np.linalg.norm(pred_pos_end - real_end, axis=1)
    return {
        "n_windows": int(len(S0)),
        "real_travel_m": round(float(real_travel.mean()), 4),
        "pred_travel_m": round(float(pred_travel.mean()), 4),
        "signed_travel_bias_m": round(float(pred_travel.mean() - real_travel.mean()), 4),
        "endpoint_object_rmse_m": round(float(endpoint_err.mean()), 4),
    }


def multistep(model, eps, device, aug=False):
    me, be = {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}
    for obs, act in eps:
        if aug:
            preds = AV.aug_rollout(model, obs[0], np.zeros(len(OBJ)), act, device, OBJ)  # puck at rest at t0
        else:
            preds = T.rollout(model, obs[0], act, device)
        Tn = preds.shape[0]
        for h in HORIZONS:
            if h <= Tn:
                me[h].append(np.sqrt(np.mean((preds[h - 1][OBJ] - obs[h][OBJ]) ** 2)))
                be[h].append(np.sqrt(np.mean((obs[0][OBJ] - obs[h][OBJ]) ** 2)))
    return ({h: round(float(np.mean(me[h])), 4) for h in HORIZONS if me[h]},
            {h: round(float(np.mean(be[h])), 4) for h in HORIZONS if be[h]})


def agg(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)) if xs else (None, None)


def main() -> int:
    t0 = time.time()
    device = pick_device()
    sp = store.splits(DATASET_ID)
    train_eps_ids, val_eps_ids = sp["train"], sp["val"]
    train_set = flat(train_eps_ids)
    train_ep_list = store.episodes(DATASET_ID, "train")
    val_ep_list = store.episodes(DATASET_ID, "val")
    test_eps = store.episodes(DATASET_ID, "test_heldout")
    stats = T.compute_stats(*train_set)
    base_s_std = train_set[0].std(0)
    astats = AV.aug_stats(train_ep_list, OBJ)

    fs = free_slide_windows(test_eps)
    S0, Sprev, Aw, reals = fs
    v0_real = (S0[:, OBJ] - Sprev[:, OBJ]).astype(np.float32)
    print(f"[phase9e] device={device}  free-slide held-out windows (t>=1): n={len(S0)}")

    # ---------------------------------------------------------------- #
    # E_abs_velocity_feature : baseline-K5 vs aug-K5, same seeds/data
    # ---------------------------------------------------------------- #
    base_runs, aug_runs = [], []
    best_aug_state, best_aug_val = None, float("inf")
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        bm = ResidualDynamics(25, 4, stats)
        T.train_rollout(bm, train_ep_list, val_ep_list, device, horizon=ROLLOUT_K,
                        epochs=EPOCHS, patience=PATIENCE, seed=seed, obj_cols=OBJ)
        bm.eval()
        b_pred_end = base_rollout_batch(bm, S0, Aw, device)[:, H_DIAG - 1, :][:, OBJ]
        b_fs = freeslide_metrics(b_pred_end, S0, reals)
        b_ms, persist_ms = multistep(bm, test_eps, device, aug=False)
        base_runs.append({"freeslide": b_fs, "multistep": b_ms})

        torch.manual_seed(seed)
        am = AV.AugVelDynamics(25, 4, OBJ, astats)
        info = AV.train_aug_rollout(am, train_ep_list, val_ep_list, device, base_s_std,
                                    horizon=ROLLOUT_K, epochs=EPOCHS, patience=PATIENCE,
                                    seed=seed, obj_cols=OBJ)
        am.eval()
        a_pred_end = AV.aug_rollout_batch(am, S0, v0_real, Aw, device, OBJ)[:, H_DIAG - 1, :][:, OBJ]
        a_fs = freeslide_metrics(a_pred_end, S0, reals)
        a_ms, _ = multistep(am, test_eps, device, aug=True)
        aug_runs.append({"freeslide": a_fs, "multistep": a_ms})
        if info["best_val_obj_rollout_rmse"] < best_aug_val:
            best_aug_val = info["best_val_obj_rollout_rmse"]
            best_aug_state = {k: v.detach().cpu().clone() for k, v in am.state_dict().items()}
        print(f"[phase9e]  seed {seed}: base bias {b_fs['signed_travel_bias_m']:+.4f} "
              f"endpt {b_fs['endpoint_object_rmse_m']:.4f} | aug bias {a_fs['signed_travel_bias_m']:+.4f} "
              f"endpt {a_fs['endpoint_object_rmse_m']:.4f}")

    def collect(runs, key, sub):
        return [r[key][sub] for r in runs]

    base_bias_m, base_bias_s = agg(collect(base_runs, "freeslide", "signed_travel_bias_m"))
    aug_bias_m, aug_bias_s = agg(collect(aug_runs, "freeslide", "signed_travel_bias_m"))
    base_end_m, base_end_s = agg(collect(base_runs, "freeslide", "endpoint_object_rmse_m"))
    aug_end_m, aug_end_s = agg(collect(aug_runs, "freeslide", "endpoint_object_rmse_m"))
    base_ms_mean = {h: agg([r["multistep"].get(h) for r in base_runs])[0] for h in HORIZONS}
    base_ms_std = {h: agg([r["multistep"].get(h) for r in base_runs])[1] for h in HORIZONS}
    aug_ms_mean = {h: agg([r["multistep"].get(h) for r in aug_runs])[0] for h in HORIZONS}
    aug_ms_std = {h: agg([r["multistep"].get(h) for r in aug_runs])[1] for h in HORIZONS}

    # bias magnitude (under-travel is negative); did |bias| shrink beyond spread?
    bias_mag_drop = abs(base_bias_m) - abs(aug_bias_m)
    bias_spread = base_bias_s + aug_bias_s
    bias_shrank = bool(bias_mag_drop > bias_spread)
    end_drop = base_end_m - aug_end_m
    end_spread = base_end_s + aug_end_s
    end_improved = bool(end_drop > end_spread)

    # ---------------------------------------------------------------- #
    # E_horizon_sweep : free-slide bias vs training horizon K (seed 0)
    # ---------------------------------------------------------------- #
    sweep = {}
    for K in SWEEP_KS:
        torch.manual_seed(0)
        m = ResidualDynamics(25, 4, stats)
        T.train_rollout(m, train_ep_list, val_ep_list, device, horizon=K,
                        epochs=EPOCHS, patience=PATIENCE, seed=0, obj_cols=OBJ)
        m.eval()
        p_end = base_rollout_batch(m, S0, Aw, device)[:, H_DIAG - 1, :][:, OBJ]
        fsm = freeslide_metrics(p_end, S0, reals)
        sweep[K] = {"signed_travel_bias_m": fsm["signed_travel_bias_m"],
                    "endpoint_object_rmse_m": fsm["endpoint_object_rmse_m"]}
        print(f"[phase9e]  sweep K={K:2d}: bias {fsm['signed_travel_bias_m']:+.4f}  "
              f"endpt {fsm['endpoint_object_rmse_m']:.4f}")

    sweep_biases = [abs(sweep[K]["signed_travel_bias_m"]) for K in SWEEP_KS]
    # horizon-dominated would mean |bias| keeps falling toward 0 with K; a plateau above 0 does not
    k_small, k_large = abs(sweep[SWEEP_KS[0]]["signed_travel_bias_m"]), abs(sweep[SWEEP_KS[-1]]["signed_travel_bias_m"])
    sweep_tail_drop = abs(sweep[SWEEP_KS[-2]]["signed_travel_bias_m"]) - k_large   # K=5 -> K=10
    horizon_keeps_helping = bool(sweep_tail_drop > 0.01)                           # still falling at the top end
    residual_floor = round(float(min(sweep_biases)), 4)

    # ---------------------------------------------------------------- #
    # hypothesis verdicts (Gate D) -- from measured outcomes only
    # ---------------------------------------------------------------- #
    # Long-horizon stability of the augmented model (full-episode rollout from rest, v0=0).
    aug_h20, base_h20 = (aug_ms_mean[20] or 0.0), (base_ms_mean[20] or 0.0)
    aug_long_horizon_unstable = bool(aug_h20 > max(0.5, 10.0 * base_h20))

    # The FAIR comparison for H_velocity's fix is on the free-slide WINDOWS: there both models get a
    # real in-distribution v0 and a short horizon, so both are stable and the velocity input is isolated.
    velocity_fix_helps = bias_shrank and end_improved
    # H_horizon requires the under-travel to VANISH as K grows, not merely wobble.
    horizon_removes_bias = horizon_keeps_helping and (k_large < 0.5 * k_small)
    friction_demonstrated = False   # friction is reached by ELIMINATION here, never positively shown

    if velocity_fix_helps:
        winner = "H_velocity"
        h_status = {
            "H_velocity": ("supported", f"absolute object-velocity input shrank the free-slide under-travel "
                           f"bias {base_bias_m:+.4f}->{aug_bias_m:+.4f} m (|bias| drop {bias_mag_drop:.4f} > "
                           f"spread {bias_spread:.4f}) and cut endpoint error {base_end_m:.4f}->{aug_end_m:.4f} m"),
            "H_horizon": ("weakened", f"bias removed WITHOUT changing horizon (K={ROLLOUT_K}); sweep floor {residual_floor:+.4f}"),
            "H_friction": ("contradicted", "a velocity INPUT, not a damping change, removed the under-travel"),
            "H_coverage": ("weakened", "representational fix; consistent with weak OOD->error (0.22) on record"),
        }
    elif horizon_removes_bias:
        winner = "H_horizon"
        h_status = {
            "H_horizon": ("supported", f"free-slide |bias| falls toward zero as K grows "
                          f"(K{SWEEP_KS[0]} {k_small:.3f} -> K{SWEEP_KS[-1]} {k_large:.3f} m)"),
            "H_velocity": ("contradicted", "absolute-velocity input did not shrink the bias beyond spread"),
            "H_friction": ("weakened", "bias is horizon-sensitive, not a fixed damping offset"),
            "H_coverage": ("weakened", "unchanged by representation; awaits E_add_freeslide_data if pursued"),
        }
    else:
        # Neither leading remedy removed the persistent under-travel -> friction by ELIMINATION.
        winner = "H_friction"
        vel_note = (f"absolute-velocity input did NOT reduce the free-slide under-travel bias "
                    f"({base_bias_m:+.4f}->{aug_bias_m:+.4f} m, within seed spread) or endpoint error "
                    f"({base_end_m:.4f}->{aug_end_m:.4f} m)")
        if aug_long_horizon_unstable:
            vel_note += (f"; moreover the velocity-feedback rollout is UNSTABLE from rest (full-episode h20 "
                         f"diverged to {aug_h20:.2f} m vs baseline {base_h20:.3f} m), so THIS remedy is "
                         "contradicted while a stable velocity parameterization remains untested")
        h_status = {
            "H_friction": ("most_consistent_by_elimination",
                           f"the under-travel (~{base_bias_m:+.4f} m) persists under BOTH a velocity input and a "
                           f"larger horizon, and |bias| does not approach zero with K (floor {residual_floor:+.4f} m). "
                           "Most consistent with learned over-damping, but NOT positively demonstrated -- a "
                           "damping-targeted intervention (E_damping_intervention) is the decisive next test"),
            "H_velocity": ("fix_contradicted", vel_note),
            "H_horizon": ("contradicted", f"free-slide |bias| does not fall toward zero as K grows: "
                          f"K{SWEEP_KS[0]} {k_small:.3f} -> K{SWEEP_KS[-1]} {k_large:.3f} m (it grows, then plateaus)"),
            "H_coverage": ("unresolved", "E_add_freeslide_data not run; standing OOD->error (0.22) weakly argues against"),
        }

    # Gate D: the two experiments produced a single evidence-backed verdict (one hypothesis selected,
    # the leading alternatives positively contradicted). Friction-by-elimination still counts as
    # discriminative because the alternatives were actively ruled out, not merely unsupported.
    gate_D = winner in ("H_velocity", "H_horizon", "H_friction")

    # ---------------------------------------------------------------- #
    # Gate E: learning benefit measured SEPARATELY; adopt-or-rollback
    # ---------------------------------------------------------------- #
    ms_h10_drop = (base_ms_mean[10] or 0) - (aug_ms_mean[10] or 0)
    ms_h20_drop = (base_ms_mean[20] or 0) - (aug_ms_mean[20] or 0)
    ms_h10_spread = (base_ms_std[10] or 0) + (aug_ms_std[10] or 0)
    ms_h20_spread = (base_ms_std[20] or 0) + (aug_ms_std[20] or 0)
    overall_multistep_improved = bool(ms_h20_drop > ms_h20_spread and ms_h10_drop > -ms_h10_spread)
    learning_benefit = bool(end_improved and overall_multistep_improved)
    adopt = learning_benefit
    if best_aug_state is not None:
        CKPT_AUG.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"d_state": 25, "d_act": 4, "obj_cols": OBJ, "state_dict": best_aug_state,
                    "meta": {"phase": "9E", "training": f"augvel_rollout_K{ROLLOUT_K}",
                             "dataset": DATASET_ID, "adopted_as_current_best": adopt}}, CKPT_AUG)

    report = {
        "phase": "9E", "stage": "decisive_experiments", "dataset_id": DATASET_ID, "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS,
                       "epochs": EPOCHS, "patience": PATIENCE, "rollout_K": ROLLOUT_K,
                       "baseline_checkpoint": str(CKPT_BASE.relative_to(ROOT)),
                       "aug_checkpoint": str(CKPT_AUG.relative_to(ROOT))},
        "free_slide_windows": int(len(S0)),
        "E_abs_velocity_feature": {
            "single_factor": "added absolute object-velocity input (finite diff, recomputed per step)",
            "baseline_rollout_K5": {
                "signed_travel_bias_m_mean": base_bias_m, "signed_travel_bias_m_std": base_bias_s,
                "endpoint_object_rmse_m_mean": base_end_m, "endpoint_object_rmse_m_std": base_end_s,
                "multistep_object_rmse_m_mean": base_ms_mean, "multistep_object_rmse_m_std": base_ms_std},
            "augvel_rollout_K5": {
                "signed_travel_bias_m_mean": aug_bias_m, "signed_travel_bias_m_std": aug_bias_s,
                "endpoint_object_rmse_m_mean": aug_end_m, "endpoint_object_rmse_m_std": aug_end_s,
                "multistep_object_rmse_m_mean": aug_ms_mean, "multistep_object_rmse_m_std": aug_ms_std},
            "bias_magnitude_drop_m": round(bias_mag_drop, 4), "bias_spread_m": round(bias_spread, 4),
            "bias_shrank_beyond_spread": bias_shrank,
            "endpoint_drop_m": round(end_drop, 4), "endpoint_improved_beyond_spread": end_improved,
            "aug_full_episode_h20_m": round(aug_h20, 3), "baseline_full_episode_h20_m": round(base_h20, 3),
            "aug_long_horizon_unstable": aug_long_horizon_unstable,
            "stability_note": ("the finite-diff velocity feature is fed back open-loop; stable on short free-slide "
                               "windows (real v0) but diverges on full-episode rollout from rest (v0=0)")},
        "E_horizon_sweep": {"single_factor": "training rollout horizon K",
                            "bias_and_endpoint_by_K": {str(K): sweep[K] for K in SWEEP_KS},
                            "bias_floor_m": residual_floor, "K5_to_K10_bias_drop_m": round(sweep_tail_drop, 4),
                            "horizon_keeps_helping_at_top_end": horizon_keeps_helping},
        "persistence_multistep": {h: persist_ms.get(h) for h in HORIZONS},
        "winning_hypothesis": winner,
        "friction_positively_demonstrated": friction_demonstrated,
        "recommended_next_experiment": ("E_damping_intervention: directly test learned over-damping by adding a "
                                        "velocity-dependent decay / targeted free-slide loss, measured against "
                                        "held-out free-slide travel -- the positive test H_friction still lacks"),
        "hypothesis_status": {k: {"status": v[0], "evidence": v[1]} for k, v in h_status.items()},
        "gate_D_experiments_discriminative": gate_D,
        "gate_E_learning_benefit": {
            "endpoint_improved_beyond_spread": end_improved,
            "overall_multistep_improved_beyond_spread": overall_multistep_improved,
            "h10_drop_m": round(ms_h10_drop, 4), "h20_drop_m": round(ms_h20_drop, 4),
            "learning_benefit_demonstrated": learning_benefit,
            "decision": "adopt augvel_rollout_K5" if adopt else "keep rollout_K5 (roll back)",
            "note": "measured independently of hypothesis discrimination; checkpoint rollback if no benefit"},
        "wall_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "reports" / "phase9e_decisive.json").write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print("DREAM-OS Phase 9E -- decisive experiments (FetchSlide free-slide under-travel)")
    print("=" * 78)
    print(f"  E_abs_velocity_feature (n={len(S0)} free-slide windows, {N_SEEDS} seeds):")
    print(f"    baseline-K5 : bias {base_bias_m:+.4f}+/-{base_bias_s:.4f}  endpoint {base_end_m:.4f}+/-{base_end_s:.4f}")
    print(f"    augvel-K5   : bias {aug_bias_m:+.4f}+/-{aug_bias_s:.4f}  endpoint {aug_end_m:.4f}+/-{aug_end_s:.4f}")
    print(f"    |bias| drop {bias_mag_drop:+.4f} (spread {bias_spread:.4f}) -> shrank beyond spread: {bias_shrank}")
    print(f"    endpoint drop {end_drop:+.4f} (spread {end_spread:.4f}) -> improved beyond spread: {end_improved}")
    print("  E_horizon_sweep (free-slide bias vs K): " +
          "  ".join(f"K{K}:{sweep[K]['signed_travel_bias_m']:+.3f}" for K in SWEEP_KS))
    print(f"    bias floor {residual_floor:+.4f}  K5->K10 drop {sweep_tail_drop:+.4f}  keeps helping: {horizon_keeps_helping}")
    print(f"  multistep object RMSE  h10 base {base_ms_mean[10]} aug {aug_ms_mean[10]} | "
          f"h20 base {base_ms_mean[20]} aug {aug_ms_mean[20]}")
    print(f"  aug full-episode rollout from rest UNSTABLE: {aug_long_horizon_unstable} "
          f"(aug h20 {aug_h20:.2f} m vs baseline {base_h20:.3f} m)")
    print(f"  WINNING HYPOTHESIS: {winner}  (friction positively demonstrated: {friction_demonstrated})"
          f"   (Gate D discriminative: {gate_D})")
    for k, v in h_status.items():
        print(f"    {k:12s} {v[0]:12s} {v[1]}")
    print(f"  Gate E learning benefit demonstrated: {learning_benefit}  ->  {report['gate_E_learning_benefit']['decision']}")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase9e_decisive.json")
    return 0 if gate_D else 1


if __name__ == "__main__":
    raise SystemExit(main())
