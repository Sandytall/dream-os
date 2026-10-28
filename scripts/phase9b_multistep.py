"""Phase 9B -- does a short-horizon ROLLOUT training loss improve multi-step
held-out prediction, without regressing one-step accuracy or resource use?

Single factor changed vs the Phase 9A baseline: the training objective horizon K.
  * K=1  = the existing one-step-loss model (`training.train`), the baseline.
  * K>1  = back-prop through a K-step open-loop rollout (`training.train_rollout`).
Everything else (architecture, stats, epochs, patience, lr, batch, seeds, splits,
held-out test) is identical, so any difference is attributable to the loss horizon.

Reports, per condition and seed-aggregated: one-step object & physical RMSE, the
full multi-step object-RMSE curve vs persistence, training time, epochs. Then an
explicit, honest verdict (Gate B): did multi-step improve beyond seed spread, and
did one-step regress? -> reports/phase9b_multistep.json
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

DATASET_ID = "fetchslide_phase8"
OBJ_COLS = [3, 4, 5]
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 15, 20]
ROLLOUT_KS = [3, 5, 10]
N_SEEDS = 3
EPOCHS, PATIENCE = 400, 40


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9b] CUDA probe failed ({exc}); using CPU.")
    return "cpu"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def flat_transitions(ep_ids):
    S, A, Sn = [], [], []
    for ep in ep_ids:
        obs = store.load_episode(DATASET_ID, ep)["observation"]
        S.append(obs[:-1]); Sn.append(obs[1:])
        A.append(store.load_episode(DATASET_ID, ep)["actions"])
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


def rmse(a, b, cols):
    return float(np.sqrt(np.mean((a[:, cols] - b[:, cols]) ** 2)))


def mean_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 5), round(float(np.std(xs)), 5)) if xs else (None, None)


def evaluate(model, test_trans, test_eps, device):
    S, A, Sn = test_trans
    pred = T.predict_batch(model, S, A, device)
    ms = T.evaluate_multistep(model, test_eps, device, HORIZONS, OBJ_COLS)
    return {"object_pos_rmse_m": rmse(pred, Sn, OBJ_COLS),
            "physical_state_rmse": rmse(pred, Sn, PHYS_COLS),
            "multistep": {h: ms[h]["model_object_rmse_m"] for h in ms},
            "persistence_multistep": {h: ms[h]["baseline_object_rmse_m"] for h in ms}}


def run_condition(label, train_fn, device, train_set, train_eps, val_set, val_eps,
                  test_trans, test_eps):
    stats = T.compute_stats(*train_set)
    one, phys, ms_runs, secs, epochs = [], [], [], [], []
    persist = None
    for s in range(N_SEEDS):
        torch.manual_seed(s)
        m = ResidualDynamics(train_set[0].shape[1], train_set[1].shape[1], stats)
        t1 = time.time()
        info = train_fn(m, s)
        secs.append(time.time() - t1); epochs.append(info["epochs_run"])
        m.eval()
        e = evaluate(m, test_trans, test_eps, device)
        one.append(e["object_pos_rmse_m"]); phys.append(e["physical_state_rmse"])
        ms_runs.append(e["multistep"]); persist = e["persistence_multistep"]
    om, osd = mean_std(one)
    ms_mean = {int(h): round(float(np.mean([r[h] for r in ms_runs if h in r])), 4) for h in HORIZONS}
    ms_std = {int(h): round(float(np.std([r[h] for r in ms_runs if h in r])), 4) for h in HORIZONS}
    print(f"[phase9b] {label:14s} 1-step {om:.5f}+/-{osd:.5f}  "
          f"h10 {ms_mean[10]:.4f}  h20 {ms_mean[20]:.4f}  "
          f"({np.mean(epochs):.0f} ep, {np.mean(secs):.1f}s)")
    return {"one_step_object_rmse_m_mean": om, "one_step_object_rmse_m_std": osd,
            "one_step_physical_rmse_mean": mean_std(phys)[0],
            "multistep_object_rmse_m_mean": ms_mean, "multistep_object_rmse_m_std": ms_std,
            "persistence_multistep": {int(h): persist[h] for h in persist},
            "train_seconds_mean": round(float(np.mean(secs)), 2),
            "epochs_mean": round(float(np.mean(epochs)), 1)}


def main() -> int:
    t0 = time.time()
    device = pick_device()
    sp = store.splits(DATASET_ID)
    train_eps, val_eps = sp["train"], sp["val"]
    train_set = flat_transitions(train_eps)
    val_set = flat_transitions(val_eps)
    val_ep_list = store.episodes(DATASET_ID, "val")
    train_ep_list = store.episodes(DATASET_ID, "train")
    test_trans = flat_transitions(sp["test_heldout"])
    test_eps = store.episodes(DATASET_ID, "test_heldout")

    conditions = {}
    # baseline: existing one-step loss
    conditions["onestep_K1"] = run_condition(
        "onestep_K1",
        lambda m, s: T.train(m, train_set, val_set, device, epochs=EPOCHS, patience=PATIENCE, seed=s),
        device, train_set, train_eps, val_set, val_eps, test_trans, test_eps)
    # treatments: rollout loss at several horizons
    for K in ROLLOUT_KS:
        conditions[f"rollout_K{K}"] = run_condition(
            f"rollout_K{K}",
            lambda m, s, K=K: T.train_rollout(m, train_ep_list, val_ep_list, device, horizon=K,
                                              epochs=EPOCHS, patience=PATIENCE, seed=s, obj_cols=OBJ_COLS),
            device, train_set, train_eps, val_set, val_eps, test_trans, test_eps)

    # ---- honest verdict (Gate B) ----
    base = conditions["onestep_K1"]
    b_one, b_one_sd = base["one_step_object_rmse_m_mean"], base["one_step_object_rmse_m_std"]
    verdicts = {}
    best_label, best_h20 = None, base["multistep_object_rmse_m_mean"][20]
    for lab, c in conditions.items():
        if lab == "onestep_K1":
            continue
        h20_d = base["multistep_object_rmse_m_mean"][20] - c["multistep_object_rmse_m_mean"][20]
        h10_d = base["multistep_object_rmse_m_mean"][10] - c["multistep_object_rmse_m_mean"][10]
        h20_spread = base["multistep_object_rmse_m_std"][20] + c["multistep_object_rmse_m_std"][20]
        one_reg = c["one_step_object_rmse_m_mean"] - b_one          # +ve = worse one-step
        one_spread = b_one_sd + c["one_step_object_rmse_m_std"]
        verdicts[lab] = {
            "h10_improvement_m": round(h10_d, 4), "h20_improvement_m": round(h20_d, 4),
            "h20_improves_beyond_spread": bool(h20_d > h20_spread),
            "one_step_regression_m": round(one_reg, 5),
            "one_step_regressed_beyond_spread": bool(one_reg > one_spread)}
        if c["multistep_object_rmse_m_mean"][20] < best_h20:
            best_h20, best_label = c["multistep_object_rmse_m_mean"][20], lab

    improved = best_label is not None and verdicts[best_label]["h20_improves_beyond_spread"]
    no_regress = best_label is not None and not verdicts[best_label]["one_step_regressed_beyond_spread"]

    report = {
        "phase": "9B", "stage": "multistep_rollout_training", "dataset_id": DATASET_ID,
        "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS,
                       "epochs": EPOCHS, "patience": PATIENCE, "rollout_horizons": ROLLOUT_KS,
                       "single_factor": "training objective horizon K (loss), all else fixed"},
        "metric": "held-out object-position RMSE (m), lower=better; mean +/- std over seeds",
        "conditions": conditions,
        "verdicts_vs_baseline": verdicts,
        "best_multistep_condition": best_label,
        "best_improves_h20_beyond_spread": improved,
        "best_no_one_step_regression": no_regress,
        "wall_seconds": round(time.time() - t0, 2),
        "gate_B_tested": True,
    }
    (ROOT / "reports" / "phase9b_multistep.json").write_text(json.dumps(report, indent=2))

    print("=" * 74)
    print("DREAM-OS Phase 9B -- rollout-loss multi-step model (FetchSlide-v4)")
    print("=" * 74)
    print("  held-out object RMSE by horizon (persistence in brackets):")
    persist = base["persistence_multistep"]
    header = "    cond           " + "".join(f"  h{h:<2d}" for h in HORIZONS)
    print(header)
    for lab, c in conditions.items():
        row = "".join(f" {c['multistep_object_rmse_m_mean'][h]:.3f}" for h in HORIZONS)
        print(f"    {lab:14s} {row}   1step {c['one_step_object_rmse_m_mean']:.5f}")
    print("    persistence    " + "".join(f" {persist[h]:.3f}" for h in HORIZONS))
    print(f"  best multi-step condition: {best_label}  "
          f"(h20 {best_h20:.4f} vs baseline {base['multistep_object_rmse_m_mean'][20]:.4f})")
    print(f"  improves h20 beyond seed spread: {improved}   no one-step regression: {no_regress}")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase9b_multistep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
