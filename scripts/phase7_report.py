"""Phase 7 -- a LEARNED acquisition function for active experiments.

Phase 6's open problem: the oracle (select by true post-hoc error) improves the
model under a matched budget, but usable pre-verification signals (uncertainty,
coverage, composite) do not find those experiments. Phase 7 tries to LEARN the
acquisition function and asks two questions, honestly:

  (A) Ranking: trained only on past verified dreams, does the learned predictor
      rank held-out candidate dreams by informativeness better than uncertainty /
      coverage / composite / random -- and how close to the oracle ceiling?
  (B) The loop: plugged into the Phase-6 augmentation under a matched real budget,
      does the learned selector recover the oracle's held-out improvement?

Guards: the acquisition model is SEPARATE from the world model; its targets are
real-simulator discrepancies, never the model grading itself; candidates are split
by START STATE into train/test so the predictor is only ever judged on dreams it
never saw. Gate is METHODOLOGICAL (disjoint split, matched budget, oracle ceiling
+ random floor, error bars), not "learned wins."
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env               # noqa: E402
from memory import episode_store as store, retrieval        # noqa: E402
from world_model import training as T, ensemble as ens      # noqa: E402
from world_model import augmentation as aug                 # noqa: E402
from dreaming import selection as sel, acquisition as acq   # noqa: E402
from verification import replay as vr, metrics as vm        # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
OBJ_COLS = [3, 4, 5]
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 20]

N_MODELS = 5
M_STARTS = 40
N_TEST_STARTS = 14          # candidates from these starts are held out from the predictor
HORIZON = 15
N_SEEDS = 5
INFO_CAP_M = 1.0
BUDGET_K = 18               # selection budget on the test pool (and Part-B real budget)


def indist_seqs(ref, d_act, H, seed):
    """In-distribution candidates: the real recorded pusher continuation + perturbations."""
    d = store.load_episode(ref["dataset_id"], ref["episode"])
    t = ref["t"]
    base = d["actions"][t:t + H].astype(np.float32)
    if len(base) < H:
        base = np.vstack([base, np.repeat(base[-1:], H - len(base), 0)]).astype(np.float32)
    clip = lambda x: np.clip(x, -1, 1).astype(np.float32)
    # SMOOTH in-distribution variations only (scales, constant offsets, damping). Per-timestep
    # jitter is out-of-distribution in ACTION space -- the model extrapolates wildly on it and
    # its one-step error saturates, which is an artifact, not graded in-distribution signal.
    off_f = np.zeros(d_act, np.float32); off_f[0], off_f[2] = 0.15, -0.03   # push a bit more in +x
    off_s = np.zeros(d_act, np.float32); off_s[1], off_s[2] = 0.15, -0.03   # push a bit more in +y
    return {"replay_true": base.copy(),
            "scale_lo": clip(base * 0.8), "scale_hi": clip(base * 1.2),
            "damp": clip(base * 0.5),
            "offset_fwd": clip(base + off_f), "offset_side": clip(base + off_s)}


def ensemble_onestep(models, S, A, device):
    """Ensemble-mean ONE-STEP prediction from given (real) states -- teacher-forced."""
    return np.mean([T.predict_batch(m, S, A, device) for m in models], axis=0)


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase7] CUDA probe failed ({exc}); using CPU.")
    return "cpu"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def mean_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 5), round(float(np.std(xs)), 5)) if xs else (None, None)


def main() -> int:
    t0 = time.time()
    device = pick_device()

    base_train = store.transitions(DATASET_ID, "train")
    val = store.transitions(DATASET_ID, "val")
    test_trans = store.transitions(DATASET_ID, "test_heldout")
    test_eps = store.episodes(DATASET_ID, "test_heldout")
    d_act = base_train[1].shape[1]
    stats = T.compute_stats(*base_train)
    train_norm = (base_train[0] - stats["s_mean"]) / np.maximum(stats["s_std"], 1e-6)

    print(f"[phase7] training ensemble of {N_MODELS} ...")
    models = ens.train_ensemble(base_train, val, N_MODELS, device, epochs=300, patience=30)

    # ---- candidate pool: features (pre-verification) + verify-all for true informativeness ----
    print(f"[phase7] building pool ({M_STARTS} starts x 6 branches) ...")
    starts = retrieval.sample_start_states(DATASET_ID, "train", n=M_STARTS, seed=2)
    test_start_ids = set(range(M_STARTS - N_TEST_STARTS, M_STARTS))
    env = make_env("FetchPush-v4")
    cands, max_recon = [], 0.0
    for si, ref in enumerate(starts):
        for bname, acts in indist_seqs(ref, d_act, HORIZON, 2000 + si).items():
            preds = ens.rollout_ensemble(models, ref["observation"], acts, device)
            mean_traj, std_traj = ens.mean_std(preds)
            nov = sel.novelty_score(ref["observation"], train_norm, stats["s_mean"], stats["s_std"])
            feats = acq.candidate_features(mean_traj, std_traj, nov, acts)
            real, recon_err = vr.reconstruct_and_replay(
                env, ref["dataset_id"], ref["episode"], ref["t"], acts)
            max_recon = max(max_recon, recon_err)
            # informativeness = world model's ONE-STEP object error along the REAL trajectory:
            # bounded + graded, and exactly the error that adding these real transitions would fix
            # (NOT imagined-vs-real divergence, which Phase 6 showed is dominated by useless blow-ups).
            pn = ensemble_onestep(models, real[:-1], acts, device)
            ose = float(np.mean(np.linalg.norm(pn[:, 3:6] - real[1:, 3:6], axis=1)))
            info = min(ose, INFO_CAP_M)
            cands.append({"start": si, "is_test": si in test_start_ids, "branch": bname,
                          "features": feats, "uncertainty": float(feats[0]), "novelty": nov,
                          "informativeness": round(info, 4),
                          "real_trans": aug.rollout_transitions(real, acts)})
    env.close()
    _in = np.array([c["informativeness"] for c in cands])
    info_dist = {"min": round(float(_in.min()), 4), "median": round(float(np.median(_in)), 4),
                 "max": round(float(_in.max()), 4), "n_at_cap": int((_in >= 1.0).sum()),
                 "by_branch_median": {b: round(float(np.median([c["informativeness"]
                                      for c in cands if c["branch"] == b])), 4)
                                      for b in ("replay_true", "scale_lo", "scale_hi",
                                                "damp", "offset_fwd", "offset_side")}}
    print(f"[phase7] informativeness (one-step obj error, m): {info_dist['min']}..{info_dist['max']} "
          f"median {info_dist['median']}  by-branch {info_dist['by_branch_median']}")

    tr_c = [c for c in cands if not c["is_test"]]
    te_c = [c for c in cands if c["is_test"]]
    X_tr = np.array([c["features"] for c in tr_c], np.float32)
    y_tr = np.array([c["informativeness"] for c in tr_c], np.float32)
    X_te = np.array([c["features"] for c in te_c], np.float32)
    y_te = np.array([c["informativeness"] for c in te_c], np.float32)

    # ---- (A) learn the acquisition function; evaluate ranking on held-out candidates ----
    print(f"[phase7] fitting acquisition predictor on {len(tr_c)} train candidates ...")
    pred = acq.fit_predictor(X_tr, y_tr, device="cpu", seed=0)
    pinfo_te = acq.predict(pred, X_te)
    pinfo_tr = acq.predict(pred, X_tr)
    rho_tr = float(spearmanr(pinfo_tr, y_tr).correlation)
    rho_te = float(spearmanr(pinfo_te, y_te).correlation)

    unc_te = np.array([c["uncertainty"] for c in te_c])
    nov_te = np.array([c["novelty"] for c in te_c])
    comp_te = sel.composite_score(unc_te, nov_te)
    K = min(BUDGET_K, len(te_c))
    hi = set(np.argsort(y_te)[::-1][:K].tolist())

    def ev(scores):
        pick = set(np.argsort(np.asarray(scores))[::-1][:K].tolist())
        return (round(float(np.mean([y_te[j] for j in pick])), 4), round(len(pick & hi) / K, 3))

    sel_rows = {}
    for name, sc in (("learned", pinfo_te), ("uncertainty", unc_te),
                     ("coverage", nov_te), ("composite", comp_te), ("oracle", y_te)):
        mi, rc = ev(sc)
        sel_rows[name] = {"mean_selected_info": mi, "recall_top_k": rc}
    r_mi, r_rc = [], []
    for s in range(N_SEEDS * 2):
        pick = sel.select_random(len(te_c), K, seed=s)
        r_mi.append(float(np.mean([y_te[j] for j in pick])))
        r_rc.append(len(set(pick) & hi) / K)
    rand_mi_m, rand_mi_s = round(float(np.mean(r_mi)), 4), round(float(np.std(r_mi)), 4)
    sel_rows["random"] = {"mean_selected_info": rand_mi_m, "mean_selected_info_std": rand_mi_s,
                          "recall_top_k": round(float(np.mean(r_rc)), 3), "seeds": N_SEEDS * 2}
    for name in ("learned", "uncertainty", "coverage", "composite", "oracle"):
        sel_rows[name]["beats_random"] = bool(sel_rows[name]["mean_selected_info"] > rand_mi_m + rand_mi_s)

    # ---- (B) the loop: matched-budget augmentation on the TEST candidates ----
    print("[phase7] Part B: augment + retrain under matched budget ...")
    seeds = list(range(N_SEEDS))

    def real_extra(idx_list):
        parts = [te_c[i]["real_trans"] for i in idx_list]
        return tuple(np.concatenate([p[j] for p in parts]).astype(np.float32) for j in range(3))

    def condition(extra, pick_per_seed=None):
        objs = []
        for s in seeds:
            ex = real_extra(pick_per_seed(s)) if pick_per_seed else extra
            m = aug.train_fresh(aug.concat_sets(base_train, ex) if ex else base_train, val, device, seed=s)
            e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
            objs.append(e["object_pos_rmse_m"])
        om, osd = mean_std(objs)
        n_added = 0 if (extra is None and pick_per_seed is None) else K * HORIZON
        return {"object_pos_rmse_m_mean": om, "object_pos_rmse_m_std": osd,
                "runs": [round(o, 5) for o in objs], "n_transitions_added": n_added}

    learned_pick = list(np.argsort(pinfo_te)[::-1][:K])
    oracle_pick = list(np.argsort(y_te)[::-1][:K])
    aug_rows = {
        "base": condition(None),
        "random": condition(None, pick_per_seed=lambda s: sel.select_random(len(te_c), K, seed=s)),
        "learned": condition(real_extra(learned_pick)),
        "oracle": condition(real_extra(oracle_pick)),
    }
    base_m = aug_rows["base"]["object_pos_rmse_m_mean"]
    rand_m = aug_rows["random"]["object_pos_rmse_m_mean"]
    rand_s = aug_rows["random"]["object_pos_rmse_m_std"] or 0.0
    aug_decisions = {}
    for name, r in aug_rows.items():
        if name == "base":
            continue
        r["beats_base"] = bool(r["object_pos_rmse_m_mean"] < base_m)
        r["beats_random"] = bool(r["object_pos_rmse_m_mean"] < rand_m - rand_s)
        aug_decisions[name] = "adopt" if r["beats_base"] else "rollback_to_base"

    # ---- methodological gate ----
    disjoint = {c["start"] for c in tr_c}.isdisjoint({c["start"] for c in te_c})
    real_added = {aug_rows[k]["n_transitions_added"] for k in ("random", "learned", "oracle")}
    gate = bool(
        disjoint and real_added == {K * HORIZON}
        and max_recon < vm.RECON_TOL_M
        and aug_rows["base"]["object_pos_rmse_m_std"] is not None
        and aug_rows["random"]["object_pos_rmse_m_std"] is not None
        and "oracle" in sel_rows and "random" in sel_rows
        and "oracle" in aug_rows and len(aug_decisions) == 3
    )

    report = {
        "phase": 7, "dataset_id": DATASET_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "ensemble_size": N_MODELS,
                       "m_starts": M_STARTS, "n_test_starts": N_TEST_STARTS, "horizon": HORIZON,
                       "budget_K": BUDGET_K, "n_seeds": N_SEEDS, "features": acq.FEATURE_NAMES},
        "pool_size": len(cands), "n_train_candidates": len(tr_c), "n_test_candidates": len(te_c),
        "candidates_disjoint_by_start": disjoint,
        "max_reconstruction_err_m": round(max_recon, 8),
        "A_ranking": {"spearman_train": round(rho_tr, 3), "spearman_test": round(rho_te, 3),
                      "budget_K": K, "informativeness_m": info_dist, "strategies": sel_rows,
                      "metric": "mean true informativeness of selected (higher=found more model "
                                "error); recall of the top-K truly-most-informative"},
        "B_loop": {"metric": "held-out object-pos RMSE (m), lower=better; real test data only",
                   "conditions": aug_rows, "rollback_decisions": aug_decisions,
                   "real_transitions_added": sorted(real_added)},
        "wall_seconds": round(time.time() - t0, 2), "gate_passed": gate,
    }
    (ROOT / "reports" / "phase7_results.json").write_text(json.dumps(report, indent=2))

    # ---- teaching summary ----
    print("=" * 76)
    print("DREAM-OS Phase 7 report  (a learned acquisition function for active experiments)")
    print("=" * 76)
    print(f"  pool={len(cands)}  train_cands={len(tr_c)}  test_cands={len(te_c)} "
          f"(disjoint starts={disjoint})  device={device}")
    print(f"  (A) does the predictor PREDICT informativeness?  "
          f"Spearman rho: train={rho_tr:.2f}  test={rho_te:.2f}")
    print(f"      ranking on held-out dreams (budget K={K}; higher mean_info=better, "
          f"oracle=ceiling, random=floor):")
    for name in ("oracle", "learned", "composite", "uncertainty", "coverage", "random"):
        r = sel_rows[name]
        extra = (f" +/- {r['mean_selected_info_std']}" if name == "random"
                 else "  beats_random=" + str(r.get("beats_random")))
        print(f"       {name:12s} mean_info={r['mean_selected_info']:.3f}{extra}"
              f"   recall={r['recall_top_k']}")
    print(f"  (B) does it IMPROVE the model under matched budget (K={K} real rollouts)?  "
          f"held-out object RMSE (lower better):")
    print(f"       {'base':10s} {aug_rows['base']['object_pos_rmse_m_mean']:.5f} "
          f"+/- {aug_rows['base']['object_pos_rmse_m_std']:.5f}")
    for name in ("random", "learned", "oracle"):
        r = aug_rows[name]
        print(f"       {name:10s} {r['object_pos_rmse_m_mean']:.5f} +/- {r['object_pos_rmse_m_std']:.5f}"
              f"   beats_base={r['beats_base']!s:5s} beats_rand={r['beats_random']!s:5s} "
              f"-> {aug_decisions[name]}")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase7_results.json")
    print("GATE PASSED (fair protocol):", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
