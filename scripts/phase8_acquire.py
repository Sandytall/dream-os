"""Phase 8 Stage 2 -- matched-budget acquisition on FetchSlide, WITH headroom.

Phase 7 showed no selection helps on saturated FetchPush. The §5 gate established
that FetchSlide has measurable headroom. This re-runs the acquisition comparison at
a base operating point on the STEEP part of the learning curve (so added data can
help), and asks the questions Phase 7 could not:

  (A) Can a learned acquisition model rank informativeness on held-out starts?
  (B) Under a matched real-rollout budget, does any selection (learned / uncertainty /
      coverage / composite / oracle) beat random and beat the no-add base?
  (C) Does verification reduce negative transfer vs training on raw imagined data?

Guards (unchanged from P5-P7): the acquisition model is separate from the world
model; its target is the world model's ONE-STEP object error along the REAL
trajectory (bounded; never the model grading itself); candidates are split by start
state, the predictor judged only on held-out starts; base/candidate/test episode
sets are disjoint. Gate is METHODOLOGICAL, not "learned wins."
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
from memory import episode_store as store                   # noqa: E402
from world_model import training as T, ensemble as ens      # noqa: E402
from world_model import augmentation as aug                 # noqa: E402
from dreaming import selection as sel, acquisition as acq   # noqa: E402
from verification import replay as vr, metrics as vm        # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ_COLS = [3, 4, 5]
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 20]
N_MODELS = 5
BASE_EPS = 50               # base model operating point (steep part of the headroom curve)
M_STARTS = 48               # candidate starts (from episodes disjoint from base & test)
N_TEST_STARTS = 16          # acquisition held-out starts
HORIZON = 20
N_SEEDS = 4
BUDGET_K = 24
INFO_CAP_M = 1.0


def transitions_from_eps(ep_ids):
    S, A, Sn = [], [], []
    for ep in ep_ids:
        d = store.load_episode(DATASET_ID, ep)
        obs = d["observation"]
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(d["actions"])
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


def sample_starts(ep_ids, n, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ep_ids), size=n, replace=len(ep_ids) < n)
    refs = []
    for k in chosen:
        ep = int(ep_ids[int(k)])
        d = store.load_episode(DATASET_ID, ep)
        H = d["actions"].shape[0]
        t = int(rng.integers(0, max(1, int(0.6 * H))))
        refs.append({"dataset_id": DATASET_ID, "episode": ep, "t": t,
                     "observation": d["observation"][t].astype(np.float32),
                     "desired_goal": d["desired_goal"].astype(np.float32)})
    return refs


def indist_seqs(ref, d_act, H):
    d = store.load_episode(ref["dataset_id"], ref["episode"])
    t = ref["t"]
    base = d["actions"][t:t + H].astype(np.float32)
    if len(base) < H:
        base = np.vstack([base, np.repeat(base[-1:], H - len(base), 0)]).astype(np.float32)
    clip = lambda x: np.clip(x, -1, 1).astype(np.float32)
    off_f = np.zeros(d_act, np.float32); off_f[0], off_f[2] = 0.15, -0.03
    off_s = np.zeros(d_act, np.float32); off_s[1], off_s[2] = 0.15, -0.03
    return {"replay_true": base.copy(), "scale_lo": clip(base * 0.8), "scale_hi": clip(base * 1.2),
            "damp": clip(base * 0.5), "offset_fwd": clip(base + off_f), "offset_side": clip(base + off_s)}


def ensemble_onestep(models, S, A, device):
    return np.mean([T.predict_batch(m, S, A, device) for m in models], axis=0)


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase8] CUDA probe failed ({exc}); CPU."); return "cpu"
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
    sp = store.splits(DATASET_ID)
    train_eps = list(sp["train"])
    val = transitions_from_eps(sp["val"])
    test_trans = transitions_from_eps(sp["test_heldout"])
    test_eps = store.episodes(DATASET_ID, "test_heldout")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_eps))
    base_eps = [train_eps[i] for i in perm[:BASE_EPS]]
    cand_eps = [train_eps[i] for i in perm[BASE_EPS:]]          # disjoint from base
    base_train = transitions_from_eps(base_eps)
    d_act = base_train[1].shape[1]
    stats = T.compute_stats(*base_train)
    train_norm = (base_train[0] - stats["s_mean"]) / np.maximum(stats["s_std"], 1e-6)

    print(f"[phase8] base={BASE_EPS} eps ({len(base_train[0])} trans); "
          f"candidate episodes={len(cand_eps)}; training ensemble ...")
    models = ens.train_ensemble(base_train, val, N_MODELS, device, epochs=300, patience=30)

    starts = sample_starts(cand_eps, M_STARTS, seed=1)
    test_start_ids = set(range(M_STARTS - N_TEST_STARTS, M_STARTS))
    env = make_env("FetchSlide-v4")
    cands, max_recon = [], 0.0
    for si, ref in enumerate(starts):
        for bname, acts in indist_seqs(ref, d_act, HORIZON).items():
            preds = ens.rollout_ensemble(models, ref["observation"], acts, device)
            mean_traj, std_traj = ens.mean_std(preds)
            nov = sel.novelty_score(ref["observation"], train_norm, stats["s_mean"], stats["s_std"])
            feats = acq.candidate_features(mean_traj, std_traj, nov, acts)
            real, recon_err = vr.reconstruct_and_replay(env, ref["dataset_id"], ref["episode"], ref["t"], acts)
            max_recon = max(max_recon, recon_err)
            pn = ensemble_onestep(models, real[:-1], acts, device)
            info = min(float(np.mean(np.linalg.norm(pn[:, 3:6] - real[1:, 3:6], axis=1))), INFO_CAP_M)
            cands.append({"start": si, "is_test": si in test_start_ids, "branch": bname,
                          "features": feats, "uncertainty": float(feats[0]), "novelty": nov,
                          "informativeness": round(info, 4),
                          "real_trans": aug.rollout_transitions(real, acts),
                          "imagined_start": ref["observation"], "actions": acts})
    env.close()

    tr_c = [c for c in cands if not c["is_test"]]
    te_c = [c for c in cands if c["is_test"]]
    X_tr = np.array([c["features"] for c in tr_c], np.float32)
    y_tr = np.array([c["informativeness"] for c in tr_c], np.float32)
    X_te = np.array([c["features"] for c in te_c], np.float32)
    y_te = np.array([c["informativeness"] for c in te_c], np.float32)

    # ---- (A) learned acquisition ranking ----
    pred = acq.fit_predictor(X_tr, y_tr, device="cpu", seed=0)
    pinfo_te, pinfo_tr = acq.predict(pred, X_te), acq.predict(pred, X_tr)
    rho_tr = float(spearmanr(pinfo_tr, y_tr).correlation)
    rho_te = float(spearmanr(pinfo_te, y_te).correlation)
    unc_te = np.array([c["uncertainty"] for c in te_c]); nov_te = np.array([c["novelty"] for c in te_c])
    comp_te = sel.composite_score(unc_te, nov_te)
    K = min(BUDGET_K, len(te_c))
    hi = set(np.argsort(y_te)[::-1][:K].tolist())

    def ev(scores):
        pick = set(np.argsort(np.asarray(scores))[::-1][:K].tolist())
        return (round(float(np.mean([y_te[j] for j in pick])), 4), round(len(pick & hi) / K, 3))

    sel_rows = {}
    for name, sc in (("learned", pinfo_te), ("uncertainty", unc_te), ("coverage", nov_te),
                     ("composite", comp_te), ("oracle", y_te)):
        mi, rc = ev(sc); sel_rows[name] = {"mean_selected_info": mi, "recall_top_k": rc}
    r_mi, r_rc = [], []
    for s in range(N_SEEDS * 2):
        pk = sel.select_random(len(te_c), K, seed=s)
        r_mi.append(float(np.mean([y_te[j] for j in pk]))); r_rc.append(len(set(pk) & hi) / K)
    rand_mi_m, rand_mi_s = round(float(np.mean(r_mi)), 4), round(float(np.std(r_mi)), 4)
    sel_rows["random"] = {"mean_selected_info": rand_mi_m, "mean_selected_info_std": rand_mi_s,
                          "recall_top_k": round(float(np.mean(r_rc)), 3)}
    for name in ("learned", "uncertainty", "coverage", "composite", "oracle"):
        sel_rows[name]["beats_random"] = bool(sel_rows[name]["mean_selected_info"] > rand_mi_m + rand_mi_s)

    # ---- (B) matched-budget augmentation on held-out candidates ----
    seeds = list(range(N_SEEDS))

    def real_extra(idx_list):
        parts = [te_c[i]["real_trans"] for i in idx_list]
        return tuple(np.concatenate([p[j] for p in parts]).astype(np.float32) for j in range(3))

    def condition(extra, pick_per_seed=None):
        objs, ms = [], []
        for s in seeds:
            ex = real_extra(pick_per_seed(s)) if pick_per_seed else extra
            m = aug.train_fresh(aug.concat_sets(base_train, ex) if ex else base_train, val, device,
                                seed=s, epochs=350, patience=35)
            e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
            objs.append(e["object_pos_rmse_m"]); ms.append(e["multistep_object_rmse_m"])
        om, osd = mean_std(objs)
        ms_mean = {h: round(float(np.mean([mm[h] for mm in ms if h in mm])), 4)
                   for h in HORIZONS if any(h in mm for mm in ms)}
        return {"object_pos_rmse_m_mean": om, "object_pos_rmse_m_std": osd,
                "multistep_object_rmse_m": ms_mean, "runs": [round(o, 5) for o in objs],
                "n_transitions_added": 0 if (extra is None and not pick_per_seed) else K * HORIZON}

    picks = {"uncertainty": list(np.argsort(unc_te)[::-1][:K]),
             "coverage": list(np.argsort(nov_te)[::-1][:K]),
             "composite": list(np.argsort(comp_te)[::-1][:K]),
             "learned": list(np.argsort(pinfo_te)[::-1][:K]),
             "oracle": list(np.argsort(y_te)[::-1][:K])}
    aug_rows = {"base": condition(None),
                "random": condition(None, pick_per_seed=lambda s: sel.select_random(len(te_c), K, seed=s))}
    for name, idx in picks.items():
        aug_rows[name] = condition(real_extra(idx))
    base_m = aug_rows["base"]["object_pos_rmse_m_mean"]
    rand_m = aug_rows["random"]["object_pos_rmse_m_mean"]; rand_s = aug_rows["random"]["object_pos_rmse_m_std"] or 0.0
    aug_dec = {}
    for name, r in aug_rows.items():
        if name == "base":
            continue
        r["beats_base"] = bool(r["object_pos_rmse_m_mean"] < base_m)
        r["beats_random"] = bool(r["object_pos_rmse_m_mean"] < rand_m - rand_s)
        aug_dec[name] = "adopt" if r["beats_base"] else "rollback_to_base"

    # ---- (C) verification vs raw imagination (same learned selection) ----
    imag_parts, dropped = [], 0
    for i in picks["learned"]:
        it, drp = aug.finite_mask(aug.imagined_transitions(models[0], te_c[i]["imagined_start"],
                                                           te_c[i]["actions"], device))
        dropped += drp
        if len(it[0]):
            imag_parts.append(it)
    imag_extra = tuple(np.concatenate([p[j] for p in imag_parts]).astype(np.float32)
                       for j in range(3)) if imag_parts else None
    partC = {"verified_real_learned": {k: aug_rows["learned"][k] for k in
                                       ("object_pos_rmse_m_mean", "object_pos_rmse_m_std")},
             "imagined_unfiltered_learned": condition(imag_extra),
             "imagined_transitions_dropped_nonfinite": dropped}
    partC["imagined_unfiltered_learned"]["beats_base"] = bool(
        partC["imagined_unfiltered_learned"]["object_pos_rmse_m_mean"] < base_m)

    # ---- gate ----
    disjoint_starts = {c["start"] for c in tr_c}.isdisjoint({c["start"] for c in te_c})
    disjoint_eps = set(base_eps).isdisjoint(set(cand_eps)) and set(base_eps).isdisjoint(set(sp["test_heldout"]))
    real_added = {aug_rows[k]["n_transitions_added"] for k in picks} | {aug_rows["random"]["n_transitions_added"]}
    gate = bool(disjoint_starts and disjoint_eps and real_added == {K * HORIZON}
                and max_recon < vm.RECON_TOL_M
                and aug_rows["base"]["object_pos_rmse_m_std"] is not None
                and aug_rows["random"]["object_pos_rmse_m_std"] is not None
                and "oracle" in sel_rows and "oracle" in aug_rows and len(aug_dec) == 6)

    report = {
        "phase": 8, "stage": "acquisition", "dataset_id": DATASET_ID, "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "ensemble_size": N_MODELS,
                       "base_eps": BASE_EPS, "m_starts": M_STARTS, "n_test_starts": N_TEST_STARTS,
                       "horizon": HORIZON, "budget_K": BUDGET_K, "n_seeds": N_SEEDS,
                       "features": acq.FEATURE_NAMES},
        "pool_size": len(cands), "n_train_candidates": len(tr_c), "n_test_candidates": len(te_c),
        "candidates_disjoint_by_start": disjoint_starts, "episode_sets_disjoint": disjoint_eps,
        "max_reconstruction_err_m": round(max_recon, 8),
        "A_ranking": {"spearman_train": round(rho_tr, 3), "spearman_test": round(rho_te, 3),
                      "budget_K": K, "strategies": sel_rows},
        "B_loop": {"metric": "held-out object-pos RMSE (m), lower=better",
                   "conditions": aug_rows, "rollback_decisions": aug_dec,
                   "real_transitions_added": sorted(real_added)},
        "C_verification_vs_imagination": partC,
        "wall_seconds": round(time.time() - t0, 2), "gate_passed": gate,
    }
    (ROOT / "reports" / "phase8_acquire.json").write_text(json.dumps(report, indent=2))

    # ---- teaching summary ----
    print("=" * 78)
    print("DREAM-OS Phase 8 Stage 2 -- acquisition WITH headroom (FetchSlide)")
    print("=" * 78)
    print(f"  base={BASE_EPS} eps  pool={len(cands)}  test_cands={len(te_c)}  K={K}  seeds={N_SEEDS}")
    print(f"  (A) learned predictor Spearman: train={rho_tr:.2f} test={rho_te:.2f}")
    for name in ("oracle", "learned", "composite", "uncertainty", "coverage", "random"):
        r = sel_rows[name]
        ex = (f" +/-{r['mean_selected_info_std']}" if name == "random"
              else " beats_rand=" + str(r.get("beats_random")))
        print(f"       {name:12s} mean_info={r['mean_selected_info']:.3f}{ex}  recall={r['recall_top_k']}")
    print(f"  (B) held-out object RMSE (lower better), budget K={K} real rollouts:")
    print(f"       {'base':12s} {aug_rows['base']['object_pos_rmse_m_mean']:.5f} +/- {aug_rows['base']['object_pos_rmse_m_std']:.5f}")
    for name in ("random", "uncertainty", "coverage", "composite", "learned", "oracle"):
        r = aug_rows[name]
        print(f"       {name:12s} {r['object_pos_rmse_m_mean']:.5f} +/- {r['object_pos_rmse_m_std']:.5f}"
              f"  beats_base={r['beats_base']!s:5s} beats_rand={r['beats_random']!s:5s} -> {aug_dec[name]}")
    pc = partC["imagined_unfiltered_learned"]
    print(f"  (C) verification value: verified-real(learned) {partC['verified_real_learned']['object_pos_rmse_m_mean']:.5f}"
          f"  vs raw-imagined(learned) {pc['object_pos_rmse_m_mean']:.5f} "
          f"({partC['imagined_transitions_dropped_nonfinite']} imagined dropped non-finite)")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase8_acquire.json")
    print("GATE PASSED (fair protocol):", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
