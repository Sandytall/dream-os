"""Phase 5 -- dream-selection baselines under a matched budget.

Builds a candidate pool of dreams, scores each with signals available BEFORE
verification (ensemble uncertainty, coverage/novelty, composite), and -- only to
*evaluate* the strategies -- verifies the whole pool to learn each dream's true
error ("informativeness"). Every strategy then picks the same budget K from the
same pool. Random is run over many seeds. Failures are reported, not hidden.

Gate is METHODOLOGICAL: identical pool + identical budget for all strategies,
multi-seed random, oracle ceiling + random floor present. Not "composite wins."
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
from envs.benchmark_interface import make_env            # noqa: E402
from memory import episode_store as store, retrieval     # noqa: E402
from world_model import training as T, ensemble as ens   # noqa: E402
from dreaming import selection as sel                     # noqa: E402
from verification import replay as vr, metrics as vm      # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
N_MODELS = 5
M_STARTS = 20
HORIZON = 15
BUDGET_K = 20
N_RANDOM_SEEDS = 10
INFO_CAP_M = 1.0          # cap very-wrong dreams so they don't dominate the mean
PUSH_MAG = 0.8


def make_action_seqs(d_act, H, start_seed):
    seqs = {"noop": np.zeros((H, d_act), np.float32)}
    seqs["random"] = np.random.default_rng(start_seed).uniform(
        -1, 1, (H, d_act)).astype(np.float32)
    for ang in (0, 90, 180, 270):
        a = np.zeros((H, d_act), np.float32)
        r = np.deg2rad(ang)
        a[:, 0], a[:, 1], a[:, 2] = PUSH_MAG * np.cos(r), PUSH_MAG * np.sin(r), -0.05
        seqs[f"push{ang}"] = a
    return seqs


def main() -> int:
    t0 = time.time()
    device = "cuda"
    tr = store.transitions(DATASET_ID, "train")
    va = store.transitions(DATASET_ID, "val")
    stats = T.compute_stats(*tr)
    train_norm = (tr[0] - stats["s_mean"]) / np.maximum(stats["s_std"], 1e-6)

    print(f"[phase5] training ensemble of {N_MODELS} models ...")
    models = ens.train_ensemble(tr, va, N_MODELS, device, epochs=300, patience=30)
    d_act = tr[1].shape[1]

    # ---- build candidate pool + score + verify (for evaluation only) ----
    starts = retrieval.sample_start_states(DATASET_ID, "test_heldout", n=M_STARTS, seed=0)
    env = make_env("FetchPush-v4")
    cands = []
    for si, ref in enumerate(starts):
        for bname, acts in make_action_seqs(d_act, HORIZON, 100 + si).items():
            preds = ens.rollout_ensemble(models, ref["observation"], acts, device)
            mean_traj, std_traj = ens.mean_std(preds)
            finite = bool(np.all(np.isfinite(mean_traj)))
            max_abs = float(np.max(np.abs(mean_traj))) if finite else float("inf")
            plausible = bool(finite and max_abs < 10.0)
            unc = sel.uncertainty_score(std_traj) if finite else 1e9
            nov = sel.novelty_score(ref["observation"], train_norm,
                                    stats["s_mean"], stats["s_std"])
            real, recon_err = vr.reconstruct_and_replay(
                env, ref["dataset_id"], ref["episode"], ref["t"], acts)
            if finite:
                per = vm.object_discrepancy(mean_traj, real)
                td = float(np.mean(per[1:]))
            else:
                td = float("inf")
            info = INFO_CAP_M if not np.isfinite(td) else min(td, INFO_CAP_M)
            cands.append({"start": si, "branch": bname, "uncertainty": unc,
                          "novelty": nov, "plausible": plausible,
                          "true_disc_m": (None if not np.isfinite(td) else round(td, 4)),
                          "informativeness": round(info, 4)})
    env.close()

    n = len(cands)
    unc = np.array([c["uncertainty"] for c in cands])
    nov = np.array([c["novelty"] for c in cands])
    info = np.array([c["informativeness"] for c in cands])
    plausible_idx = [i for i, c in enumerate(cands) if c["plausible"]]

    def compare(index):
        idx = np.asarray(list(index))
        k = min(BUDGET_K, len(idx))
        u, nv, inf = unc[idx], nov[idx], info[idx]
        cp = sel.composite_score(u, nv)
        hi = set(idx[sel.select_topk(inf, k)].tolist())           # truly most-wrong dreams

        def ev(local):
            g = set(idx[np.asarray(local)].tolist())
            return {"mean_selected_info": round(float(np.mean([info[j] for j in g])), 4),
                    "recall_of_high_error": round(len(g & hi) / k, 3)}

        out = {"uncertainty": ev(sel.select_topk(u, k)),
               "coverage": ev(sel.select_topk(nv, k)),
               "composite": ev(sel.select_topk(cp, k)),
               "oracle_ceiling": ev(sel.select_topk(inf, k))}
        ri, rr = [], []
        for s in range(N_RANDOM_SEEDS):
            e = ev(sel.select_random(len(idx), k, seed=s))
            ri.append(e["mean_selected_info"])
            rr.append(e["recall_of_high_error"])
        out["random"] = {"mean_selected_info": round(float(np.mean(ri)), 4),
                         "mean_selected_info_std": round(float(np.std(ri)), 4),
                         "recall_of_high_error": round(float(np.mean(rr)), 3),
                         "seeds": N_RANDOM_SEEDS}
        rm, rs = out["random"]["mean_selected_info"], out["random"]["mean_selected_info_std"]
        for kk in ("uncertainty", "coverage", "composite", "oracle_ceiling"):
            out[kk]["beats_random"] = bool(out[kk]["mean_selected_info"] > rm + rs)
        return out, int(k)

    strategies, _ = compare(range(n))
    strategies_plausible, k_plaus = compare(plausible_idx)

    # ---- methodological gate ----
    gate = bool(N_RANDOM_SEEDS >= 5 and "oracle_ceiling" in strategies
                and "random" in strategies and len(cands) == M_STARTS * 6)

    report = {
        "phase": 5, "dataset_id": DATASET_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ensemble_size": N_MODELS, "pool_size": n, "budget_K": BUDGET_K,
        "random_seeds": N_RANDOM_SEEDS, "info_cap_m": INFO_CAP_M,
        "metric": "mean object-position error of selected dreams (higher=found more "
                  "model errors); recall of the top-K truly-most-wrong dreams",
        "n_plausible_dreams": len(plausible_idx),
        "strategies_all_dreams": strategies,
        "strategies_plausible_only": strategies_plausible,
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": gate,
    }
    (ROOT / "reports" / "phase5_results.json").write_text(json.dumps(report, indent=2))

    print("=" * 70)
    print("DREAM-OS Phase 5 report  (dream-selection baselines, matched budget)")
    print("=" * 70)
    print(f"  pool={n} dreams  budget K={BUDGET_K}  ensemble={N_MODELS}  "
          f"random_seeds={N_RANDOM_SEEDS}")
    print(f"  higher mean-info = better at finding the model's errors "
          f"(oracle=ceiling, random=floor)")
    def show(title, strat, k):
        print(f"  [{title}]   budget K={k}")
        for name in ("oracle_ceiling", "uncertainty", "composite", "coverage", "random"):
            s = strat[name]
            extra = (f" +/- {s['mean_selected_info_std']}" if name == "random"
                     else "  beats_random=" + str(s.get("beats_random")))
            print(f"     {name:16s} mean_info={s['mean_selected_info']:.3f}{extra}"
                  f"   recall={s['recall_of_high_error']}")

    show("ALL dreams (incl. obvious divergences)", strategies, BUDGET_K)
    show(f"PLAUSIBLE only (subtle errors, n={len(plausible_idx)})",
         strategies_plausible, k_plaus)
    print(f"  wall={report['wall_seconds']}s  -> reports/phase5_results.json")
    print("GATE PASSED (fair protocol):", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
