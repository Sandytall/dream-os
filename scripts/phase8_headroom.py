"""Phase 8 -- FIRST ACCEPTANCE GATE (section 5): demonstrate learning headroom.

Before any acquisition claim, establish whether adding real-simulator data
measurably improves held-out prediction on FetchSlide. Train the same world-model
architecture at several data budgets (by number of collected episodes), multiple
seeds each, and evaluate on a FIXED held-out test split in real units.

Gate = headroom exists if held-out object-position error at the largest budget is
lower than at the smallest budget by more than the seed-to-seed spread (a defensible
evaluation regime with measurable headroom). If not, this reports that honestly and
prints diagnostics rather than proceeding to acquisition claims.
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
from memory import episode_store as store                  # noqa: E402
from world_model import training as T, augmentation as aug  # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ_COLS = [3, 4, 5]
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 20]
BUDGETS_EP = [20, 50, 100, 168]      # number of training episodes
N_SEEDS = 4


def transitions_from_eps(ep_ids):
    S, A, Sn = [], [], []
    for ep in ep_ids:
        d = store.load_episode(DATASET_ID, ep)
        obs = d["observation"]
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(d["actions"])
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase8] CUDA probe failed ({exc}); using CPU.")
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
    train_eps = store.splits(DATASET_ID)["train"]
    val = transitions_from_eps(store.splits(DATASET_ID)["val"])
    test_trans = transitions_from_eps(store.splits(DATASET_ID)["test_heldout"])
    test_eps = store.episodes(DATASET_ID, "test_heldout")

    # persistence baseline on held-out (one-step: predict no change)
    S, _, Sn = test_trans
    def rmse(a, b, cols):
        return float(np.sqrt(np.mean((a[:, cols] - b[:, cols]) ** 2)))
    persistence = {"object_pos_rmse_m": round(rmse(S, Sn, OBJ_COLS), 5),
                   "physical_state_rmse": round(rmse(S, Sn, PHYS_COLS), 5)}

    peak_vram = None
    rows = {}
    for b in BUDGETS_EP:
        objs, phys, ms_list, secs = [], [], [], []
        for s in range(N_SEEDS):
            rng = np.random.default_rng(s)
            ep_subset = (list(train_eps) if b >= len(train_eps)
                         else [int(train_eps[i]) for i in rng.choice(len(train_eps), b, replace=False)])
            tr = transitions_from_eps(ep_subset)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            m = aug.train_fresh(tr, val, device, seed=s, epochs=400, patience=40)
            secs.append(time.time() - t1)
            e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
            objs.append(e["object_pos_rmse_m"]); phys.append(e["physical_state_rmse"])
            ms_list.append(e["multistep_object_rmse_m"])
            if device == "cuda":
                peak_vram = max(peak_vram or 0, round(torch.cuda.max_memory_allocated() / 1e9, 3))
        om, osd = mean_std(objs)
        ms_mean = {h: round(float(np.mean([m[h] for m in ms_list if h in m])), 4)
                   for h in HORIZONS if any(h in m for m in ms_list)}
        rows[b] = {"n_transitions": int(len(transitions_from_eps(
                       list(train_eps) if b >= len(train_eps)
                       else [int(train_eps[i]) for i in np.random.default_rng(0).choice(len(train_eps), b, replace=False)])[0])),
                   "object_pos_rmse_m_mean": om, "object_pos_rmse_m_std": osd,
                   "physical_state_rmse_mean": mean_std(phys)[0],
                   "multistep_object_rmse_m": ms_mean,
                   "object_runs": [round(o, 5) for o in objs],
                   "train_seconds_mean": round(float(np.mean(secs)), 2)}
        print(f"[phase8] budget={b:3d} eps  object 1-step RMSE "
              f"{om:.5f} +/- {osd:.5f}  (persist {persistence['object_pos_rmse_m']})  "
              f"h10={ms_mean.get(10)}")

    # ---- gate: does the largest budget beat the smallest beyond seed spread? ----
    lo, hi = rows[BUDGETS_EP[0]], rows[BUDGETS_EP[-1]]
    delta = lo["object_pos_rmse_m_mean"] - hi["object_pos_rmse_m_mean"]
    spread = (lo["object_pos_rmse_m_std"] or 0) + (hi["object_pos_rmse_m_std"] or 0)
    headroom = bool(delta > spread and delta > 0)
    beats_persistence = bool(hi["object_pos_rmse_m_mean"] < persistence["object_pos_rmse_m"])

    report = {
        "phase": 8, "stage": "headroom_gate", "dataset_id": DATASET_ID, "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS,
                       "budgets_episodes": BUDGETS_EP, "peak_vram_gb": peak_vram},
        "metric": "held-out object-position one-step RMSE (m), lower=better; mean +/- std over seeds",
        "persistence_baseline": persistence,
        "by_budget": rows,
        "headroom_delta_m": round(delta, 5), "headroom_seed_spread_m": round(spread, 5),
        "model_beats_persistence_at_full": beats_persistence,
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": headroom,
    }
    (ROOT / "reports" / "phase8_headroom.json").write_text(json.dumps(report, indent=2))

    print("=" * 72)
    print("DREAM-OS Phase 8 -- learning-headroom gate (FetchSlide-v4)")
    print("=" * 72)
    print(f"  persistence (no-change) object 1-step RMSE: {persistence['object_pos_rmse_m']} m")
    for b in BUDGETS_EP:
        r = rows[b]
        print(f"   {b:3d} eps ({r['n_transitions']:5d} trans)  obj 1-step "
              f"{r['object_pos_rmse_m_mean']:.5f} +/- {r['object_pos_rmse_m_std']:.5f}   "
              f"phys {r['physical_state_rmse_mean']:.4f}   h10={r['multistep_object_rmse_m'].get(10)}")
    print(f"  more-data improvement (20->168 eps): {delta:+.5f} m  vs seed spread {spread:.5f} m")
    print(f"  model beats persistence at full budget: {beats_persistence}")
    print(f"  peak_vram={peak_vram}GB  wall={report['wall_seconds']}s  -> reports/phase8_headroom.json")
    print("HEADROOM GATE PASSED:", headroom)
    return 0 if headroom else 1


if __name__ == "__main__":
    raise SystemExit(main())
