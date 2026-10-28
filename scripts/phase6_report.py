"""Phase 6 -- does verified imagination improve the world model?

The honest, budgeted experiment. Real interaction is the expensive resource. We
have a POOL of candidate action sequences (cheap to imagine) and a BUDGET of B
real rollouts we may actually execute. Question: does using the world model's
imagination to CHOOSE which experiments to run yield a better model, per unit of
real interaction, than choosing at random -- and do bad dreams degrade it?

Conditions (every augmenting condition adds the SAME number of real transitions):
  base              -- no new data                              (control / floor)
  random            -- B randomly chosen real rollouts          (multi-seed)
  dream_composite   -- B rollouts imagination flags informative (uncertainty+coverage)
  dream_uncertainty -- B by ensemble uncertainty only           (ablation)
  oracle            -- B rollouts with truly-highest real error (ceiling, not usable)
  imagined_harmful  -- composite selection, trained on the model's OWN imagined
                       transitions instead of the real ones     (safety test)

TWO PANELS, because selection cannot be judged without controlling WHAT the dreams
explore:
  OFF-distribution  -- exploratory dreams (constant pushes / random / noop) far from
                       the scripted-pusher test distribution. Expected: negative
                       transfer; "most informative" dreams are the least useful.
  IN-distribution   -- the real recorded pusher trajectory plus small perturbations,
                       close to deployment. Here selection (composite/oracle vs
                       random) is tested fairly, without the distribution-shift confound.

Data hygiene: augmentation starts come from the TRAIN split; the evaluation set is
the FIXED Phase-1 held-out split, never trained on nor used to pick augmentations.
Improvement is measured in real units on real test data only -- the model never
certifies its own predictions. Checkpoint rollback: any condition that does not
beat base on held-out is REJECTED, not adopted.

Gate is METHODOLOGICAL (matched budget, disjoint eval, multi-seed error bars,
oracle ceiling + random floor + base control + harmful test present, rollback
decided, both panels run). It is NOT "dreams win"; the outcome is reported either way.
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
from envs.benchmark_interface import make_env               # noqa: E402
from memory import episode_store as store, retrieval        # noqa: E402
from world_model import training as T, ensemble as ens      # noqa: E402
from world_model import augmentation as aug                 # noqa: E402
from dreaming import selection as sel                        # noqa: E402
from verification import replay as vr, metrics as vm         # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
OBJ_COLS = [3, 4, 5]
ROT_COLS = [11, 12, 13]
PHYS_COLS = [i for i in range(25) if i not in ROT_COLS]
HORIZONS = [1, 5, 10, 20]

N_MODELS = 5          # ensemble (the imaginer); trained on base train only
M_STARTS = 25         # distinct real start states (from TRAIN split -> disjoint from eval)
HORIZON = 15          # length of each candidate rollout
BUDGET_B = 30         # real rollouts the budget allows (pool is M_STARTS * 6 = 150)
N_SEEDS = 5           # init seeds (and random-selection seeds) for error bars
PUSH_MAG = 0.8
INFO_CAP_M = 1.0


def explore_seqs(ref, d_act, H, seed):
    """OFF-distribution: exploratory open-loop action sequences (6 branches)."""
    seqs = {"noop": np.zeros((H, d_act), np.float32)}
    seqs["random"] = np.random.default_rng(seed).uniform(-1, 1, (H, d_act)).astype(np.float32)
    for ang in (0, 90, 180, 270):
        a = np.zeros((H, d_act), np.float32)
        r = np.deg2rad(ang)
        a[:, 0], a[:, 1], a[:, 2] = PUSH_MAG * np.cos(r), PUSH_MAG * np.sin(r), -0.05
        seqs[f"push{ang}"] = a
    return seqs


def indist_seqs(ref, d_act, H, seed):
    """IN-distribution: the real recorded pusher continuation + small perturbations."""
    d = store.load_episode(ref["dataset_id"], ref["episode"])
    t = ref["t"]
    base = d["actions"][t:t + H].astype(np.float32)
    if len(base) < H:                                   # pad by repeating the last action
        base = np.vstack([base, np.repeat(base[-1:], H - len(base), 0)]).astype(np.float32)
    rng = np.random.default_rng(seed)

    def clip(x):
        return np.clip(x, -1, 1).astype(np.float32)

    return {
        "replay_true": base.copy(),
        "noise_s": clip(base + rng.normal(0, 0.10, base.shape)),
        "noise_m": clip(base + rng.normal(0, 0.25, base.shape)),
        "noise_l": clip(base + rng.normal(0, 0.40, base.shape)),
        "scale_lo": clip(base * 0.8),
        "scale_hi": clip(base * 1.2),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def mean_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 5), round(float(np.std(xs)), 5)) if xs else (None, None)


def pick_device():
    """Probe CUDA with a real op; fall back to CPU if init is flaky (tiny models)."""
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:                                  # transient driver init failure
            print(f"[phase6] CUDA probe failed ({exc}); using CPU (models are tiny).")
    return "cpu"


def build_pool(seq_fn, starts, models, stats, train_norm, d_act, device, env):
    """Imagine every candidate (cheap) and execute it in the real sim (the budgeted cost)."""
    cands, max_recon = [], 0.0
    for si, ref in enumerate(starts):
        for bname, acts in seq_fn(ref, d_act, HORIZON, 1000 + si).items():
            preds = ens.rollout_ensemble(models, ref["observation"], acts, device)
            mean_traj, std_traj = ens.mean_std(preds)
            finite = bool(np.all(np.isfinite(mean_traj)))
            unc = sel.uncertainty_score(std_traj) if finite else 1e9
            nov = sel.novelty_score(ref["observation"], train_norm, stats["s_mean"], stats["s_std"])
            real, recon_err = vr.reconstruct_and_replay(
                env, ref["dataset_id"], ref["episode"], ref["t"], acts)
            max_recon = max(max_recon, recon_err)
            td = (float(np.mean(vm.object_discrepancy(mean_traj, real)[1:]))
                  if finite else float("inf"))
            info = INFO_CAP_M if not np.isfinite(td) else min(td, INFO_CAP_M)
            cands.append({"start": si, "branch": bname, "uncertainty": unc, "novelty": nov,
                          "informativeness": round(info, 4),
                          "real_trans": aug.rollout_transitions(real, acts),
                          "imagined_start": ref["observation"], "actions": acts})
    return cands, max_recon


def run_panel(cands, base_train, val, test_trans, test_eps, base_mean, models, device):
    """Score the pool, run every condition under a matched budget, decide rollback."""
    B = BUDGET_B
    unc = np.array([c["uncertainty"] for c in cands])
    nov = np.array([c["novelty"] for c in cands])
    info = np.array([c["informativeness"] for c in cands])
    composite = sel.composite_score(unc, nov)
    picks = {"dream_composite": sel.select_topk(composite, B),
             "dream_uncertainty": sel.select_topk(unc, B),
             "oracle": sel.select_topk(info, B)}

    def real_extra(idx_list):
        parts = [cands[i]["real_trans"] for i in idx_list]
        return tuple(np.concatenate([p[j] for p in parts]).astype(np.float32) for j in range(3))

    def condition(extra, seeds):
        objs, phys, ms = [], [], []
        for s in seeds:
            m = aug.train_fresh(aug.concat_sets(base_train, extra), val, device, seed=s)
            e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
            objs.append(e["object_pos_rmse_m"]); phys.append(e["physical_state_rmse"]); ms.append(e["multistep_object_rmse_m"])
        om, osd = mean_std(objs)
        ms_mean = {h: round(float(np.mean([m[h] for m in ms if h in m])), 4)
                   for h in HORIZONS if any(h in m for m in ms)}
        return {"object_pos_rmse_m_mean": om, "object_pos_rmse_m_std": osd,
                "physical_state_rmse_mean": mean_std(phys)[0], "multistep_object_rmse_m": ms_mean,
                "runs": [round(o, 5) for o in objs],
                "n_transitions_added": 0 if extra is None else len(extra[0])}

    seeds = list(range(N_SEEDS))
    res = {}

    # random floor: selection AND init both vary with the seed
    r_objs, r_phys, r_ms, r_added = [], [], [], []
    for s in seeds:
        idx = sel.select_random(len(cands), B, seed=s)
        tr = aug.concat_sets(base_train, real_extra(idx))
        m = aug.train_fresh(tr, val, device, seed=s)
        e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
        r_objs.append(e["object_pos_rmse_m"]); r_phys.append(e["physical_state_rmse"])
        r_ms.append(e["multistep_object_rmse_m"]); r_added.append(len(tr[0]) - len(base_train[0]))
    rom, rosd = mean_std(r_objs)
    res["random"] = {"object_pos_rmse_m_mean": rom, "object_pos_rmse_m_std": rosd,
                     "physical_state_rmse_mean": mean_std(r_phys)[0],
                     "multistep_object_rmse_m": {h: round(float(np.mean([m[h] for m in r_ms if h in m])), 4)
                                                 for h in HORIZONS if any(h in m for m in r_ms)},
                     "runs": [round(o, 5) for o in r_objs],
                     "n_transitions_added": int(np.max(r_added)), "seeds": N_SEEDS}

    for name, idx in picks.items():
        res[name] = condition(real_extra(idx), seeds)

    # harmful-dreams safety test: same selection, model's OWN imagined transitions
    imag_parts, dropped = [], 0
    for i in picks["dream_composite"]:
        it, d = aug.finite_mask(aug.imagined_transitions(
            models[0], cands[i]["imagined_start"], cands[i]["actions"], device))
        dropped += d
        if len(it[0]):
            imag_parts.append(it)
    imag_extra = tuple(np.concatenate([p[j] for p in imag_parts]).astype(np.float32)
                       for j in range(3)) if imag_parts else None
    res["imagined_harmful"] = condition(imag_extra, seeds)
    res["imagined_harmful"]["imagined_transitions_dropped_nonfinite"] = dropped

    # checkpoint-rollback decision: adopt only if it beats base on held-out
    rm, rsd = res["random"]["object_pos_rmse_m_mean"], res["random"]["object_pos_rmse_m_std"] or 0.0
    decisions = {}
    for name, r in res.items():
        m = r["object_pos_rmse_m_mean"]
        r["beats_base"] = bool(m < base_mean)
        r["beats_random"] = bool(m < rm - rsd)
        decisions[name] = "adopt" if r["beats_base"] else "rollback_to_base"
    return res, decisions


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

    print(f"[phase6] training ensemble of {N_MODELS} (the imaginer) on base train ...")
    models = ens.train_ensemble(base_train, val, N_MODELS, device, epochs=300, patience=30)

    # base control (identical across panels): no new data, N_SEEDS inits
    print("[phase6] base control ...")
    base_objs, base_ms_list = [], []
    for s in range(N_SEEDS):
        m = aug.train_fresh(base_train, val, device, seed=s)
        e = aug.eval_on_test(m, test_trans, test_eps, device, OBJ_COLS, PHYS_COLS, HORIZONS)
        base_objs.append(e["object_pos_rmse_m"]); base_ms_list.append(e["multistep_object_rmse_m"])
    base_mean, base_std = mean_std(base_objs)
    base_ms_mean = {h: round(float(np.mean([m[h] for m in base_ms_list if h in m])), 4)
                    for h in HORIZONS if any(h in m for m in base_ms_list)}
    base_res = {"object_pos_rmse_m_mean": base_mean, "object_pos_rmse_m_std": base_std,
                "multistep_object_rmse_m": base_ms_mean,
                "runs": [round(o, 5) for o in base_objs], "n_transitions_added": 0,
                "rollback_reference": True}

    starts = retrieval.sample_start_states(DATASET_ID, "train", n=M_STARTS, seed=1)
    env = make_env("FetchPush-v4")
    panels, max_recon_all = {}, 0.0
    for pname, seq_fn in (("off_distribution", explore_seqs),
                          ("in_distribution", indist_seqs)):
        print(f"[phase6] panel: {pname}  (build pool + run conditions) ...")
        cands, max_recon = build_pool(seq_fn, starts, models, stats, train_norm, d_act, device, env)
        max_recon_all = max(max_recon_all, max_recon)
        res, decisions = run_panel(cands, base_train, val, test_trans, test_eps,
                                   base_mean, models, device)
        panels[pname] = {"conditions": res, "rollback_decisions": decisions,
                         "pool_size": len(cands)}
    env.close()

    # ---- methodological gate ----
    train_eps_set = set(store.splits(DATASET_ID)["train"])
    test_eps_set = set(store.splits(DATASET_ID)["test_heldout"])
    disjoint = train_eps_set.isdisjoint(test_eps_set)
    real_added = set()
    for p in panels.values():
        for k in ("random", "dream_composite", "dream_uncertainty", "oracle"):
            real_added.add(p["conditions"][k]["n_transitions_added"])
    gate = bool(
        real_added == {BUDGET_B * HORIZON}                       # matched real budget everywhere
        and disjoint                                             # no train/test leakage
        and max_recon_all < vm.RECON_TOL_M                       # starts reconstructed exactly
        and base_std is not None                                 # base error bar present
        and all(p["conditions"]["random"]["object_pos_rmse_m_std"] is not None
                for p in panels.values())                        # random error bars present
        and all(len(p["rollback_decisions"]) == 5 for p in panels.values())  # all non-base decided
        and all("oracle" in p["conditions"] and "imagined_harmful" in p["conditions"]
                for p in panels.values())                        # ceiling + safety test present
    )

    report = {
        "phase": 6, "dataset_id": DATASET_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "ensemble_size": N_MODELS,
                       "m_starts": M_STARTS, "horizon": HORIZON, "budget_B": BUDGET_B,
                       "n_seeds": N_SEEDS, "seeds": list(range(N_SEEDS))},
        "metric": "held-out object-position RMSE (m), lower=better; mean +/- std over init "
                  "seeds; measured on real test data only (model never certifies itself)",
        "max_reconstruction_err_m": round(max_recon_all, 8),
        "augmentation_starts_disjoint_from_test": disjoint,
        "real_transitions_added_per_condition": sorted(real_added),
        "base": base_res,
        "panels": panels,
        "wall_seconds": round(time.time() - t0, 2),
        "gate_passed": gate,
    }
    (ROOT / "reports" / "phase6_results.json").write_text(json.dumps(report, indent=2))

    # ---- teaching summary ----
    order = ["random", "dream_uncertainty", "dream_composite", "oracle", "imagined_harmful"]
    print("=" * 78)
    print("DREAM-OS Phase 6 report  (does verified imagination improve the model?)")
    print("=" * 78)
    print(f"  budget B={BUDGET_B} real rollouts  horizon={HORIZON}  seeds={N_SEEDS}  "
          f"ensemble={N_MODELS}  device={device}")
    print(f"  real transitions added per augmenting condition: {sorted(real_added)}  (matched)")
    print(f"  aug starts disjoint from eval test: {disjoint}   "
          f"start reconstruction max err: {max_recon_all:.1e} m")
    print(f"  metric = held-out object-pos RMSE (m), lower=better  [mean +/- std over seeds]")
    print(f"     {'base (control)':20s} {base_mean:.5f} +/- {base_std:.5f}   "
          f"h10={base_res['multistep_object_rmse_m'].get(10)}")
    for pname in ("off_distribution", "in_distribution"):
        res, dec = panels[pname]["conditions"], panels[pname]["rollback_decisions"]
        print(f"  [{pname}]")
        for name in order:
            r = res[name]
            m, sd = r["object_pos_rmse_m_mean"], r["object_pos_rmse_m_std"]
            h10 = r["multistep_object_rmse_m"].get(10)
            print(f"     {name:20s} {m:.5f} +/- {sd:.5f}   h10={h10}   "
                  f"beats_base={r['beats_base']!s:5s} beats_rand={r['beats_random']!s:5s} "
                  f"-> {dec[name]}")
        hd = res["imagined_harmful"]
        print(f"     (harmful test used {hd['n_transitions_added']} imagined transitions, "
              f"{hd['imagined_transitions_dropped_nonfinite']} dropped non-finite)")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase6_results.json")
    print("GATE PASSED (fair protocol):", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
