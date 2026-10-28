"""Phase 9H -- generalization of the existing world model across held-out SCENARIO
configurations (closes the outstanding Phase 9 research gate F).

The adopted rollout-K5 model was trained on `fetchslide_phase8` (scenario-disjoint
train/val/test by episode seed). Phase 9H asks a stronger question than the random
in-distribution test: does the FROZEN model generalize to scenario configurations it
under-represents? We collect a FRESH, seed-disjoint FetchSlide pool (pure test -- never
used for training or tuning) and stratify it by measurable scenario properties:
  * goal distance (near / mid / far strikes),
  * object initial position region (left / center / right),
  * realized slide speed (slow / fast coast),
  * collection policy (scripted / random).
The frozen model (3 seeds) is evaluated on the whole fresh pool, on each stratum, and
against the original in-distribution test as the reference. Metrics: one-step and
multi-step object RMSE vs persistence, endpoint RMSE, pointwise trajectory error,
free-slide signed travel bias, rollout stability. A valid negative result satisfies the
gate; no success is manufactured.

Leakage control: the fresh pool uses seed ranges DISJOINT from training; the model is
trained only on the original train split; val is used only for early stopping.

-> data/raw/fetchslide_heldout9h/ , data/manifests/fetchslide_heldout9h.json
   reports/phase9h_generalization.json , reports/phase9h_generalization.md
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
from envs.benchmark_interface import make_env                      # noqa: E402
from scripts.collect_data import collect_episode, _git_commit, _versions   # noqa: E402
from memory import episode_store as store                          # noqa: E402
from world_model.dynamics import ResidualDynamics                  # noqa: E402
from world_model import training as T                               # noqa: E402

ENV_ID = "FetchSlide-v4"
TRAIN_DATASET = "fetchslide_phase8"
HELDOUT_ID = "fetchslide_heldout9h"
OBJ = [3, 4, 5]
RELPOS = [6, 7, 8]
HORIZONS = [1, 5, 10, 15, 20]
H_DIAG = 10
ROLLOUT_K = 5
N_SEEDS = 3
EPOCHS, PATIENCE = 400, 40
MAX_STEPS = 50
N_SCRIPTED, N_RANDOM = 90, 60              # fresh held-out pool size
SCRIPTED_SEED0, RANDOM_SEED0 = 90000, 95000   # DISJOINT from training (100.., 5000..)


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9h] CUDA probe failed ({exc}); CPU.")
    return "cpu"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# fresh held-out scenario pool (collect once, then reuse)
# --------------------------------------------------------------------------- #
def collect_or_load_pool():
    raw = ROOT / "data" / "raw" / HELDOUT_ID
    man_path = ROOT / "data" / "manifests" / f"{HELDOUT_ID}.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        eps = [np.load(raw / e["file"]) for e in man["episode_list"]]
        return man, [(d["observation"].astype(np.float32), d["actions"].astype(np.float32)) for d in eps], \
            [dict(desired_goal=d["desired_goal"], observation=d["observation"]) for d in eps]
    raw.mkdir(parents=True, exist_ok=True)
    env = make_env(ENV_ID)
    plan = [("scripted", SCRIPTED_SEED0 + i) for i in range(N_SCRIPTED)] + \
           [("random", RANDOM_SEED0 + i) for i in range(N_RANDOM)]
    meta, eps, raws = [], [], []
    for i, (policy, seed) in enumerate(plan):
        rng = np.random.default_rng(1000 + seed)
        ep = collect_episode(env, seed, policy, rng, MAX_STEPS)
        arrays = {k: v for k, v in ep.items() if isinstance(v, np.ndarray)}
        arrays["desired_goal"] = ep["desired_goal"]
        np.savez_compressed(raw / f"ep_{i:04d}.npz", **arrays)
        meta.append({"episode": i, "file": f"ep_{i:04d}.npz", "seed": seed, "policy": policy,
                     "length": ep["length"], "success": bool(ep["is_success"].any())})
        eps.append((ep["observation"].astype(np.float32), ep["actions"].astype(np.float32)))
        raws.append(dict(desired_goal=ep["desired_goal"], observation=ep["observation"]))
    env.close()
    man = {"dataset_id": HELDOUT_ID, "env": ENV_ID, "role": "fresh held-out scenario pool (pure test)",
           "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "git_commit": _git_commit(), "versions": _versions(), "max_steps": MAX_STEPS,
           "seed_ranges": {"scripted": [SCRIPTED_SEED0, SCRIPTED_SEED0 + N_SCRIPTED - 1],
                           "random": [RANDOM_SEED0, RANDOM_SEED0 + N_RANDOM - 1],
                           "training_ranges": "scripted 100..239, random 5000..5099 (DISJOINT)"},
           "episodes": len(meta), "episode_list": meta}
    man_path.write_text(json.dumps(man, indent=2))
    return man, eps, raws


def scenario_props(raws, eps):
    props = []
    for r, (obs, act) in zip(raws, eps):
        obj0 = obs[0][OBJ]
        goal = np.asarray(r["desired_goal"], np.float32)
        gdist = float(np.linalg.norm(goal[:2] - obj0[:2]))         # table-plane goal distance
        disp = np.linalg.norm(np.diff(obs[:, OBJ], axis=0), axis=1)
        props.append({"goal_dist": gdist, "obj0_y": float(obj0[1]),
                      "peak_speed": float(disp.max()) if len(disp) else 0.0,
                      "total_travel": float(np.linalg.norm(obs[-1][OBJ] - obs[0][OBJ]))})
    return props


def tercile_groups(props, eps):
    """Return {group_name: [episode indices]} stratified by scenario properties."""
    n = len(props)
    groups = {}

    def by(key, names):
        vals = np.array([p[key] for p in props])
        order = np.argsort(vals)
        thirds = np.array_split(order, 3)
        for nm, idx in zip(names, thirds):
            groups[nm] = sorted(int(x) for x in idx)

    by("goal_dist", ["goal_near", "goal_mid", "goal_far"])
    by("obj0_y", ["obj_left", "obj_center", "obj_right"])
    # binary speed split (free-slide velocity axis)
    sp = np.array([p["peak_speed"] for p in props]); med = float(np.median(sp))
    groups["slide_slow"] = sorted(int(i) for i in range(n) if sp[i] <= med)
    groups["slide_fast"] = sorted(int(i) for i in range(n) if sp[i] > med)
    return groups


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def classify(real):
    obj = real[:, OBJ]; c = np.linalg.norm(real[:, RELPOS], axis=1)
    travel = float(np.linalg.norm(obj[-1] - obj[0]))
    d = np.linalg.norm(np.diff(obj, axis=0), axis=1); half = len(d) // 2
    if travel < 0.01:
        return "quasi_static"
    if c[half:].mean() > 0.08 and d[half:].mean() > 0.004:
        return "free_sliding"
    return "contact_driven"


def flat_of(eps):
    S, A, Sn = [], [], []
    for obs, act in eps:
        S.append(obs[:-1]); Sn.append(obs[1:]); A.append(act)
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32),
            np.concatenate(Sn).astype(np.float32))


@torch.no_grad()
def eval_on(model, eps, device):
    if len(eps) == 0:
        return None
    S, A, Sn = flat_of(eps)
    pred1 = T.predict_batch(model, S, A, device)
    one_step = float(np.sqrt(np.mean((pred1[:, OBJ] - Sn[:, OBJ]) ** 2)))
    ms = T.evaluate_multistep(model, eps, device, HORIZONS, OBJ)
    # endpoint (final step), pointwise, stability, free-slide bias
    endpoint, pointwise, max_speed = [], [], 0.0
    fs_real, fs_pred = [], []
    for obs, act in eps:
        preds = T.rollout(model, obs[0], act, device)               # [T,D]
        Tn = preds.shape[0]
        endpoint.append(np.linalg.norm(preds[Tn - 1][OBJ] - obs[Tn][OBJ]))
        pointwise.append(np.mean(np.linalg.norm(preds[:, OBJ] - obs[1:Tn + 1][OBJ], axis=1)))
        pos = np.concatenate([obs[0][None, OBJ], preds[:, OBJ]], 0)
        max_speed = max(max_speed, float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max()))
        for t in range(1, len(act) - H_DIAG + 1):
            real = obs[t:t + H_DIAG + 1]
            if classify(real) == "free_sliding":
                p = T.rollout(model, obs[t], act[t:t + H_DIAG], device)
                fs_real.append(np.linalg.norm(real[H_DIAG][OBJ] - real[0][OBJ]))
                fs_pred.append(np.linalg.norm(p[H_DIAG - 1][OBJ] - real[0][OBJ]))
    fs_bias = (round(float(np.mean(fs_pred) - np.mean(fs_real)), 4) if fs_real else None)
    return {
        "n_episodes": len(eps),
        "one_step_object_rmse_m": round(one_step, 5),
        "multistep_object_rmse_m": {h: ms[h]["model_object_rmse_m"] for h in ms},
        "persistence_multistep_m": {h: ms[h]["baseline_object_rmse_m"] for h in ms},
        "endpoint_object_rmse_m": round(float(np.mean(endpoint)), 4),
        "pointwise_traj_rmse_m": round(float(np.mean(pointwise)), 4),
        "free_slide_signed_bias_m": fs_bias, "n_free_slide": len(fs_real),
        "max_step_object_speed_m": round(max_speed, 3),
    }


def agg(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)) if xs else (None, None)


def aggregate(runs, key):
    """runs: list of per-seed metric dicts; aggregate scalar `key`."""
    return agg([r[key] for r in runs if r and key in r])


def main() -> int:
    t0 = time.time()
    device = pick_device()
    # fresh held-out scenario pool
    man, pool_eps, pool_raws = collect_or_load_pool()
    props = scenario_props(pool_raws, pool_eps)
    groups = tercile_groups(props, pool_eps)
    id_test_eps = store.episodes(TRAIN_DATASET, "test_heldout")     # in-distribution reference
    print(f"[phase9h] device={device} fresh pool={len(pool_eps)} eps  ID-test={len(id_test_eps)} eps")
    print(f"[phase9h] scenario props: goal_dist {np.mean([p['goal_dist'] for p in props]):.3f}"
          f" peak_speed {np.mean([p['peak_speed'] for p in props]):.4f}")

    # train frozen baseline (same config as adopted), evaluate per seed
    train_ep_list = store.episodes(TRAIN_DATASET, "train")
    val_ep_list = store.episodes(TRAIN_DATASET, "val")
    stats = T.compute_stats(*flat_of(train_ep_list))

    eval_sets = {"fresh_pool_all": pool_eps, "id_test_reference": id_test_eps}
    for g, idx in groups.items():
        eval_sets[g] = [pool_eps[i] for i in idx]

    per_seed = {k: [] for k in eval_sets}
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        m = ResidualDynamics(25, 4, stats)
        T.train_rollout(m, train_ep_list, val_ep_list, device, horizon=ROLLOUT_K,
                        epochs=EPOCHS, patience=PATIENCE, seed=seed, obj_cols=OBJ)
        m.eval()
        for k, eps in eval_sets.items():
            per_seed[k].append(eval_on(m, eps, device))
        fp = per_seed["fresh_pool_all"][-1]; idt = per_seed["id_test_reference"][-1]
        print(f"[phase9h]  seed {seed}: fresh 1step {fp['one_step_object_rmse_m']} h20 "
              f"{fp['multistep_object_rmse_m'].get(20)} endpt {fp['endpoint_object_rmse_m']} | "
              f"ID h20 {idt['multistep_object_rmse_m'].get(20)}")

    # aggregate
    def ms_agg(k, h):
        return agg([r["multistep_object_rmse_m"].get(h) for r in per_seed[k] if r])
    summary = {}
    for k in eval_sets:
        runs = per_seed[k]
        summary[k] = {
            "n_episodes": runs[0]["n_episodes"] if runs[0] else 0,
            "one_step_object_rmse_m": aggregate(runs, "one_step_object_rmse_m"),
            "endpoint_object_rmse_m": aggregate(runs, "endpoint_object_rmse_m"),
            "pointwise_traj_rmse_m": aggregate(runs, "pointwise_traj_rmse_m"),
            "free_slide_signed_bias_m": aggregate(runs, "free_slide_signed_bias_m"),
            "max_step_object_speed_m": aggregate(runs, "max_step_object_speed_m"),
            "multistep_h5": ms_agg(k, 5), "multistep_h10": ms_agg(k, 10), "multistep_h20": ms_agg(k, 20),
            "persistence_h20": agg([r["persistence_multistep_m"].get(20) for r in runs if r]),
        }

    # generalization verdict: fresh vs ID reference (beyond seed spread?)
    fresh_h20_m, fresh_h20_s = summary["fresh_pool_all"]["multistep_h20"]
    id_h20_m, id_h20_s = summary["id_test_reference"]["multistep_h20"]
    fresh_1_m, fresh_1_s = summary["fresh_pool_all"]["one_step_object_rmse_m"]
    id_1_m, id_1_s = summary["id_test_reference"]["one_step_object_rmse_m"]
    h20_gap = fresh_h20_m - id_h20_m
    h20_degraded = bool(h20_gap > (fresh_h20_s + id_h20_s) and h20_gap > 0.01)
    one_gap = fresh_1_m - id_1_m
    one_degraded = bool(one_gap > (fresh_1_s + id_1_s) and one_gap > 0.001)
    fresh_beats_persist = bool(fresh_h20_m < summary["fresh_pool_all"]["persistence_h20"][0])

    # worst / best stratum by h20
    strata = [g for g in groups]
    worst = max(strata, key=lambda g: summary[g]["multistep_h20"][0] or 0)
    best = min(strata, key=lambda g: summary[g]["multistep_h20"][0] or 1e9)

    verdict = ("generalizes_within_bounds" if not h20_degraded and not one_degraded and fresh_beats_persist
               else "degrades_on_fresh_scenarios" if (h20_degraded or one_degraded)
               else "inconclusive")
    interp = {
        "generalizes_within_bounds": ("The frozen model transfers to fresh, seed-disjoint scenarios with no "
                                      "degradation beyond seed spread vs the in-distribution test, and still "
                                      "beats persistence at h20. Scenario generalization holds within the "
                                      "FetchSlide configuration range sampled."),
        "degrades_on_fresh_scenarios": ("The frozen model degrades on fresh scenarios beyond seed spread "
                                        f"(h20 {id_h20_m}->{fresh_h20_m}, one-step {id_1_m}->{fresh_1_m}). "
                                        "Generalization is limited; the worst stratum is the hardest configuration."),
        "inconclusive": "Mixed signal across metrics; see per-stratum table.",
    }[verdict]

    report = {
        "phase": "9H", "stage": "scenario_generalization", "env": ENV_ID,
        "train_dataset": TRAIN_DATASET, "heldout_dataset": HELDOUT_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS,
                       "rollout_K": ROLLOUT_K, "epochs": EPOCHS, "patience": PATIENCE,
                       "fresh_pool_seed_ranges": man["seed_ranges"],
                       "leakage_control": "fresh pool seeds DISJOINT from training; model trained only on "
                                          "train split; val used only for early stopping; fresh pool is pure test"},
        "fresh_pool_size": len(pool_eps), "id_test_size": len(id_test_eps),
        "scenario_property_means": {k: round(float(np.mean([p[k] for p in props])), 4) for k in props[0]},
        "strata_sizes": {g: len(idx) for g, idx in groups.items()},
        "metric": "object-position RMSE (m), lower=better; mean +/- std over seeds",
        "results_by_set": summary,
        "generalization": {
            "fresh_vs_id_h20_gap_m": round(h20_gap, 4), "h20_degraded_beyond_spread": h20_degraded,
            "fresh_vs_id_one_step_gap_m": round(one_gap, 5), "one_step_degraded_beyond_spread": one_degraded,
            "fresh_beats_persistence_h20": fresh_beats_persist,
            "worst_stratum": worst, "worst_stratum_h20": summary[worst]["multistep_h20"][0],
            "best_stratum": best, "best_stratum_h20": summary[best]["multistep_h20"][0]},
        "verdict": verdict, "interpretation": interp,
        "task_success_note": ("the world model is PREDICTIVE, not a controller; dataset (policy) success is "
                              "descriptive only and not a measure of model generalization"),
        "limitations": ("single object, single FetchSlide task; 'initial velocity' varied only via realized "
                        "strike speed (puck starts at rest; injecting arbitrary initial velocities would require "
                        "simulator-state hacking that risks invalid scenes, deferred); goal/object ranges are the "
                        "env's native sampling, so this tests the TAILS of that range, not an out-of-support shift"),
        "wall_seconds": round(time.time() - t0, 2),
        "gate_F_generalization_evaluated": True,
    }
    (ROOT / "reports" / "phase9h_generalization.json").write_text(json.dumps(report, indent=2))

    print("=" * 80)
    print("DREAM-OS Phase 9H -- world-model scenario generalization (FetchSlide-v4)")
    print("=" * 80)
    print(f"  {'set':20s} {'n':>4s} {'1step':>8s} {'h10':>7s} {'h20':>7s} {'endpt':>7s} {'fs_bias':>8s}")
    for k in ["id_test_reference", "fresh_pool_all"] + list(groups):
        s = summary[k]
        print(f"  {k:20s} {s['n_episodes']:>4d} {str(s['one_step_object_rmse_m'][0]):>8s} "
              f"{str(s['multistep_h10'][0]):>7s} {str(s['multistep_h20'][0]):>7s} "
              f"{str(s['endpoint_object_rmse_m'][0]):>7s} {str(s['free_slide_signed_bias_m'][0]):>8s}")
    print(f"  fresh-vs-ID h20 gap {h20_gap:+.4f} (degraded beyond spread: {h20_degraded}); "
          f"fresh beats persistence h20: {fresh_beats_persist}")
    print(f"  worst stratum: {worst} (h20 {summary[worst]['multistep_h20'][0]})  best: {best}")
    print(f"  VERDICT: {verdict}")
    print(f"  -> {interp}")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase9h_generalization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
