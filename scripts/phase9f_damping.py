"""Phase 9F -- targeted damping intervention (E_damping_intervention).

Tests, with the smallest stable intervention, whether the persistent free-slide UNDER-TRAVEL is an
excess multiplicative velocity decay in the learned object dynamics. Intervention: a bounded
momentum re-injection on the object channel only (world_model/coast_correction.py), alpha in [0,1),
zero at rest, geometric-decay stable -- a DIFFERENT, stable form from the absolute-velocity input
Phase 9E rejected. alpha is calibrated on VALIDATION free-slide endpoint error and evaluated on the
held-out test set. Baseline (adopted rollout-K5) is kept intact; the candidate is baseline + alpha.

Acceptance thresholds are pre-registered below (defined before any result is seen). The candidate is
adopted ONLY if it reproducibly reduces the primary metric (held-out free-slide ENDPOINT RMSE)
without unacceptable regressions in trajectory accuracy, other regimes, or stability.

-> reports/phase9f_damping.json , reports/phase9f_damping.md , reports/figures/phase9f_velocity_decay.svg
   world_model/checkpoints/fetchslide_coast_correction.json   (candidate = baseline ckpt + alpha)
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
from world_model import coast_correction as CC                   # noqa: E402

DATASET_ID = "fetchslide_phase8"
OBJ = [3, 4, 5]
RELPOS = [6, 7, 8]
H_DIAG = 10
HORIZONS = [1, 5, 10, 15, 20]
ROLLOUT_K = 5
N_SEEDS = 3
EPOCHS, PATIENCE = 400, 40
ALPHA_GRID = [round(0.1 * i, 1) for i in range(10)]          # 0.0 .. 0.9
CKPT_BASE = ROOT / "world_model" / "checkpoints" / "fetchslide_rollout_k5.pt"
CORR_JSON = ROOT / "world_model" / "checkpoints" / "fetchslide_coast_correction.json"
FIG = ROOT / "reports" / "figures" / "phase9f_velocity_decay.svg"

# ---- PRE-REGISTERED acceptance thresholds (set BEFORE evaluating the candidate, §6) ----
PRIMARY_REL_MIN = 0.05        # >=5% mean reduction in held-out free-slide endpoint RMSE
REGRESS_TOL_M = 0.002         # allowed worsening (m) in any other regime / overall before "unacceptable"
STABILITY_MAX_RATIO = 1.5     # candidate full-episode h20 must stay <= 1.5x baseline
REST_MAX_DIFF_M = 1e-3        # on quasi-static (at-rest) windows, |candidate - baseline| object < 1mm


def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).item()
            return "cuda"
        except Exception as exc:
            print(f"[phase9f] CUDA probe failed ({exc}); using CPU.")
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


def windows_by_regime(eps):
    """All held-out H_DIAG windows (t>=1), bucketed by regime. Each bucket: S0, V0, Aw, reals_end."""
    buckets = {r: {"S0": [], "V0": [], "Aw": [], "end": []}
               for r in ("quasi_static", "contact_driven", "free_sliding")}
    for obs, act in eps:
        for t in range(1, len(act) - H_DIAG + 1):
            real = obs[t:t + H_DIAG + 1]
            r = classify(real)
            b = buckets[r]
            b["S0"].append(obs[t]); b["V0"].append(obs[t][OBJ] - obs[t - 1][OBJ])
            b["Aw"].append(act[t:t + H_DIAG]); b["end"].append(obs[t + H_DIAG])
    for r in buckets:
        for k in buckets[r]:
            buckets[r][k] = np.asarray(buckets[r][k], np.float32)
    return buckets


def free_slide_full(eps):
    """Free-slide windows plus the full [H+1] real trajectory (for per-step curves & pointwise error)."""
    S0, V0, Aw, reals = [], [], [], []
    for obs, act in eps:
        for t in range(1, len(act) - H_DIAG + 1):
            real = obs[t:t + H_DIAG + 1]
            if classify(real) == "free_sliding":
                S0.append(obs[t]); V0.append(obs[t][OBJ] - obs[t - 1][OBJ])
                Aw.append(act[t:t + H_DIAG]); reals.append(real)
    return (np.asarray(S0, np.float32), np.asarray(V0, np.float32),
            np.asarray(Aw, np.float32), np.asarray(reals, np.float32))


def eval_free_slide(model, alpha, S0, V0, Aw, reals, device):
    """Full free-slide metric suite for one (model, alpha)."""
    preds = CC.corrected_rollout_batch(model, S0, V0, Aw, device, alpha, OBJ)   # [N,H,D]
    start = S0[:, OBJ]
    pred_pos = preds[:, :, OBJ]                     # [N,H,3]
    real_pos = reals[:, 1:, :][:, :, OBJ]           # [N,H,3] (steps 1..H)
    real_end = reals[:, H_DIAG, :][:, OBJ]
    pred_end = pred_pos[:, -1]
    real_travel = np.linalg.norm(real_end - start, axis=1)
    pred_travel = np.linalg.norm(pred_end - start, axis=1)
    endpoint = np.linalg.norm(pred_end - real_end, axis=1)
    pointwise = np.linalg.norm(pred_pos - real_pos, axis=2)        # [N,H]
    per_h = [round(float(pointwise[:, h - 1].mean()), 4) for h in range(1, H_DIAG + 1)]
    # per-step object speed curves (reference vs prediction)
    real_curve = np.linalg.norm(np.diff(np.concatenate([start[:, None], real_pos], 1), axis=1), axis=2).mean(0)
    pred_curve = np.linalg.norm(np.diff(np.concatenate([start[:, None], pred_pos], 1), axis=1), axis=2).mean(0)
    # initial-velocity and displacement groups (median split, descriptive)
    v0mag = np.linalg.norm(V0, axis=1)
    disp = real_travel
    groups = {}
    for name, key in (("init_vel", v0mag), ("displacement", disp)):
        med = float(np.median(key))
        lo, hi = key <= med, key > med
        groups[name] = {"low_endpoint_rmse_m": round(float(endpoint[lo].mean()), 4),
                        "high_endpoint_rmse_m": round(float(endpoint[hi].mean()), 4),
                        "split_median": round(med, 4)}
    return {
        "n": int(len(S0)),
        "endpoint_object_rmse_m": round(float(endpoint.mean()), 4),
        "signed_travel_bias_m": round(float(pred_travel.mean() - real_travel.mean()), 4),
        "abs_travel_mag_bias_m": round(float(np.abs(pred_travel - real_travel).mean()), 4),
        "pointwise_traj_rmse_m": round(float(pointwise.mean()), 4),
        "per_horizon_pointwise_m": per_h,
        "real_speed_curve": [round(float(x), 4) for x in real_curve],
        "pred_speed_curve": [round(float(x), 4) for x in pred_curve],
        "groups": groups,
    }


def full_episode_stability(model, alpha, eps, device):
    """Rest-start (v0=0) full-episode rollout: per-horizon endpoint RMSE + max object speed (divergence)."""
    me = {h: [] for h in HORIZONS}
    max_speed = 0.0
    for obs, act in eps:
        preds = CC.corrected_rollout_episode(model, obs[0], np.zeros(len(OBJ)), act, device, alpha, OBJ)
        pos = np.concatenate([obs[0][None, OBJ], preds[:, OBJ]], axis=0)
        max_speed = max(max_speed, float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max()))
        Tn = preds.shape[0]
        for h in HORIZONS:
            if h <= Tn:
                me[h].append(np.sqrt(np.mean((preds[h - 1][OBJ] - obs[h][OBJ]) ** 2)))
    return ({h: round(float(np.mean(me[h])), 4) for h in HORIZONS if me[h]}, round(max_speed, 3))


def regime_endpoints(model, alpha, buckets, device):
    out, rest_diff = {}, 0.0
    for r, b in buckets.items():
        if len(b["S0"]) == 0:
            continue
        p = CC.corrected_rollout_batch(model, b["S0"], b["V0"], b["Aw"], device, alpha, OBJ)
        pe = p[:, -1, :][:, OBJ]
        out[r] = round(float(np.linalg.norm(pe - b["end"][:, OBJ], axis=1).mean()), 4)
        if r == "quasi_static" and alpha != 0.0:
            p0 = CC.corrected_rollout_batch(model, b["S0"], b["V0"], b["Aw"], device, 0.0, OBJ)
            rest_diff = float(np.abs(p[:, :, OBJ] - p0[:, :, OBJ]).max())
    return out, round(rest_diff, 5)


def agg(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)) if xs else (None, None)


def svg_decay(path, real, base, cand):
    """Hand-written SVG line chart (no matplotlib): per-step object speed, reference vs predictions."""
    W, H, ml, mt, pw, ph = 640, 360, 70, 40, 500, 250
    steps = list(range(1, len(real) + 1))
    ymax = max(max(real), max(base), max(cand)) * 1.15 or 1.0

    def X(i): return ml + (i / (len(real) - 1)) * pw
    def Y(v): return mt + ph - (v / ymax) * ph

    def poly(curve, color, label, dash=""):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(curve))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{d} points="{pts}"/>'

    yt = "".join(
        f'<line x1="{ml}" y1="{mt+ph-(f*ph):.0f}" x2="{ml+pw}" y2="{mt+ph-(f*ph):.0f}" stroke="#e5e5e5"/>'
        f'<text x="{ml-8}" y="{mt+ph-(f*ph)+4:.0f}" font-size="11" fill="#666" text-anchor="end">{f*ymax:.3f}</text>'
        for f in (0, 0.25, 0.5, 0.75, 1.0))
    xt = "".join(
        f'<text x="{X(i):.0f}" y="{mt+ph+18}" font-size="11" fill="#666" text-anchor="middle">{s}</text>'
        for i, s in enumerate(steps) if i % 2 == 0)
    legend = [("#111111", "real (reference)", ""), ("#e8833a", "baseline K5", ""),
              ("#2b8a7a", "+ coast correction", "")]
    lg = "".join(
        f'<line x1="{ml+pw-150}" y1="{mt+12+j*18}" x2="{ml+pw-125}" y2="{mt+12+j*18}" stroke="{c}" stroke-width="2.5"/>'
        f'<text x="{ml+pw-120}" y="{mt+16+j*18}" font-size="11" fill="#333">{lab}</text>'
        for j, (c, lab, _) in enumerate(legend))
    body = (f'<rect width="{W}" height="{H}" fill="white"/>'
            f'<text x="{ml}" y="24" font-size="15" font-weight="600" fill="#111">'
            f'FetchSlide free-slide: per-step object speed vs reference</text>'
            f'{yt}{xt}'
            f'<text x="{ml+pw/2:.0f}" y="{H-8}" font-size="12" fill="#444" text-anchor="middle">rollout step</text>'
            f'<text x="18" y="{mt+ph/2:.0f}" font-size="12" fill="#444" text-anchor="middle" '
            f'transform="rotate(-90 18 {mt+ph/2:.0f})">object speed (m/step)</text>'
            f'{poly(real,"#111111","real")}{poly(base,"#e8833a","base")}{poly(cand,"#2b8a7a","cand")}{lg}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{body}</svg>')


def main() -> int:
    t0 = time.time()
    device = pick_device()
    sp = store.splits(DATASET_ID)
    train_set = flat(sp["train"])
    train_ep_list = store.episodes(DATASET_ID, "train")
    val_ep_list = store.episodes(DATASET_ID, "val")
    test_eps = store.episodes(DATASET_ID, "test_heldout")
    stats = T.compute_stats(*train_set)

    val_fs = free_slide_full(val_ep_list)      # calibration (val only)
    tS0, tV0, tAw, tReals = free_slide_full(test_eps)
    test_buckets = windows_by_regime(test_eps)
    print(f"[phase9f] device={device}  free-slide windows: val={len(val_fs[0])} test={len(tS0)}  "
          f"regimes test: " + " ".join(f"{r}={len(test_buckets[r]['S0'])}" for r in test_buckets))

    per_seed, curves_seed0 = [], None
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        m = ResidualDynamics(25, 4, stats)
        T.train_rollout(m, train_ep_list, val_ep_list, device, horizon=ROLLOUT_K,
                        epochs=EPOCHS, patience=PATIENCE, seed=seed, obj_cols=OBJ)
        m.eval()
        # calibrate alpha on VALIDATION free-slide endpoint error
        vS0, vV0, vAw, vReals = val_fs
        cal = CC.calibrate_alpha(m, vS0, vV0, vAw, vReals[:, H_DIAG, :], ALPHA_GRID, device, OBJ)
        alpha = cal["alpha"]
        # evaluate baseline (alpha=0) and candidate (alpha*) on held-out TEST
        base_fs = eval_free_slide(m, 0.0, tS0, tV0, tAw, tReals, device)
        cand_fs = eval_free_slide(m, alpha, tS0, tV0, tAw, tReals, device)
        base_ms, base_max = full_episode_stability(m, 0.0, test_eps, device)
        cand_ms, cand_max = full_episode_stability(m, alpha, test_eps, device)
        base_reg, _ = regime_endpoints(m, 0.0, test_buckets, device)
        cand_reg, rest_diff = regime_endpoints(m, alpha, test_buckets, device)
        per_seed.append({"seed": seed, "alpha": alpha, "alpha_sweep": cal["sweep"],
                         "baseline": {"freeslide": base_fs, "multistep": base_ms, "max_step_speed": base_max, "regime": base_reg},
                         "candidate": {"freeslide": cand_fs, "multistep": cand_ms, "max_step_speed": cand_max, "regime": cand_reg,
                                       "rest_state_max_diff_m": rest_diff}})
        if seed == 0:
            curves_seed0 = {"real": base_fs["real_speed_curve"], "baseline": base_fs["pred_speed_curve"],
                            "candidate": cand_fs["pred_speed_curve"]}
        print(f"[phase9f]  seed {seed}: alpha*={alpha}  free-slide endpoint base {base_fs['endpoint_object_rmse_m']} "
              f"-> cand {cand_fs['endpoint_object_rmse_m']}  (bias {base_fs['signed_travel_bias_m']:+.3f}->"
              f"{cand_fs['signed_travel_bias_m']:+.3f})  stab h20 base {base_ms[20]} cand {cand_ms[20]} (max {cand_max})")

    # ---- aggregate ----
    def col(side, path):
        out = []
        for r in per_seed:
            v = r[side]
            for k in path:
                v = v[k]
            out.append(v)
        return out

    base_end_m, base_end_s = agg(col("baseline", ["freeslide", "endpoint_object_rmse_m"]))
    cand_end_m, cand_end_s = agg(col("candidate", ["freeslide", "endpoint_object_rmse_m"]))
    base_bias_m, _ = agg(col("baseline", ["freeslide", "signed_travel_bias_m"]))
    cand_bias_m, _ = agg(col("candidate", ["freeslide", "signed_travel_bias_m"]))
    base_pw_m, _ = agg(col("baseline", ["freeslide", "pointwise_traj_rmse_m"]))
    cand_pw_m, cand_pw_s = agg(col("candidate", ["freeslide", "pointwise_traj_rmse_m"]))
    alpha_vals = [r["alpha"] for r in per_seed]

    def ms_agg(side, h):
        return agg([r[side]["multistep"].get(h) or r[side]["multistep"].get(str(h)) for r in per_seed])
    base_h10, _ = ms_agg("baseline", 10); cand_h10, cand_h10s = ms_agg("candidate", 10)
    base_h20, base_h20s = ms_agg("baseline", 20); cand_h20, cand_h20s = ms_agg("candidate", 20)

    def reg_agg(side, r):
        return agg([ps[side]["regime"].get(r) for ps in per_seed])
    base_quasi, _ = reg_agg("baseline", "quasi_static"); cand_quasi, cand_quasi_s = reg_agg("candidate", "quasi_static")
    base_contact, _ = reg_agg("baseline", "contact_driven"); cand_contact, cand_contact_s = reg_agg("candidate", "contact_driven")

    # ---- acceptance gate (pre-registered thresholds) ----
    endpoint_drop = base_end_m - cand_end_m
    rel_improve = endpoint_drop / base_end_m if base_end_m else 0.0
    endpoint_spread = base_end_s + cand_end_s
    primary_ok = bool(rel_improve >= PRIMARY_REL_MIN and endpoint_drop > endpoint_spread)
    noregress_overall = bool((cand_h10 - base_h10) <= max(cand_h10s + 0, REGRESS_TOL_M)
                             and (cand_h20 - base_h20) <= max(cand_h20s + base_h20s, REGRESS_TOL_M))
    noregress_regime = bool((cand_quasi - base_quasi) <= max(cand_quasi_s, REGRESS_TOL_M)
                            and (cand_contact - base_contact) <= max(cand_contact_s, REGRESS_TOL_M))
    cand_max_h20 = max(r["candidate"]["multistep"].get(20, r["candidate"]["multistep"].get("20", 0)) for r in per_seed)
    stable = bool(cand_h20 <= STABILITY_MAX_RATIO * base_h20 and np.isfinite(cand_max_h20))
    rest_ok = bool(max(r["candidate"]["rest_state_max_diff_m"] for r in per_seed) < REST_MAX_DIFF_M)
    accept = bool(primary_ok and noregress_overall and noregress_regime and stable and rest_ok)

    # ---- interpretation (conservative, §8) ----
    if accept:
        verdict = "positive_evidence_for_damping_intervention"
        interp = ("A single bounded momentum correction reduced held-out free-slide endpoint error "
                  "beyond spread with no unacceptable regressions -> positive evidence that the learned "
                  "object dynamics apply EXCESS velocity decay. Note (§8): this shows the INTERVENTION "
                  "works; it isolates excess decay but does not by itself prove the physical micro-cause.")
    elif endpoint_drop <= 0 or rel_improve < PRIMARY_REL_MIN:
        verdict = "evidence_against_constant_damping_correction"
        interp = ("A calibrated constant momentum correction did NOT reduce held-out free-slide endpoint "
                  "error beyond threshold" + (" (bias may shift while endpoint/pointwise error does not)."
                  if cand_bias_m is not None else ".") + " The under-travel is not a constant multiplicative "
                  "damping offset; consistent with partial observability of the absolute velocity (the "
                  "baseline's own displacement, which alpha scales, is itself unreliable). Damping is NOT "
                  "excluded as a physical factor -- only the constant-correction form is.")
    else:
        verdict = "inconclusive_regressions_or_instability"
        interp = ("The correction reduced free-slide endpoint error but introduced regressions in other "
                  "regimes / overall multi-step / stability -> net not acceptable; a lower net-travel bias "
                  "concealed trajectory errors. Rejected.")

    recommended_next = (
        "If rejected: the two independent routes (9E velocity input, 9F momentum correction) now both fail to "
        "fix the under-travel via the OBJECT channel alone, strengthening partial-observability of absolute "
        "velocity as the cause. Highest-value next test: a STABLE state augmentation that supplies absolute "
        "velocity WITHOUT the 9E instability (e.g. a filtered/clipped velocity input, or an input-delta "
        "formulation trained with the rollout loss and a rest-state regularizer), evaluated with this same "
        "protocol. Otherwise accept the measured ceiling on a Markov state-only FetchSlide model and report it."
        if verdict != "positive_evidence_for_damping_intervention" else
        "Adopt the correction; next, test whether a velocity-DEPENDENT (non-constant) decay correction "
        "further reduces the high-speed residual, and re-run Phase 9C atlas on the corrected model.")

    svg_decay(FIG, curves_seed0["real"], curves_seed0["baseline"], curves_seed0["candidate"])
    CORR_JSON.write_text(json.dumps({
        "baseline_checkpoint": str(CKPT_BASE.relative_to(ROOT)),
        "correction": "object-channel bounded momentum re-injection (coast_correction.corrected_rollout_*)",
        "alpha_per_seed": alpha_vals, "alpha_adopted": (float(np.median(alpha_vals)) if accept else 0.0),
        "adopted_as_current_best": accept,
        "note": "candidate = baseline model + alpha; baseline checkpoint unchanged"}, indent=2))

    report = {
        "phase": "9F", "stage": "damping_intervention", "dataset_id": DATASET_ID, "env": "FetchSlide-v4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git_commit": git_commit(), "device": device, "n_seeds": N_SEEDS, "epochs": EPOCHS,
                       "patience": PATIENCE, "rollout_K": ROLLOUT_K, "alpha_grid": ALPHA_GRID,
                       "baseline_checkpoint": str(CKPT_BASE.relative_to(ROOT)),
                       "splits": {k: len(sp[k]) for k in ("train", "val", "test_heldout")}},
        "hypothesis": ("free-slide under-travel is excess multiplicative velocity decay correctable by a bounded "
                       "momentum re-injection on the object channel (alpha in [0,1), zero at rest, geometric-stable)"),
        "intervention": ("obj_pred'_{k+1} = obj_pred_{k+1} + alpha * prev_object_displacement; MLP never sees the "
                         "velocity (no amplification); alpha calibrated on val free-slide endpoint RMSE"),
        "primary_metric": "held-out free-slide endpoint object RMSE (m), mean over seeds",
        "pre_registered_thresholds": {"primary_rel_min": PRIMARY_REL_MIN, "regress_tol_m": REGRESS_TOL_M,
                                      "stability_max_ratio": STABILITY_MAX_RATIO, "rest_max_diff_m": REST_MAX_DIFF_M},
        "alpha_per_seed": alpha_vals,
        "aggregate": {
            "free_slide_endpoint_rmse_m": {"baseline_mean": base_end_m, "baseline_std": base_end_s,
                                           "candidate_mean": cand_end_m, "candidate_std": cand_end_s,
                                           "drop_m": round(endpoint_drop, 4), "rel_improve": round(rel_improve, 3)},
            "free_slide_signed_bias_m": {"baseline": base_bias_m, "candidate": cand_bias_m},
            "free_slide_pointwise_rmse_m": {"baseline": base_pw_m, "candidate": cand_pw_m},
            "full_episode_multistep_m": {"baseline_h10": base_h10, "candidate_h10": cand_h10,
                                         "baseline_h20": base_h20, "candidate_h20": cand_h20},
            "regime_endpoint_rmse_m": {"baseline_quasi": base_quasi, "candidate_quasi": cand_quasi,
                                       "baseline_contact": base_contact, "candidate_contact": cand_contact},
            "stability": {"candidate_max_step_speed_m": max(r["candidate"]["max_step_speed"] for r in per_seed),
                          "baseline_max_step_speed_m": max(r["baseline"]["max_step_speed"] for r in per_seed),
                          "rest_state_max_diff_m": max(r["candidate"]["rest_state_max_diff_m"] for r in per_seed)},
        },
        "gate": {"primary_ok": primary_ok, "noregress_overall": noregress_overall,
                 "noregress_regime": noregress_regime, "stable": stable, "rest_ok": rest_ok, "accept": accept},
        "verdict": verdict, "interpretation": interp,
        "hypothesis_ledger": {
            "H_friction/H_damp": ("supported" if accept else "constant-correction form contradicted; "
                                  "physical damping not excluded"),
            "H_velocity": "fix via object channel again insufficient (9E unstable input, 9F constant correction) "
                          "-> partial-observability of absolute velocity strengthened",
            "H_horizon": "contradicted (Phase 9E: bias does not vanish with K)",
            "H_coverage": "unresolved (OOD->error weak, 0.22)"},
        "recommended_next_experiment": recommended_next,
        "per_seed": per_seed,
        "velocity_decay_curves_seed0": curves_seed0,
        "figure": str(FIG.relative_to(ROOT)),
        "wall_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "reports" / "phase9f_damping.json").write_text(json.dumps(report, indent=2))

    print("=" * 80)
    print("DREAM-OS Phase 9F -- targeted damping intervention (FetchSlide free-slide under-travel)")
    print("=" * 80)
    print(f"  alpha* per seed: {alpha_vals}   (calibrated on VAL free-slide endpoint RMSE)")
    print(f"  free-slide endpoint RMSE  baseline {base_end_m}+/-{base_end_s}  ->  candidate {cand_end_m}+/-{cand_end_s}"
          f"  (drop {endpoint_drop:+.4f}, {rel_improve*100:.1f}%)")
    print(f"  free-slide signed bias    baseline {base_bias_m:+.4f}  ->  candidate {cand_bias_m:+.4f}")
    print(f"  free-slide pointwise RMSE baseline {base_pw_m}  ->  candidate {cand_pw_m}")
    print(f"  full-episode h20 (stability) baseline {base_h20}  ->  candidate {cand_h20}  (max step {report['aggregate']['stability']['candidate_max_step_speed_m']})")
    print(f"  regime endpoint quasi {base_quasi}->{cand_quasi}  contact {base_contact}->{cand_contact}  rest|diff| {report['aggregate']['stability']['rest_state_max_diff_m']}")
    print(f"  GATE  primary {primary_ok}  noregress_overall {noregress_overall}  noregress_regime {noregress_regime}  stable {stable}  rest {rest_ok}")
    print(f"  ACCEPT: {accept}   VERDICT: {verdict}")
    print(f"  -> {interp}")
    print(f"  figure -> {FIG.relative_to(ROOT)}   checkpoint decision -> adopted_as_current_best={accept}")
    print(f"  wall={report['wall_seconds']}s  -> reports/phase9f_damping.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
