# DREAM-OS Phase 9 — Baseline Audit (deliverable §15.1)

Date: 2026-10-28 · Benchmark: **FetchSlide-v4** · git commit at audit: `9ad92c5`

This audit distinguishes **verified existing functionality** (code that was run and
whose numbers reproduced) from **proposed / not-yet-built Phase 9 functionality**.
Per §2, no historical reports, checkpoints, datasets, or configs were overwritten.

---

## 1. Environment (verified, not assumed)

| Package | Reported earlier | Measured now |
|---|---|---|
| torch | 2.11.0+cu128 | **2.11.0+cu128**, CUDA available ✓ |
| mujoco | 3.3.6 | **3.3.6** ✓ |
| gymnasium | 1.4.0 | **1.4.0** ✓ |
| gymnasium-robotics | 1.4.2 | **1.4.2** ✓ |
| numpy / scipy | 2.2.6 / 1.15.3 | **2.2.6 / 1.15.3** ✓ |

Hardware: RTX 4050 (6 GB). The world model uses ~0.03 GB peak VRAM — far inside budget.

## 2. Verified existing functionality (ran this session, reproduced)

- **Experience collector** (`scripts/collect_data.py`) + **episodic memory**
  (`memory/episode_store.py`): per-episode NPZ + provenance manifests, scenario splits.
- **World model** (`world_model/dynamics.py`, `training.py`): `ResidualDynamics`,
  79,897 params, predicts normalized Δstate; stats as buffers; **one-step training loss**.
- **Ensemble** (`world_model/ensemble.py`): 5-member bootstrap; disagreement = uncertainty.
- **Dream generator** (`dreaming/counterfactuals.py`) & **selection** (`selection.py`,
  `acquisition.py`): counterfactual branches, uncertainty/coverage/composite/learned selectors.
- **Reference-simulator verification** (`verification/replay.py`, `metrics.py`):
  reset+replay reconstruction (reconstruction error reported 0.0 in Phase 3/8), predicted-vs-real
  classification with explicit tolerances.
- **Datasets**: `fetchslide_phase8` — 240 eps / 12,000 transitions, success 7.9%,
  splits train 168 / val 36 / test 36 (scenario-level, distinct seeds ⇒ held-out scenarios).

**Reproduction check (§2):** Phase 9A retrained the full-budget model and reproduced the
Phase 8 headroom numbers — one-step object RMSE **0.00395 ± 0.00008 m** (Phase 8: 0.00412 ± 0.0003)
and multistep h20 **0.1192 m** (Phase 8: 0.1217). Historical results stand.

**Historical results preserved (unchanged):**
- Phase 7 (FetchPush, saturated): no acquisition strategy beats base; honest negative. `reports/phase7_results.json`.
- Phase 8 Stage 1 (FetchSlide headroom gate): data halves 1-step error (0.0098→0.0041). PASSED. `reports/phase8_headroom.json`.
- Phase 8 Stage 2 (acquisition): learned ranker Spearman 0.89 / recall 0.79; every strategy beats base but none beats random beyond noise at K=24; verified-real (0.00538) beats raw-imagined (0.00638). `reports/phase8_acquire.json`.

## 3. Not yet built (Phase 9 scope — proposed, honestly flagged)

- **Unit/integration tests**: `tests/` is **empty**. §14 requires tests per new component — to be added with each Phase 9 module.
- **Multi-step training**: the model is trained on a one-step loss only; no rollout loss exists yet (Phase 9B).
- **Failure atlas module** (9C), **hypothesis schema** (9D), **experiment planner** (9E),
  **verified-experiment learning loop** on hypotheses (9G): none exist yet.
- Obs-layout note (verified from data): `object_velp` cols 14:17 are stored **relative to the gripper**
  (only the z-axis correlates with raw object displacement), so regime analysis uses the measured
  per-step object displacement, not those columns.

## 4. Phase 9A baseline findings → where the model succeeds and fails

Source: `reports/phase9a_baseline.json` (3 seeds, full 168-ep train, held-out test).

**Succeeds**
- One-step object prediction: **0.00395 m**, 3× better than persistence (0.01175 m).
- Generalization gap train→held-out: **+0.00019 m** — negligible overfitting.
- Static / far-from-contact regimes (one-step object err ~0.003 m).

**Fails (the Phase 9B/9C headroom)**
- **Multi-step open-loop error compounds 74×**: h1 0.0016 → h5 0.0217 → h10 0.0587 → h20 0.1192 m.
- Only ~25% better than persistence at h20 (0.1192 vs 0.1569) and **tied with persistence at h5** (0.0217 vs 0.0217). Long-horizon prediction is weak.
- Worst regimes (one-step object err): **fast-sliding** object 0.00802 vs 0.00295 near-static; **near-contact** (<5 cm gripper–object) 0.00794 vs 0.00396 far. Physically the hard cases.

**Uncertainty vs error vs OOD (kept separate, §6)**
- Ensemble disagreement ranks held-out error: Spearman **0.568**; high-uncertainty tercile error 0.00707 vs low 0.00237 (3×). Uncertainty is informative but imperfect.
- Nearest-train-neighbour distance (OOD) ranks error weaker: Spearman **0.388**. Distinct from uncertainty (their mutual Spearman is reported in the JSON).

## 5. Gate A — Baseline characterized: **PASSED**

Prediction error, its horizon dependence (74× growth, tied-with-persistence at h5), the
failure regimes (fast-sliding, near-contact), the generalization gap, and uncertainty
calibration are all measured from real held-out logs. This defines a concrete,
measurable target for Phase 9B: **reduce multi-step open-loop error without regressing
one-step accuracy or resource use.**
