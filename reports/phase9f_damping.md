# DREAM-OS Phase 9F — Targeted Damping Intervention (deliverable §9)

Benchmark **FetchSlide-v4** · baseline = adopted **rollout-K5** (`fetchslide_rollout_k5.pt`, unchanged) ·
3 seeds · splits 168 train / 36 val / 36 held-out test · source `reports/phase9f_damping.json`
(git `9ad92c5`, GPU, 96 s). Every number traces to held-out windows; nothing is asserted without a count.

> **Outcome: evidence AGAINST the constant damping correction — rejected, baseline preserved.**
> The optimal momentum coefficient is **α\* = 0.0 on all three seeds**: any positive correction strictly
> worsens held-out free-slide trajectory error. This is a third independent failure to fix the under-travel
> through the object channel, and it sharpens the diagnosis.

## 1. Hypothesis and motivation

Phases 9C–9E showed the model under-predicts how far a free-sliding puck coasts (−0.052 m, ~21% short) and
that this survives both a longer training horizon and an absolute-velocity input (which was also unstable).
Damping was the leading explanation *by elimination*, never positively demonstrated.

Because the model is a learned residual MLP (`s_{t+1}=s_t+f(s_t,a_t)`) with **no explicit damping parameter**,
"damping" can only be an emergent property of the learned delta. **Falsifiable H_damp:** the under-travel is an
*excess, roughly multiplicative velocity decay* that a bounded momentum re-injection can correct. If true, a
single calibrated α reduces held-out free-slide **endpoint** error toward zero; if false (partial observability
of absolute velocity), no constant α can, because it only scales the baseline's own unreliable displacement.

## 2. The exact intervention (single factor, stable by construction)

On the object channel only, during open-loop rollout:

```
obj_pred'_{k+1} = obj_pred_{k+1} + α · (previous object displacement)
```

α∈[0,1); the previous displacement is the real finite difference at a window start, **0 at an episode start**
(puck at rest). The MLP never sees the velocity, so it cannot amplify it; ignoring the MLP, δ_k = α·δ_{k−1}
→ geometric decay, bounded for α<1, and **exactly the baseline at rest or at α=0**. This is a *different,
stable* form from the absolute-velocity input Phase 9E rejected (unit-tested in `tests/test_coast_correction.py`:
α=0≡baseline, rest-safe, α=1 recovers a constant coast exactly, α<1 bounded). **α is calibrated on the
validation set**, then evaluated once on held-out test.

## 3. Baseline and evaluation protocol

For each of 3 seeds: train rollout-K5 (identical config to the adopted model), calibrate α over
{0.0,…,0.9} by **validation free-slide endpoint RMSE** (458 val free-slide windows), then on the 354 held-out
test free-slide windows measure baseline (α=0) vs candidate (α\*): endpoint RMSE, signed & absolute travel
bias, pointwise trajectory RMSE, per-horizon error, predicted-vs-reference speed-decay curve, by initial-velocity
and displacement group. Plus full-episode rest-start stability (h1…h20 + max step speed) and per-regime endpoint
RMSE (quasi 651 / contact 435 / free 354 test windows). Train/val/test kept separate throughout.

## 4. Per-seed quantitative results — the calibration *is* the result

**Validation free-slide endpoint RMSE vs α (monotonically increasing → optimum at α=0):**

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| seed 0 | **0.073** | 0.077 | 0.089 | 0.114 | 0.151 | 0.205 | 0.393 | 0.819 |
| seed 1 | **0.069** | 0.074 | 0.089 | 0.115 | 0.154 | 0.209 | 0.398 | 0.809 |
| seed 2 | **0.062** | 0.068 | 0.086 | 0.115 | 0.156 | 0.214 | 0.413 | 0.849 |

→ **α\* = 0.0 for every seed.** Any momentum re-injection makes the held-out trajectory *worse*, smoothly and
reproducibly. The candidate therefore equals the baseline, so all held-out metrics are identical:

| metric (held-out, mean over seeds) | baseline | candidate |
|---|---|---|
| free-slide endpoint RMSE (m) | 0.0936 ± 0.0041 | 0.0936 ± 0.0041 |
| free-slide signed travel bias (m) | −0.0522 | −0.0522 |
| free-slide pointwise traj RMSE (m) | 0.0496 | 0.0496 |
| full-episode h10 / h20 (m) | 0.0314 / 0.0653 | 0.0314 / 0.0653 |
| regime endpoint quasi / contact (m) | 0.0324 / 0.060 | 0.0324 / 0.060 |

Error grows with the coast, as expected: endpoint RMSE is 0.070 (low-displacement) → **0.118** (high), and
0.084 (low initial speed) → **0.104** (high) — the faster/farther the puck, the worse, which is exactly where a
*correct* momentum model would help and this constant one cannot.

## 5. Stability and rest-state results

Because α\*=0 the candidate is identical to the baseline; trivially stable. More importantly the **calibration
landscape itself is stable and clean** — monotone in α, no noise, so "no correction helps" is robust, not a
near-miss. Rest-state check: on quasi-static (at-rest) windows the correction's max deviation from baseline is
**0.0 m** by construction (δ≈0). Full-episode max step speed 0.126 m (bounded; no divergence) — in deliberate
contrast to Phase 9E's unstable velocity input (h20 diverged to 3.9 m).

## 6. Prediction-vs-reference visualisation

`reports/figures/phase9f_velocity_decay.svg` — per-step object speed. The **reference** puck accelerates to a
peak of **0.033 m/step** (steps 5–6) then coasts down; the **model** stays flat at ~0.024 and **never reaches
the peak**. The deficit is in *peak speed*, not in late-stage over-deceleration.

`reports/figures/phase9f_alpha_sweep.svg` — the three-seed α-sweep, optimum marked at α=0.

**This is the mechanism:** the model does not over-damp a fast puck — it never predicts the puck getting fast in
the first place. That is the signature of conditional-mean regression under partial observability (the state has
no absolute object velocity), and a forward momentum nudge along the model's flat, partly-mis-directed trajectory
can only overshoot the endpoint, which is exactly what the α-sweep shows.

## 7. Acceptance / rejection decision

Pre-registered thresholds (set before evaluation): primary = ≥5% held-out free-slide endpoint reduction beyond
seed spread; no regime/overall regression beyond 0.002 m; stability ≤1.5× baseline h20; rest deviation <1 mm.

| gate | primary | no-regress overall | no-regress regime | stable | rest | **accept** |
|---|---|---|---|---|---|---|
| result | **FAIL** (0% improvement) | pass | pass | pass | pass | **REJECT** |

**Candidate rejected. Baseline `fetchslide_rollout_k5.pt` preserved as current best**
(`fetchslide_coast_correction.json`: `adopted_as_current_best: false`).

## 8. Updated hypothesis ledger (conservative, §8)

| hypothesis | status after 9F |
|---|---|
| **H_damp / H_friction** (excess velocity decay) | **constant-correction form contradicted** — no α>0 reduces held-out endpoint error. Physical damping is *not excluded*; a *constant multiplicative* correction is. A failed correction does not prove damping irrelevant. |
| **H_velocity** (partial observability of absolute velocity) | **strengthened.** Two independent object-channel interventions — 9E's velocity *input* and 9F's momentum *correction* — both fail. The model can't recover absolute coast speed from a Markov state where velocity is only gripper-relative; it regresses to the conditional-mean (slower) displacement. The 9E *fix* was unstable, but the *diagnosis* now has convergent support. |
| **H_horizon** | contradicted (9E: bias does not vanish with K). |
| **H_coverage** | unresolved (OOD→error weak, ρ=0.22, weakly against). |

Note the distinction the brief demands: this is evidence about the *particular intervention*, not proof about the
broad physical cause. The intervention tells us the under-travel is **not a constant damping offset**; it does
not by itself prove the micro-physics of the real contact.

## 9. Known limitations

- α is a single global constant; a velocity-*dependent* decay was not tested (deliberately — §8: don't stack
  untested corrections before interpreting the simplest one).
- The correction acts only on object position; it does not touch the velocity channels or retrain the dynamics.
- Held-out test is 36 episodes / 354 free-slide windows from one FetchSlide configuration; generalization across
  object/goal configurations is Phase 9H, not yet run.
- "Partial observability" is inferred convergently (two failed fixes + the peak-speed deficit curve), not proven
  by a positive construction; the decisive positive test is a *stable* absolute-velocity representation.

## 10. Recommended next experiment

Two independent object-channel fixes have now failed, and the speed-decay curve localizes the deficit to **peak
coast speed the Markov state cannot encode**. Highest-value next test: a **stable absolute-velocity
representation** — a filtered/clipped velocity input or an input-delta formulation trained with the rollout loss
*and a rest-state regularizer* to prevent the 9E divergence — evaluated with this exact protocol. If that also
fails to reduce held-out free-slide endpoint error, the honest conclusion is a **measured ceiling for a
state-only Markov world model on FetchSlide coasting**, which DREAM-OS should report as a well-supported negative
rather than keep correcting.
