# DREAM-OS Phase 9E — Decisive Experiments (deliverable §15.6/§15.7)

Benchmark **FetchSlide-v4** · failure under study: free-sliding puck **under-travel** (Phase 9C/9D) ·
354 held-out free-slide windows · 3 seeds · source `reports/phase9e_decisive.json` (git `9ad92c5`, GPU, 328 s).
Two experiments, each changing **one factor**, separate the two live contenders left by Phase 9D
(**H_velocity** vs **H_horizon**). Every number traces to the held-out set; nothing is asserted without a count.

## Gate verdicts

| gate | question | result |
|---|---|---|
| **Gate D** | did the experiments discriminate? | **PASS** — a single hypothesis selected, the two leading alternatives positively contradicted |
| **Gate E** | measured learning benefit (separate from discrimination)? | **FAIL (honest)** — the fix did not improve held-out prediction → **roll back to `rollout_K5`** |

The adopted model is **unchanged** (`fetchslide_rollout_k5.pt`). The augmented checkpoint is saved for the
record but flagged `adopted_as_current_best: False`.

## E_abs_velocity_feature — add an absolute object-velocity input (H_velocity's proposed fix)

Single factor: the state is augmented with a **finite-difference absolute object velocity**, recomputed from
the model's own predictions each open-loop step (loop stays closed). Same architecture, horizon (K=5), data, seeds.
A synthetic mechanism test (`tests/test_aug_velocity.py`) first confirmed the augmentation *can* recover coasting a
position-only model cannot (h10 error 5.3× lower) — so the machinery is trustworthy.

On the free-slide windows (where both models get a real `v0` and are stable):

| model | travel bias (m) | endpoint object RMSE (m) |
|---|---|---|
| baseline rollout-K5 | **−0.0522 ± 0.0037** | **0.0936 ± 0.0041** |
| **+ absolute-velocity input** | −0.0487 ± 0.0063 | 0.0922 ± 0.0099 |
| change | |bias| −0.0035 (spread 0.0100) → **not beyond spread** | −0.0014 (spread 0.0140) → **not beyond spread** |

**The velocity input did not reduce the under-travel.** Both the bias and the endpoint error are statistically
unchanged (within seed spread). H_velocity predicted a *substantial* shrink; it did not happen.

**Plus an unanticipated, measured failure:** the velocity-feedback rollout is **numerically unstable from rest**.
On full-episode rollout (start at the puck's resting state, `v0 = 0`) the augmented model **diverges** —

| horizon | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| baseline-K5 object RMSE (m) | 0.004 | 0.016 | 0.031 | 0.047 | **0.065** |
| **augmented object RMSE (m)** | 1.97 | 2.34 | 2.99 | 3.60 | **3.88** |

A single step from rest already predicts a ~1.5 m object jump (real ≈ 0); the fed-back velocity feature then
amplifies. So this *particular* remedy is **contradicted and unstable**. (A different, stable velocity
parameterization — filtered, clipped, or not fed back — was not tested, so H_velocity as a *hypothesis* is not
killed, only its one concrete fix.)

## E_horizon_sweep — free-slide bias vs training horizon K (H_horizon's prediction)

H_horizon predicted the under-travel bias → 0 as K grows. Measured (seed 0):

| K | 1 | 3 | 5 | 10 |
|---|---|---|---|---|
| signed travel bias (m) | **−0.017** | −0.033 | −0.057 | −0.053 |
| endpoint object RMSE (m) | 0.120 | 0.097 | 0.094 | **0.092** |

**H_horizon is contradicted.** The travel bias does **not** fall toward zero as K grows — its magnitude *grows*
(−0.017 → −0.057) then plateaus. Larger horizons do *not* remove the under-travel.

A methodological note worth recording: bias magnitude and endpoint error move in **opposite** directions across K
(K=1 has the smallest |bias| but the *worst* endpoint error; K=10 the largest |bias| but the *best* endpoint).
Net travel-magnitude bias is therefore a **confounded** proxy for prediction quality — the K=1 model is less wrong
in net distance but more wrong pointwise. We report both and trust the endpoint error for quality.

## Hypothesis status (demonstrated vs negative vs unresolved)

| hypothesis | status | basis |
|---|---|---|
| **H_velocity** | **fix contradicted** (negative) | the absolute-velocity input left bias & endpoint unchanged *and* destabilized long-horizon rollout |
| **H_horizon** | **contradicted** (negative) | free-slide \|bias\| does not fall to zero as K grows; it grows then plateaus |
| **H_friction** | **most consistent by elimination — NOT positively demonstrated** | the under-travel survives both interventions; |bias| floors at −0.017 m |
| **H_coverage** | **unresolved** | `E_add_freeslide_data` not run; standing OOD→error (ρ=0.22) weakly argues against |

**Headline (honest):** the free-sliding under-travel is a **persistent, robust** property of the learned
dynamics — it is removed by *neither* a longer training horizon *nor* an added absolute-velocity input. Of the
registered hypotheses it is **most consistent with learned over-damping (H_friction)**, but that is reached by
*elimination*, not by a positive test. Phase 9E's verification layer did its job: it refused to adopt a change
that looked plausible but did not improve held-out prediction, and it surfaced an instability the headline metric
alone would have hidden.

## Recommended next experiment (the positive test H_friction still lacks)

**E_damping_intervention** — directly probe learned over-damping: add a velocity-dependent decay / a targeted
free-slide term and measure the held-out free-slide travel. Unlike E_abs_velocity_feature this intervenes on the
*dynamics* rather than the *input*, so a reduction in under-travel would be positive evidence for H_friction rather
than elimination.
