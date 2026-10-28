# DREAM-OS Phase 9C — Predictive Failure Atlas (deliverable §15.4)

Benchmark: **FetchSlide-v4** · subject: adopted **rollout K=5** model vs one-step baseline ·
diagnostic horizon **H=10** · 1476 held-out windows · source: `reports/phase9c_failure_atlas.json`
(git `9ad92c5`). Every number below traces to those windows; nothing is asserted without a count.

## 1. Where the model fails — by observable motion regime

Object-position error at H=10 (metres), rollout K=5 vs one-step baseline:

| regime | windows | rollout err | baseline err | rollout gain |
|---|---|---|---|---|
| quasi-static (puck barely moves) | 665 (45%) | **0.032** | 0.054 | −41% |
| contact-driven (moves while gripper near) | 453 (31%) | **0.061** | 0.106 | −42% |
| **free-sliding (coasts, gripper far)** | 358 (24%) | **0.094** | 0.117 | **−20%** |

Rollout training (9B) helps everywhere, but **helps the free-sliding regime least** (−20% vs −41%).
Free-sliding is the worst regime at 0.094 m — this is the residual failure Phase 9D targets.

## 2. Contact × travel failure map (rollout K=5)

| | travel low (<0.02) | travel mid | travel high (>0.08) |
|---|---|---|---|
| gripper **near** (<0.05) | 0.043 (n=33) | 0.104 (n=39) | 0.080 (n=296) |
| gripper **mid** | 0.041 (n=236) | 0.047 (n=61) | 0.065 (n=154) |
| gripper **far** (>0.10) | 0.029 (n=451) | 0.025 (n=66) | **0.116 (n=140)** |

The single worst cell is **gripper-far + high-travel** (0.116 m, n=140): the puck has left the gripper and
is coasting a long way. Error also rises monotonically with peak puck speed — slow 0.034 → fast **0.092**
(n=406) — and *falls* with action magnitude (low 0.066 → high 0.039), because large actions mean the gripper
is actively pushing (in-distribution contact control) while small actions leave the puck to coast (the hard case).

## 3. The systematic discrepancy — direction (feeds Phase 9D)

On the 358 free-sliding windows, measured against the real simulator:

- real puck travel over 10 steps: **0.265 m**
- rollout model predicts: **0.208 m** → signed bias **−0.057 m** (~21% short)
- one-step baseline predicts: 0.194 m → bias −0.071 m

**The model stops the sliding puck too early — it over-estimates friction / decelerates the puck too
aggressively.** (I measured the sign rather than assuming it; my prior guess that it would over-shoot was
wrong.) This is a *recurring, directional* error, the right kind of target for competing physical hypotheses.

## 4. Error vs uncertainty vs out-of-distribution (§6: three distinct concepts)

On the adopted rollout ensemble (5 members):

| signal | Spearman vs realized error | hi-tercile err | lo-tercile err |
|---|---|---|---|
| ensemble disagreement (**uncertainty**) | **0.662** | 0.081 | 0.018 (4.6×) |
| nearest-train distance (**OOD / coverage**) | 0.215 | 0.071 | 0.034 (2.1×) |

**Uncertainty is a good failure predictor; OOD is weak.** The model's disagreement reliably flags its own
free-sliding errors (the regime is high-uncertainty and high-error together), but those windows are *not*
strongly out-of-distribution. So the failure is **not primarily a training-coverage gap** — it points at the
model's representation or learned dynamics, not merely "unseen states." This is already evidence that
discriminates among the Phase 9D hypotheses before any experiment is run.

## 5. Atlas summary

- **Succeeds**: quasi-static and in-contact pushing (0.03–0.06 m); low-speed, actively-gripped motion.
- **Fails**: free-sliding / fast-coasting puck after the gripper leaves (0.09–0.12 m), under-predicting travel by ~21%.
- **Knows it fails**: ensemble uncertainty ranks the error at ρ=0.66 (flags the free-slide regime).
- **Not a coverage problem**: OOD ranks error only ρ=0.22; free-slide windows are in-distribution yet wrong.

→ Phase 9D operationalizes competing explanations for the free-sliding under-travel bias (learned friction
vs object-velocity representation vs training coverage vs objective horizon) with testable, discriminating predictions.
