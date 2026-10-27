# DREAM-OS — Stage 0 Audit & Verification Report

**Date:** 2026-10-27 · **Host:** `sandeep-pc` · **Operator:** automated audit (inline)

This report records what was actually inspected, installed, and verified. Machine-
readable results: [`stage0_results.json`](stage0_results.json). Nothing here is
assumed; every value came from a command that was run and its output inspected.

---

## 1. Hardware & OS audit

| Resource | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138 |
| CPU | AMD Ryzen 7 7435HS — 8 cores / 16 threads |
| RAM | 23 GiB total (~17 GiB free at audit) |
| GPU | NVIDIA GeForce RTX 4050 Laptop — **6.0 GB VRAM** (5.3 GB free) |
| GPU driver | 575.64.03 (supports CUDA ≤ 12.9) |
| System CUDA toolkit | nvcc 11.5 (unused; torch bundles its own runtime) |
| Disk | / had 55 GB free at start; 43 GB after install (venv ≈ 6.8 GB) |
| Python | system CPython 3.10.12 only (no conda/pyenv) |
| Env manager | `uv 0.12.21` (used) |
| Build tools | git 2.34, gcc 11.4, cmake 3.22 |

**Environment hazard:** the login shell injects ROS Humble into `PYTHONPATH`
(`/opt/ros/humble/...`). All project Python is run with `PYTHONPATH` unset and in an
isolated venv (no system-site), so ROS packages cannot leak in.

## 2. Pre-existing packages (user-site, not used by the project)

`torch 2.10.0+cu128`, `gymnasium 1.3.0`, `numpy 1.26.4`, `scipy 1.15.3`,
`stable_baselines3 2.9.0`. **`mujoco` was not installed anywhere.** Isaac Sim is
present on disk but is **not** used (Stage 0 is MuJoCo-first per project rules).

## 3. Decisions

- **Isolated env:** `uv venv` on system Python 3.10 at `~/dream-os/.venv`, no
  system-site-packages. Clean of ROS (`ros_leak_in_path: False` verified).
- **Primary benchmark:** `FetchPush-v4` (gymnasium-robotics) — genuine object
  pushing, goal-conditioned, explicit `is_success`, seedable init, structured state.
  **Secondary:** `Pusher-v5` (gymnasium MuJoCo) as a zero-extra-dep pipeline smoke.
- **Device:** GPU (CUDA) with verified CPU fallback.

## 4. Blockers found and resolved (this is what Stage 0 is for)

1. **Default torch is CUDA-13, driver too old.** `pip install torch` gave
   `2.14.1+cu130`; `cuda.is_available()` was **False** ("driver too old, found
   12090"). → Reinstalled `torch==2.11.0` from `https://download.pytorch.org/whl/cu128`.
   GPU now initialises (matmul on RTX 4050 OK).
2. **mujoco 3.17.0 breaks gymnasium-robotics 1.4.2.** `FetchPush-v4` failed to
   build: `set_joint_qpos` asserts `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`, and
   under mujoco 3.17 `mjtJoint` is a strict enum, so `np.int32(2) in (…)` is
   **False** while `np.int32(2) == mjJNT_SLIDE` is True (asymmetric). gymnasium-
   robotics 1.4.2 only pins `mujoco>=2.2.0` and is the latest release (no fix
   upstream). → Pinned `mujoco==3.3.6`; membership works, both envs build + step.
3. **Benchmark id correction.** The current id is `FetchPush-v4`, not `FetchPush-v3`
   (versioning moved on in gymnasium-robotics 1.4.2).

A transient issue was self-inflicted and fixed: hand-pruning CUDA-13 orphan
packages deleted a shared `libcudnn.so.9`; a full `--reinstall` of torch restored a
consistent cu128 set. Lesson recorded: don't hand-prune nvidia sub-packages.

## 5. Verified stack (pinned)

```
torch==2.11.0+cu128   mujoco==3.3.6   gymnasium==1.4.0
gymnasium-robotics==1.4.2   numpy==2.2.6   scipy==1.15.3   pyyaml==6.0.3
```

## 6. Smoke-test results (all PASS)

| Test | Result |
|---|---|
| torch | PASS — `2.11.0+cu128`, device=cuda, VRAM 5.47/6.04 GB |
| mujoco | PASS — `3.3.6`, free body fell ~0.2 m over 100 steps |
| Pusher-v5 | PASS — obs_dim 23, qpos 11, reproducible=True |
| FetchPush-v4 | PASS — obs_dim 25, qpos 22, is_success=False (random actions), reproducible=True |

Run: `wall≈0.5 s`, `peak_ram≈0.8 GB` (smoke only).

## 7. Stage 0 acceptance criterion — MET

> Activate the env, load and run a real prebuilt MuJoCo benchmark, execute an
> episode, inspect the resulting state and task outcome, and reproduce it.

Demonstrated: `FetchPush-v4` episode executed; structured state retrieved
(`qpos_dim=22`); task outcome read (`is_success=False`); fixed seed + fixed action
sequence reproduces the trajectory exactly. Reproduce with
`python scripts/stage0_report.py` (or `./setup.sh` from scratch).

## 8. Known issues / risks carried forward

- **6 GB VRAM** is small: keep world models tiny; measure VRAM per Phase 13 rules.
- **Bleeding-edge pins:** gymnasium 1.4.0 / gymnasium-robotics 1.4.2 are the newest
  available and newly constrained mujoco — revisit if either updates.
- **No git commit yet:** `stage0_results.json` records `git_commit: uninitialised`.
  An initial commit is recommended to activate commit-based provenance (not done
  automatically; operator manages git).
- Random-action FetchPush does not reach the goal (expected); a scripted/baseline
  controller for data collection is Phase 1 work.
