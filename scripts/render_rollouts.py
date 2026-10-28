"""Visualization utility -- render the real sim and export imagination-vs-reality data.

This renders ONLY real MuJoCo rollouts (what actually happened). The model's
imagined trajectories are not simulator states, so they are exported as numeric
paths for overlay plotting, never rendered as if they were real. Every number here
comes from recorded data or a real replay; nothing is fabricated.

Outputs (reports/figures/):
  episode_success.gif   -- a real scripted-push episode (the task)
  dream_noop.gif        -- real rollout of the contradicted noop dream
  dream_supported.gif   -- real rollout of a supported dream
  viz_data.json         -- predicted vs real object paths, classifications, training curve
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # GPU offscreen rendering (probed working)

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env, obs_parts    # noqa: E402
from memory import episode_store as store                   # noqa: E402
from dreaming import dream_store as ds                       # noqa: E402
from verification import replay as vr                        # noqa: E402

BATCH_ID = "phase3_fetchpush"
DATASET_ID = "fetchpush_scripted_phase1"
OBJ = slice(3, 6)
FIG = ROOT / "reports" / "figures"
FRAME = 300          # output GIF square size (px)


def save_gif(frames, path, ms=70):
    imgs = [Image.fromarray(f).resize((FRAME, FRAME)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0, optimize=True)
    return len(imgs), path.stat().st_size


def render_real_episode(env, dataset_id, episode):
    seed = vr.episode_seed(dataset_id, episode)
    d = store.load_episode(dataset_id, episode)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for a in d["actions"]:
        obs, *_ = env.step(a)
        frames.append(env.render())
    return frames


def render_dream_real(env, ref, dream_actions):
    """Reach the dream's real start (reset+replay), then render the dream's real rollout."""
    seed = vr.episode_seed(ref["dataset_id"], ref["episode"])
    d = store.load_episode(ref["dataset_id"], ref["episode"])
    obs, _ = env.reset(seed=seed)
    for a in d["actions"][: ref["t"]]:
        obs, *_ = env.step(a)
    frames = [env.render()]
    for a in dream_actions:
        obs, *_ = env.step(a)
        frames.append(env.render())
    return frames


def find_success_episode():
    for split in ("test_heldout", "train"):
        for ep in store.splits(DATASET_ID)[split]:
            d = store.load_episode(DATASET_ID, ep)
            if "is_success" in d and float(np.asarray(d["is_success"]).reshape(-1)[-1]) >= 1.0:
                return ep
    return store.splits(DATASET_ID)["test_heldout"][0]        # fallback: any episode


def training_curve():
    """A real training run's validation curve, logged on the OBJECT position (the
    metric that actually matters) so real learning is visible, not hidden under the
    near-unpredictable rotation floor. Persistence baseline reported as a reference."""
    import torch
    from world_model import training as T
    from world_model.dynamics import ResidualDynamics
    device = "cpu"
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).item(); device = "cuda"
        except Exception:
            device = "cpu"
    OBJ_C = [3, 4, 5]
    tr = store.transitions(DATASET_ID, "train")
    va = store.transitions(DATASET_ID, "val")
    stats = T.compute_stats(*tr)
    torch.manual_seed(0)
    m = ResidualDynamics(tr[0].shape[1], tr[1].shape[1], stats).to(device)
    S, A, Sn = (torch.as_tensor(x, device=device) for x in tr)
    Sv, Av, Snv = (torch.as_tensor(x, device=device) for x in va)
    d_true = ((Sn - S) - m.d_mean) / m.d_std.clamp_min(1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = S.shape[0]
    hist, best, bad, best_state = [], float("inf"), 0, None
    for epoch in range(400):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            loss = ((m.pred_delta_norm(S[idx], A[idx]) - d_true[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m.predict(Sv, Av)
            val_all = float(torch.sqrt(((pv - Snv) ** 2).mean()))
            val_obj = float(torch.sqrt(((pv[:, OBJ_C] - Snv[:, OBJ_C]) ** 2).mean()))
        hist.append({"epoch": epoch, "val_object_rmse": round(val_obj, 5),
                     "val_all_rmse": round(val_all, 5)})
        if val_all < best - 1e-7:
            best, bad = val_all, 0
            best_state = epoch
        else:
            bad += 1
            if bad >= 40:
                break
    persist_obj = float(np.sqrt(np.mean((va[0][:, OBJ_C] - va[2][:, OBJ_C]) ** 2)))
    return {"history": hist, "epochs_run": len(hist), "device": device,
            "best_epoch": best_state,
            "best_val_object_rmse": round(min(h["val_object_rmse"] for h in hist), 5),
            "persistence_val_object_rmse": round(persist_obj, 5)}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    dm = json.loads((ROOT / "data/manifests" / f"dreams_{BATCH_ID}.json").read_text())
    vmf = json.loads((ROOT / "data/manifests" / f"verification_{BATCH_ID}.json").read_text())
    dreams = {d["dream_id"]: d for d in dm["dreams"]}
    records = {r["dream_id"]: r for r in vmf["records"]}

    noop_id = next(r["dream_id"] for r in vmf["records"] if r["branch"] == "noop")
    sup_id = next(r["dream_id"] for r in vmf["records"]
                  if r["classification"] == "supported" and r["branch"] != "noop")

    env = make_env("FetchPush-v4", render_mode="rgb_array")
    gifs = {}

    succ_ep = find_success_episode()
    print(f"[render] success episode {succ_ep} ...")
    n, sz = save_gif(render_real_episode(env, DATASET_ID, succ_ep), FIG / "episode_success.gif")
    gifs["episode_success"] = {"episode": int(succ_ep), "frames": n, "bytes": sz}

    for tag, did, fname in ((("noop", noop_id, "dream_noop.gif")),
                            ("supported", sup_id, "dream_supported.gif")):
        d = dreams[did]
        arr = ds.load_dream_arrays(BATCH_ID, did)
        print(f"[render] dream {did} ({tag}) ...")
        frames = render_dream_real(env, d["start_ref"], arr["actions"])
        n, sz = save_gif(frames, FIG / fname)
        gifs[tag] = {"dream_id": did, "branch": d["branch"]["name"],
                     "classification": records[did]["classification"], "frames": n, "bytes": sz}
    env.close()

    # ---- imagination-vs-reality paths for the two rendered dreams ----
    overlays = {}
    for tag, did in (("noop", noop_id), ("supported", sup_id)):
        arr = ds.load_dream_arrays(BATCH_ID, did)
        real = np.load(ROOT / "data/verification" / BATCH_ID / f"{did}.npz")["real_states"]
        pred = arr["predicted_states"]
        overlays[tag] = {
            "dream_id": did, "branch": dreams[did]["branch"]["name"],
            "classification": records[did]["classification"],
            "mean_obj_disc_m": records[did].get("mean_obj_disc_m"),
            "predicted_obj": np.asarray(pred[:, OBJ], float).round(4).tolist(),
            "real_obj": np.asarray(real[:, OBJ], float).round(4).tolist(),
        }

    # ---- compact scatter: predicted vs real object DISPLACEMENT for all 30 dreams ----
    scatter = []
    for did, d in dreams.items():
        arr = ds.load_dream_arrays(BATCH_ID, did)
        real = np.load(ROOT / "data/verification" / BATCH_ID / f"{did}.npz")["real_states"]
        pred = arr["predicted_states"]
        pd = float(np.linalg.norm(pred[-1, OBJ] - pred[0, OBJ]))
        rd = float(np.linalg.norm(real[-1, OBJ] - real[0, OBJ]))
        scatter.append({"dream_id": did, "branch": d["branch"]["name"],
                        "plausible": bool(d["plausible"]),
                        "classification": records[did]["classification"],
                        "pred_disp_m": round(min(pd, 5.0), 4),   # clip diverged for plotting
                        "pred_disp_raw_m": round(pd, 4),
                        "real_disp_m": round(rd, 4)})

    print("[render] training curve ...")
    curve = training_curve()

    out = {
        "generated_from": "real recorded data + real MuJoCo replays (no fabrication)",
        "gifs": gifs, "overlays": overlays, "scatter": scatter, "training_curve": curve,
        "phase6": json.loads((ROOT / "reports/phase6_results.json").read_text()),
    }
    (FIG / "viz_data.json").write_text(json.dumps(out, indent=2))

    print("=" * 64)
    print("Render complete -> reports/figures/")
    for k, v in gifs.items():
        print(f"  {k:18s} {v['frames']} frames  {v['bytes']/1024:.0f} KB")
    print(f"  overlays: {list(overlays)}   scatter: {len(scatter)} dreams")
    print(f"  training curve: {curve['epochs_run']} epochs, best val-object {curve['best_val_object_rmse']} "
          f"vs persistence {curve['persistence_val_object_rmse']} (device {curve['device']})")
    print(f"  -> reports/figures/viz_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
