"""High-quality MP4 renders: real FetchPush clips at 720p + a multi-box physics sandbox.

Part A re-renders real benchmark rollouts at 720p with the clean default camera:
the push-to-goal episode, the do-nothing dream where the real block coasts and
drops off the table, and a supported dream. These are honest DREAM-OS rollouts.

Part B is a clearly-labelled VISUAL-ONLY sandbox: a hand-built MuJoCo scene with
nine boxes given an outward shove so they tumble off a table. Real MuJoCo physics,
for show only -- NOT part of the learning system, which uses the unmodified
single-block FetchPush benchmark.

Video is written by piping raw frames to the system ffmpeg (libx264).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from envs.benchmark_interface import make_env               # noqa: E402
from memory import episode_store as store                   # noqa: E402
from dreaming import dream_store as ds                       # noqa: E402
from verification import replay as vr                        # noqa: E402

DATASET_ID = "fetchpush_scripted_phase1"
BATCH_ID = "phase3_fetchpush"
FIG = ROOT / "reports" / "figures"
RES = 720


def write_mp4(frames, path, fps):
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    err = p.stderr.read().decode()[:300]
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err}")
    return path.stat().st_size


# --------------------------------------------------------------------------- #
# Part A: real FetchPush clips (720p, default camera)
# --------------------------------------------------------------------------- #
def render_real_episode(env, episode):
    seed = vr.episode_seed(DATASET_ID, episode)
    d = store.load_episode(DATASET_ID, episode)
    env.reset(seed=seed)
    frames = [env.render()]
    for a in d["actions"]:
        env.step(a); frames.append(env.render())
    return frames


def render_dream_real(env, ref, dream_actions):
    seed = vr.episode_seed(ref["dataset_id"], ref["episode"])
    d = store.load_episode(ref["dataset_id"], ref["episode"])
    env.reset(seed=seed)
    for a in d["actions"][: ref["t"]]:
        env.step(a)
    frames = [env.render()]
    for a in dream_actions:
        env.step(a); frames.append(env.render())
    return frames


def find_success_episode():
    for split in ("test_heldout", "train"):
        for ep in store.splits(DATASET_ID)[split]:
            d = store.load_episode(DATASET_ID, ep)
            if "is_success" in d and float(np.asarray(d["is_success"]).reshape(-1)[-1]) >= 1.0:
                return ep
    return store.splits(DATASET_ID)["test_heldout"][0]


def part_a():
    dm = json.loads((ROOT / "data/manifests" / f"dreams_{BATCH_ID}.json").read_text())
    vmf = json.loads((ROOT / "data/manifests" / f"verification_{BATCH_ID}.json").read_text())
    dreams = {d["dream_id"]: d for d in dm["dreams"]}
    noop_id = next(r["dream_id"] for r in vmf["records"] if r["branch"] == "noop")
    sup_id = next(r["dream_id"] for r in vmf["records"]
                  if r["classification"] == "supported" and r["branch"] != "noop")

    env = make_env("FetchPush-v4", render_mode="rgb_array", width=RES, height=RES)
    clips = {}

    succ = find_success_episode()
    print(f"[hq] success episode {succ} ...")
    f = render_real_episode(env, succ)
    clips["success"] = {"file": "assets/hq_success.mp4", "kb": round(write_mp4(f, FIG / "hq_success.mp4", 18) / 1024)}

    print(f"[hq] do-nothing real drop ({noop_id}) ...")
    f = render_dream_real(env, dreams[noop_id]["start_ref"], ds.load_dream_arrays(BATCH_ID, noop_id)["actions"])
    Image.fromarray(f[len(f) // 2]).save(FIG / "sample_drop_mid.png")
    clips["drop_real"] = {"file": "assets/hq_drop.mp4", "kb": round(write_mp4(f, FIG / "hq_drop.mp4", 12) / 1024)}

    print(f"[hq] supported dream ({sup_id}) ...")
    f = render_dream_real(env, dreams[sup_id]["start_ref"], ds.load_dream_arrays(BATCH_ID, sup_id)["actions"])
    clips["match"] = {"file": "assets/hq_match.mp4", "kb": round(write_mp4(f, FIG / "hq_match.mp4", 12) / 1024)}

    env.close()
    for k, v in clips.items():
        print(f"   {k}: {v['kb']} KB")
    return clips


# --------------------------------------------------------------------------- #
# Part B: visual-only multi-box sandbox
# --------------------------------------------------------------------------- #
SANDBOX_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual><global offwidth="{res}" offheight="{res}"/><headlight diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.72 0.70 0.66"
             width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="6 6" reflectance="0.1"/>
    <material name="wood" rgba="0.80 0.66 0.44 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 2.2" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="floor"/>
    <geom name="table" type="box" size="0.30 0.30 0.02" pos="0 0 0.50" material="wood"/>
    <geom name="leg1" type="box" size="0.02 0.02 0.25" pos="0.26 0.26 0.25" material="wood"/>
    <geom name="leg2" type="box" size="0.02 0.02 0.25" pos="-0.26 0.26 0.25" material="wood"/>
    <geom name="leg3" type="box" size="0.02 0.02 0.25" pos="0.26 -0.26 0.25" material="wood"/>
    <geom name="leg4" type="box" size="0.02 0.02 0.25" pos="-0.26 -0.26 0.25" material="wood"/>
    {boxes}
  </worldbody>
</mujoco>
"""

BOX_COLORS = [(0.90, 0.30, 0.24), (0.25, 0.55, 0.90), (0.95, 0.72, 0.20),
              (0.35, 0.72, 0.42), (0.70, 0.40, 0.80), (0.95, 0.55, 0.25),
              (0.30, 0.75, 0.75), (0.85, 0.45, 0.60), (0.55, 0.60, 0.65)]


def build_sandbox():
    boxes, pos = [], []
    k = 0
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            x, y = 0.11 * ix, 0.11 * iy
            r, g, b = BOX_COLORS[k]
            boxes.append(f'<body name="b{k}" pos="{x:.3f} {y:.3f} 0.558"><freejoint/>'
                         f'<geom type="box" size="0.032 0.032 0.032" rgba="{r} {g} {b} 1" '
                         f'friction="0.6 0.02 0.001"/></body>')
            pos.append((x, y)); k += 1
    xml = SANDBOX_XML.format(res=RES, boxes="\n    ".join(boxes))
    return mujoco.MjModel.from_xml_string(xml), pos


def part_b():
    model, pos = build_sandbox()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(0)
    for i, (x, y) in enumerate(pos):
        v = np.array([x, y])
        v = v / (np.linalg.norm(v) + 1e-6) if np.linalg.norm(v) > 1e-6 else np.array([1.0, 0.0])
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")
        adr = model.jnt_dofadr[model.body_jntadr[bid]]
        data.qvel[adr:adr + 3] = [v[0] * 1.7 + rng.normal(0, 0.15),
                                  v[1] * 1.7 + rng.normal(0, 0.15), 0.55]
        data.qvel[adr + 3:adr + 6] = rng.normal(0, 5, 3)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.32]; cam.distance = 2.1; cam.azimuth = 128; cam.elevation = -16
    renderer = mujoco.Renderer(model, RES, RES)
    frames = []
    for step in range(1400):
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            renderer.update_scene(data, cam)
            frames.append(renderer.render())
    renderer.close()
    Image.fromarray(frames[len(frames) // 4]).save(FIG / "sample_sandbox.png")
    sz = write_mp4(frames, FIG / "sandbox_boxes.mp4", fps=50)
    print(f"[hq] sandbox: {len(frames)} frames  {sz/1024:.0f} KB")
    return {"file": "assets/sandbox_boxes.mp4", "boxes": len(pos), "frames": len(frames),
            "kb": round(sz / 1024)}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    clips = part_a()
    sandbox = part_b()
    meta = {"resolution": RES, "clips": clips, "sandbox": sandbox,
            "note": "FetchPush clips are real benchmark rollouts; sandbox is a visual-only "
                    "physics scene, not used for training or evaluation"}
    (FIG / "hq_meta.json").write_text(json.dumps(meta, indent=2))
    print("=" * 60)
    print(f"HQ render complete ({RES}p) -> reports/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
