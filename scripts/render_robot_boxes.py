"""High-quality recording of the Fetch ROBOT pushing a line of boxes off the table.

Adds extra boxes to the real FetchPush MuJoCo scene, lines them up, then scripts the
arm (via the env's own mocap control) to sweep them off the near table edge. Real
robot, real physics. The extra boxes are a visualization of the real benchmark
scene; they are not used anywhere in the learning pipeline.
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
import gymnasium_robotics
from gymnasium_robotics.envs.fetch import push as P

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
RES = 720
FETCH = Path(gymnasium_robotics.__file__).parent / "envs" / "assets" / "fetch"

COLORS = [(0.90, 0.30, 0.24), (0.25, 0.55, 0.90), (0.95, 0.72, 0.20),
          (0.35, 0.72, 0.42), (0.70, 0.40, 0.80), (0.30, 0.75, 0.75)]
# box line along y (the push axis), at table x=1.30; gripper plows them toward -y edge (y=0.40)
LINE_Y = [0.78, 0.72, 0.66, 0.60, 0.54, 0.48, 0.42]   # object0..object6
BOX_X = 1.30
BOX_Z = 0.43


def build_model_xml():
    base = (FETCH / "push.xml").read_text()
    rows = ""
    for i, (r, g, b) in enumerate(COLORS, 1):
        rows += (f'<body name="object{i}" pos="{BOX_X} {LINE_Y[i]} {BOX_Z}">'
                 f'<joint name="object{i}:joint" type="free" damping="0.01"></joint>'
                 f'<geom size="0.025 0.025 0.025" type="box" condim="3" name="object{i}" '
                 f'rgba="{r} {g} {b} 1" mass="0.5"></geom></body>\n\t\t')
    (FETCH / "push_multi.xml").write_text(base.replace("<light directional", rows + "<light directional"))
    P.MODEL_XML_PATH = "fetch/push_multi.xml"


def write_mp4(frames, path, fps):
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close(); err = p.stderr.read().decode()[:300]; p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err}")
    return path.stat().st_size


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    build_model_xml()
    env = P.MujocoFetchPushEnv(render_mode="rgb_array", width=RES, height=RES)
    env.reset(seed=3)
    m, d = env.unwrapped.model, env.unwrapped.data
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")

    # scatter 7 boxes at random, non-overlapping positions and orientations on the table
    rng = np.random.default_rng(5)
    pts = []
    while len(pts) < 7:
        p = np.array([rng.uniform(1.17, 1.43), rng.uniform(0.52, 0.92)])
        if all(np.linalg.norm(p - q) > 0.075 for q in pts):
            pts.append(p)
    for i, p in enumerate(pts):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"object{i}:joint")
        adr = m.jnt_qposadr[jid]
        yaw = rng.uniform(-0.6, 0.6)
        d.qpos[adr:adr + 7] = [p[0], p[1], BOX_Z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid] + 6] = 0
    mujoco.mj_forward(m, d)

    frames = []

    def render():
        return env.render()

    def gpos():
        return d.site_xpos[grip].copy()

    def go_to(target, steps, every=1, k=10.0, tol=0.008, max_a=1.0):
        target = np.asarray(target, float)
        for s in range(steps):
            err = target - gpos()
            if np.linalg.norm(err) < tol:
                break
            a = np.clip(err * k, -max_a, max_a)
            env.step(np.array([a[0], a[1], a[2], 0.0], np.float32))
            if s % every == 0:
                frames.append(render())

    # settle boxes while the arm holds still
    for _ in range(12):
        env.step(np.zeros(4, np.float32))
    frames.append(render())

    def box_pos(i):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"object{i}")].copy()

    def on_table():
        out = []
        for i in range(7):
            p = box_pos(i)
            if p[2] > 0.37 and 0.41 < p[1] < 1.11 and 1.06 < p[0] < 1.55:
                out.append(i)
        return out

    for _ in range(4):                                          # record the scattered layout
        frames.append(render())
    Image.fromarray(frames[-1]).save(FIG / "qa_robot_start.png")

    def clear_box(i):
        p = box_pos(i); bx = float(np.clip(p[0], 1.12, 1.48)); by = float(p[1])
        go_to([bx, by + 0.14, 0.54], 55, every=3, k=9)         # lift + get behind this box
        go_to([bx, by + 0.11, 0.43], 22, every=3, k=9)         # lower behind it
        bx = float(np.clip(box_pos(i)[0], 1.12, 1.48))         # re-center on the box's live x
        go_to([bx, 0.28, 0.43], 160, every=2, k=7, max_a=0.45)  # push it off the -y edge
        go_to([bx, 0.28, 0.54], 16, every=3, k=9)              # lift off

    start = gpos()
    go_to([start[0], start[1], 0.56], 18, every=2, k=8)         # lift clear first
    it = 0
    while on_table() and it < 14:                               # clear whichever box is nearest the edge
        i = min(on_table(), key=lambda j: box_pos(j)[1])
        clear_box(i)
        if it == 1:
            Image.fromarray(frames[-1]).save(FIG / "qa_robot_mid.png")
        it += 1
    for _ in range(14):                                         # hold on the final frame
        frames.append(render())
    Image.fromarray(frames[-1]).save(FIG / "qa_robot_end.png")
    print(f"passes: {it}   boxes remaining on table: {len(on_table())}")

    env.close()
    sz = write_mp4(frames, FIG / "robot_boxes.mp4", fps=30)
    (FIG / "robot_boxes_meta.json").write_text(json.dumps(
        {"resolution": RES, "boxes": 7, "frames": len(frames), "kb": round(sz / 1024),
         "file": "assets/robot_boxes.mp4"}, indent=2))
    print(f"robot_boxes.mp4: {len(frames)} frames  {sz/1024:.0f} KB  -> reports/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
