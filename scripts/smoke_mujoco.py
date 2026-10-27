"""Stage 0 smoke test: MuJoCo physics actually steps.

Loads a real MuJoCo model, advances the physics, and asserts the state changes
in the direction physics requires (a free body falls under gravity). This proves
the installed engine computes dynamics, not that any benchmark works.
"""
from __future__ import annotations

import mujoco

# A minimal but genuine MJCF model: one free body under gravity.
_MJCF = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run(steps: int = 100) -> dict:
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)

    z_start = float(data.qpos[2])  # free joint qpos = [x y z qw qx qy qz]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    z_end = float(data.qpos[2])

    fell = z_end < z_start - 0.01
    return {
        "mujoco_version": mujoco.__version__,
        "steps": steps,
        "z_start": round(z_start, 4),
        "z_end": round(z_end, 4),
        "dropped_m": round(z_start - z_end, 4),
        "passed": bool(fell),
    }


if __name__ == "__main__":
    import json

    r = run()
    print(json.dumps(r, indent=2))
    assert r["passed"], "MuJoCo did not step physics (body did not fall)"
    print(f"MUJOCO SMOKE: PASS (mujoco {r['mujoco_version']}, "
          f"ball dropped {r['dropped_m']} m)")
