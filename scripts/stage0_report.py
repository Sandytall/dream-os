"""Stage 0 orchestrator: run every smoke test, measure resources, emit a report.

Produces reports/stage0_results.json with provenance (timestamp, versions, git
commit, seeds) and a human summary. Honest by construction: it records whatever
the smoke tests actually returned, including failures.
"""
from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import smoke_torch
import smoke_mujoco
import smoke_benchmark


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uninitialised"


def main() -> int:
    t0 = time.time()
    report = {
        "stage": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": {"reset_seed": 0, "action_seed": 123},
        "tests": {},
    }

    report["tests"]["torch"] = smoke_torch.run()
    report["tests"]["mujoco"] = smoke_mujoco.run()
    report["tests"]["benchmark"] = smoke_benchmark.run()

    report["wall_seconds"] = round(time.time() - t0, 2)
    # ru_maxrss is KiB on Linux
    report["peak_ram_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)

    report["all_passed"] = bool(
        report["tests"]["torch"]["passed"]
        and report["tests"]["mujoco"]["passed"]
        and report["tests"]["benchmark"]["passed"])

    out = ROOT / "reports" / "stage0_results.json"
    out.write_text(json.dumps(report, indent=2))

    print("=" * 60)
    print("DREAM-OS Stage 0 smoke report")
    print("=" * 60)
    t = report["tests"]
    print(f"  torch      : {'PASS' if t['torch']['passed'] else 'FAIL'}  "
          f"({t['torch']['torch_version']}, device={t['torch']['device']})")
    print(f"  mujoco     : {'PASS' if t['mujoco']['passed'] else 'FAIL'}  "
          f"({t['mujoco']['mujoco_version']})")
    b = t["benchmark"]
    print(f"  benchmark  : {'PASS' if b['passed'] else 'FAIL'}  "
          f"(gymnasium {b['gymnasium_version']})")
    for key in ("pusher", "fetchpush"):
        if key in b and isinstance(b[key], dict):
            e = b[key]
            print(f"      - {e.get('env_id', key):16s} "
                  f"obs_dim={e.get('obs_dim')} qpos={e.get('qpos_dim')} "
                  f"success={e.get('is_success')} reproducible={e.get('reproducible')}")
    print(f"  wall={report['wall_seconds']}s  peak_ram={report['peak_ram_gb']}GB")
    print(f"  -> written to {out}")
    print("ALL PASSED:", report["all_passed"])
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
