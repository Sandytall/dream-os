#!/usr/bin/env bash
# Reproduce the DREAM-OS Stage 0 environment.
# Target: Ubuntu 22.04, Python 3.10, NVIDIA driver >=570 (CUDA 12.x), uv installed.
# Creates an ISOLATED venv (no system-site, ROS PYTHONPATH stripped) and verifies it.
set -euo pipefail
cd "$(dirname "$0")"

# ROS Humble injects /opt/ros/... into PYTHONPATH on this machine; keep it out.
unset PYTHONPATH

echo ">> creating isolated venv (system CPython 3.10)"
uv venv --python /usr/bin/python3.10 .venv

echo ">> installing torch (CUDA 12.8 build — matches driver 575 / CUDA 12.9)"
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0

echo ">> installing simulator + benchmark stack (PyPI)"
uv pip install --python .venv/bin/python \
  mujoco==3.3.6 gymnasium==1.4.0 gymnasium-robotics==1.4.2 \
  numpy==2.2.6 scipy==1.15.3 pyyaml==6.0.3

echo ">> running Stage 0 smoke tests"
.venv/bin/python scripts/stage0_report.py

echo ">> done. Activate with:  unset PYTHONPATH && source .venv/bin/activate"
