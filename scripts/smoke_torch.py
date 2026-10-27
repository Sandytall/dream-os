"""Stage 0 smoke test: PyTorch install + device.

Verifies tensor ops work and reports whether CUDA is actually usable on this
machine (driver-compatible), with an honest CPU fallback. No result is assumed;
every value is measured at runtime.
"""
from __future__ import annotations

import torch


def run() -> dict:
    info: dict = {
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cpu",
    }

    # basic CPU tensor op (must always work)
    a = torch.randn(512, 512)
    cpu_ok = torch.isfinite((a @ a).sum()).item()
    info["cpu_matmul_ok"] = bool(cpu_ok)

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        info["gpu_matmul_ok"] = bool(torch.isfinite(torch.tensor(y)).item())
        free, total = torch.cuda.mem_get_info()
        info["vram_free_gb"] = round(free / 1e9, 2)
        info["vram_total_gb"] = round(total / 1e9, 2)

    info["passed"] = bool(info["cpu_matmul_ok"])
    return info


if __name__ == "__main__":
    import json

    r = run()
    print(json.dumps(r, indent=2))
    print("TORCH SMOKE:", "PASS" if r["passed"] else "FAIL",
          f"(device={r['device']})")
