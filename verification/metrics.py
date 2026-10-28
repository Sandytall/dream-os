"""Verifier metrics + classification (Component F).

Compares a predicted trajectory (s-hat) to the replayed real trajectory (s) and
classifies the dream as supported / contradicted / unresolved using explicit,
reported tolerances. The thresholds are design choices, not hidden magic: the raw
discrepancies are always reported alongside the label.
"""
from __future__ import annotations

import numpy as np

OBJ = slice(3, 6)                 # object position dims in the observation
SUPPORT_TOL_M = 0.05              # mean object-pos discrepancy at/under this -> supported
CONTRADICT_TOL_M = 0.10          # at/over this -> contradicted; between -> unresolved
RECON_TOL_M = 1e-3                # start must reconstruct this well, else unresolved


def object_discrepancy(pred_states: np.ndarray, real_states: np.ndarray) -> np.ndarray:
    """Per-step L2 distance between predicted and real object position."""
    n = min(len(pred_states), len(real_states))
    diff = pred_states[:n, OBJ] - real_states[:n, OBJ]
    return np.linalg.norm(diff, axis=1)       # [n]


def classify(per_step, plausible: bool, recon_err: float) -> tuple[str, dict]:
    """Return (label, stats). Labels: supported | contradicted | unresolved."""
    if recon_err is None or recon_err > RECON_TOL_M:
        return "unresolved", {"reason": "start not reconstructed", "recon_err": recon_err}
    if per_step is None or not np.all(np.isfinite(per_step)):
        return "contradicted", {"reason": "non-finite prediction"}
    if not plausible:
        # diverged (out-of-distribution) dream: model fantasy vs bounded reality
        return "contradicted", {"reason": "diverged prediction",
                                "mean_obj_disc_m": float(np.mean(per_step[1:]))}
    mean_d = float(np.mean(per_step[1:])) if len(per_step) > 1 else float(per_step[0])
    final_d = float(per_step[-1])
    stats = {"mean_obj_disc_m": round(mean_d, 4), "final_obj_disc_m": round(final_d, 4)}
    if mean_d <= SUPPORT_TOL_M:
        return "supported", stats
    if mean_d >= CONTRADICT_TOL_M:
        return "contradicted", stats
    return "unresolved", stats
