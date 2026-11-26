"""Phase-gated gripper control used at execution time (numpy, no JAX dependency)."""

from __future__ import annotations

import numpy as np

from recast import PHASES

GRASP, CARRY, RELEASE = PHASES.index("grasp"), PHASES.index("carry"), PHASES.index("release")


def gate_gripper(gripper_cmd: float | np.ndarray, phase: int | np.ndarray) -> np.ndarray:
    """1 = open. Forced closed in grasp/carry, free to open only in release, open in idle/approach."""
    phase = np.asarray(phase)
    closed = (phase == GRASP) | (phase == CARRY)
    return np.where(closed, 0.0, np.where(phase == RELEASE, gripper_cmd, 1.0))
