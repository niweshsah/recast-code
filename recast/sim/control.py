"""Controllers for the simulated Panda.

* ``OSC``: operational-space control (Khatib 1987), the controller used for policy evaluation. Joint torques
      tau = J^T Lambda (Kp e - Kd J qdot) + N^T (posture torque) + bias,
  with Lambda = (J M^-1 J^T)^-1 and the dynamically consistent null-space projector N.
* ``IK``: damped least squares on the TCP site (shared with the renderer), used by the scripted demonstration.
"""

from __future__ import annotations

import numpy as np

from recast.editing.render import HOME, IK  # noqa: F401  (re-exported)


def orientation_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Small-angle rotation error (world frame) driving R towards R_des."""
    return 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))


def full_inertia(mj, m, d, out: np.ndarray) -> None:
    """Dense joint-space inertia; mj_fullM takes (m, d, dst) in MuJoCo >= 3.5 and (m, dst, qM) before."""
    try:
        mj.mj_fullM(m, d, out)
    except TypeError:
        mj.mj_fullM(m, out, d.qM)


class OSC:
    def __init__(self, m, site: str = "tcp", kp_pos: float = 300.0, kp_rot: float = 60.0, damping: float = 1.0,
                 kp_null: float = 10.0, q_null: np.ndarray = HOME):
        import mujoco

        self.mj, self.m = mujoco, m
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        self.kp = np.array([kp_pos] * 3 + [kp_rot] * 3)
        self.kd = 2.0 * damping * np.sqrt(self.kp)          # critically damped per axis
        self.kp_null, self.q_null = kp_null, np.asarray(q_null)
        self.lo, self.hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]

    def torque(self, d, p_des: np.ndarray, R_des: np.ndarray) -> np.ndarray:
        mj, m = self.mj, self.m
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mj.mj_jacSite(m, d, jp, jr, self.sid)
        J = np.vstack([jp[:, :7], jr[:, :7]])
        M_full = np.zeros((m.nv, m.nv))
        full_inertia(mj, m, d, M_full)
        M = M_full[:7, :7]
        M_inv = np.linalg.inv(M)
        Lam = np.linalg.pinv(J @ M_inv @ J.T, rcond=1e-3)
        qd = d.qvel[:7]
        e = np.concatenate([p_des - d.site_xpos[self.sid], orientation_error(d.site_xmat[self.sid].reshape(3, 3),
                                                                             R_des)])
        F = Lam @ (self.kp * e - self.kd * (J @ qd))
        J_bar = M_inv @ J.T @ Lam                             # dynamically consistent pseudo-inverse
        N = np.eye(7) - J.T @ J_bar.T
        tau_null = M @ (self.kp_null * (self.q_null - d.qpos[:7]) - 2 * np.sqrt(self.kp_null) * qd)
        tau = J.T @ F + N @ tau_null + d.qfrc_bias[:7]
        return np.clip(tau, self.lo, self.hi)
