"""Simulated Panda with either joint-position servoing (IK targets) or torque control (OSC targets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recast.editing.render import HOME
from recast.geometry import invert_T, make_T
from recast.sim.control import IK, OSC
from recast.sim.scene import SceneSpec, write_scene

OPEN, CLOSED = 255.0, 0.0                    # Panda hand actuator (0..255, 255 = open)
TIMESTEP = 0.002


class PandaSim:
    def __init__(self, spec: SceneSpec, panda_dir: Path, control: str = "position", seed: int = 0,
                 render_size: tuple[int, int] = (480, 640)):
        import mujoco

        self.mj, self.spec, self.control = mujoco, spec, control
        self.m = mujoco.MjModel.from_xml_path(str(write_scene(spec, panda_dir, control, seed)))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1920, 1080
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.obj_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "object")
        if control == "position":       # servoed arm: compensate gravity on the robot links only
            for b in range(self.m.nbody):
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                self.m.body_gravcomp[b] = 1.0 if name.startswith(("link", "hand", "left_finger", "right_finger")) else 0
        self.ik, self.osc = IK(self.m), OSC(self.m) if control == "torque" else None
        self.q, self.grip = HOME.copy(), OPEN
        self.d.qpos[:7], self.d.qpos[7:9] = self.q, 0.04
        self.d.ctrl[:7] = self.q if control == "position" else 0.0
        self.d.ctrl[7] = self.grip
        mujoco.mj_forward(self.m, self.d)
        self.target = self.tcp_pose()
        self.renderers: dict[str, object] = {}
        self.render_size = render_size
        self.frame_hooks: list = []
        self.steps = 0

    # ----------------------------------------------------------------------------------------- state
    def tcp_pose(self) -> np.ndarray:
        return make_T(self.d.site_xmat[self.sid].reshape(3, 3).copy(), self.d.site_xpos[self.sid].copy())

    def object_pose(self) -> np.ndarray:
        return make_T(self.d.xmat[self.obj_id].reshape(3, 3).copy(), self.d.xpos[self.obj_id].copy())

    def gripper_opening(self) -> float:
        return float(self.d.qpos[7] + self.d.qpos[8])

    @property
    def time(self) -> float:
        return self.steps * TIMESTEP

    # ----------------------------------------------------------------------------------------- stepping
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.control == "position":
                self.d.ctrl[:7] = self.q
            else:
                self.d.ctrl[:7] = self.osc.torque(self.d, self.target[:3, 3], self.target[:3, :3])
            self.d.ctrl[7] = self.grip
            self.mj.mj_step(self.m, self.d)
            self.steps += 1
            for hook in self.frame_hooks:
                hook(self)

    def settle(self, seconds: float) -> None:
        self.step(int(seconds / TIMESTEP))

    def set_target(self, T_des: np.ndarray) -> None:
        """Torque mode: new OSC target. Position mode: IK from the current joint targets."""
        self.target = T_des
        if self.control == "position":
            self.q, _, _ = self.ik.solve_from(self.q, T_des[:3, 3], T_des[:3, :3])

    def move_to(self, p: np.ndarray, R: np.ndarray, seconds: float) -> None:
        """Minimum-jerk Cartesian line from the current TCP to p with fixed orientation R."""
        p0 = self.tcp_pose()[:3, 3]
        n = max(int(seconds / 0.02), 2)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * s * (10 - 15 * s + 6 * s * s)
            self.set_target(make_T(R, p0 + (p - p0) * s))
            self.step(10)

    def joint_move(self, q_goal: np.ndarray, seconds: float) -> None:
        q0, n = self.q.copy(), max(int(seconds / 0.02), 2)
        for i in range(1, n + 1):
            s = i / n
            self.q = q0 + (q_goal - q0) * s * s * (3 - 2 * s)
            self.step(10)

    def set_gripper(self, value: float, seconds: float) -> None:
        g0, n = self.grip, max(int(seconds / 0.02), 2)
        for i in range(1, n + 1):
            self.grip = g0 + (value - g0) * i / n
            self.step(10)

    def apply_action(self, action: np.ndarray, gripper: float, seconds: float) -> None:
        """Policy action: base-frame delta pose [dp, drot (axis-angle)] and gripper in [0, 1] (1 = open)."""
        from recast.geometry import quat_to_mat

        T = self.target.copy()
        T[:3, 3] += action[:3]
        angle = np.linalg.norm(action[3:6])
        if angle > 1e-9:
            ax = action[3:6] / angle
            q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * ax])
            T[:3, :3] = quat_to_mat(q) @ T[:3, :3]
        self.set_target(T)
        self.grip = float(np.clip(gripper, 0, 1)) * OPEN
        self.step(int(seconds / TIMESTEP))

    # ----------------------------------------------------------------------------------------- observations
    def render(self, camera: str = "rs", size: tuple[int, int] | None = None) -> np.ndarray:
        """RGB uint8."""
        h, w = size or ((self.spec.k["height"], self.spec.k["width"]) if camera == "rs" else self.render_size)
        key = (camera, h, w)
        if key not in self.renderers:
            self.renderers[key] = self.mj.Renderer(self.m, h, w)
        r = self.renderers[key]
        r.update_scene(self.d, camera=camera)
        return r.render().copy()

    def object_keypoints_ee(self, local_points: np.ndarray) -> np.ndarray:
        """Ground-truth object keypoints (given in the object frame) expressed in the end-effector frame."""
        T_EO = invert_T(self.tcp_pose()) @ self.object_pose()
        return local_points @ T_EO[:3, :3].T + T_EO[:3, 3]

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()
