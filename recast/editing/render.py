"""(v) Render a Franka Panda (MuJoCo Menagerie) at the extracted gripper poses and composite it over the
inpainted frames with depth, so real scene geometry in front of the robot still occludes it.

The camera sits at the origin of the MuJoCo world (OpenCV convention) with the real colour intrinsics, and the
robot base is placed at T^C_B. With a calibrated camera, T^C_B = (T^W_C)^{-1} with W the robot base. Without
calibration, ``virtual_base`` places the base on the floor plane fitted to the sensor depth, out of view, so that
every target is reachable and the robot covers as little of the image as possible.

Occlusion uses a static scene depth (per-pixel median over frames without the human) plus the object's current
depth. IK is warm-started from the previous frame so the arm configuration stays consistent over time.

    MUJOCO_GL=egl python -m recast.editing.render runs/cube --menagerie third_party/mujoco_menagerie
Output ``<run>/edited/<frame>.png`` (robot composite), ``robot_setup.json`` and ``joints.npz`` (IK joint angles,
the proprioceptive state of the edited demonstration).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from recast.geometry import invert_T, mat_to_quat, unit

TCP_OFFSET = 0.1034                                   # Panda hand frame -> centre between the finger pads (m)
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
MAX_JAW = 0.08


def mujoco_principal(cx: float, cy: float, width: int, height: int) -> tuple[float, float]:
    """OpenCV principal point -> MuJoCo ``principalpixel`` (offset from the image centre)."""
    return (width - 1) / 2 - cx, cy - (height - 1) / 2


def panda_xml(panda_dir: Path) -> str:
    """Menagerie panda.xml with a TCP site between the finger pads."""
    xml = (Path(panda_dir) / "panda.xml").read_text()
    return re.sub(r'(<body name="hand"[^>]*>)',
                  r'\1<site name="tcp" pos="0 0 %.4f" size="0.002" rgba="0 0 0 0"/>' % TCP_OFFSET, xml, count=1)


def build_camera_model(panda_dir: Path, T_cam_base: np.ndarray, k: dict):
    """MuJoCo model in the camera frame: camera 'rs' at the origin, robot base at T_cam_base."""
    import mujoco

    W, H = int(k["width"]), int(k["height"])
    q = mat_to_quat(T_cam_base[:3, :3])
    pos = " ".join(f"{v:.6f}" for v in T_cam_base[:3, 3])
    xml = panda_xml(panda_dir).replace(
        '<body name="link0" childclass="panda">',
        f'<body name="link0" childclass="panda" pos="{pos}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">', 1)
    px, py = mujoco_principal(k["cx"], k["cy"], W, H)
    cam = (f'<camera name="rs" pos="0 0 0" quat="0 1 0 0" resolution="{W} {H}" focalpixel="{k["fx"]} {k["fy"]}" '
           f'principalpixel="{px} {py}" sensorsize="0.00384 0.00288"/>'
           '<light pos="0 -0.5 0" dir="0 1 1" diffuse="0.7 0.7 0.7"/>'
           '<light pos="0.5 -1 0.5" dir="-0.4 1 0.4" diffuse="0.4 0.4 0.4"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + cam, 1)
    scene = Path(panda_dir) / "_recast_camera_scene.xml"   # written next to panda.xml so mesh paths resolve
    scene.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(scene))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 640), max(H, 480)
    return m, mujoco.MjData(m)


class IK:
    """Damped-least-squares IK on the TCP site (position + orientation)."""

    def __init__(self, m, d=None):
        import mujoco

        self.mj, self.m = mujoco, m
        self.d = d if d is not None else mujoco.MjData(m)
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.lo, self.hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]

    def solve_from(self, q0: np.ndarray, p: np.ndarray, R: np.ndarray, iters: int = 300):
        mj, m, d = self.mj, self.m, self.d
        q = q0.copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            Rc = d.site_xmat[self.sid].reshape(3, 3)
            e_p = p - d.site_xpos[self.sid]
            e_r = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mj.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.concatenate([e_p, e_r])),
                        self.lo, self.hi)
        d.qpos[:7] = q
        mj.mj_kinematics(m, d)
        Rc = d.site_xmat[self.sid].reshape(3, 3)
        pe = float(np.linalg.norm(p - d.site_xpos[self.sid]))
        re_ = float(np.degrees(np.arccos(np.clip((np.trace(Rc.T @ R) - 1) / 2, -1, 1))))
        return q, pe, re_

    def solve(self, p: np.ndarray, R: np.ndarray, seeds: int = 12, rng=None):
        """Best of ``seeds`` random restarts around the home pose: (q, position error, rotation error)."""
        rng = rng or np.random.default_rng(0)
        best = None
        for s in range(seeds):
            q0 = HOME if s == 0 else np.clip(HOME + rng.normal(scale=0.6, size=7), self.lo, self.hi)
            q, pe, re_ = self.solve_from(q0, p, R)
            if best is None or pe + 0.01 * re_ < best[1] + 0.01 * best[2]:
                best = (q, pe, re_)
        return best


# --------------------------------------------------------------------------------------------- virtual base
def fit_floor(points: np.ndarray) -> tuple[np.ndarray, float]:
    """RANSAC plane n.x + d = 0 with n pointing up (towards camera -y)."""
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    (a, b, c, d), _ = pc.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=2000)
    n = np.array([a, b, c])
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    if np.dot(n, [0, -1, 0]) < 0:
        n, d = -n, -d
    return n, float(d)


def base_pose(target_c: np.ndarray, n: np.ndarray, d: float, right: float, away: float) -> np.ndarray:
    """Base pose (camera frame) on the floor, offset right/away from the targets, x-axis facing them."""
    def horiz(v):
        return unit(v - np.dot(v, n) * n)

    c_f = target_c - (np.dot(n, target_c) + d) * n
    b = c_f + right * horiz(np.array([1.0, 0, 0])) + away * horiz(np.array([0, 0, 1.0]))
    x = horiz(c_f - b)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, np.cross(n, x), n], axis=1), b
    return T


def virtual_base(panda_dir: Path, k: dict, targets_cam: list[np.ndarray], floor: tuple[np.ndarray, float],
                 max_pos_err: float = 0.005, max_rot_err: float = 5.0) -> np.ndarray:
    """Search base placements right of / behind the workspace: out of view, every target reachable, least robot
    pixels, no link within 15 cm of the lens."""
    import mujoco

    n, dpl = floor
    W, H = int(k["width"]), int(k["height"])
    c = np.mean([T[:3, 3] for T in targets_cam], axis=0)
    best = None
    for right in np.arange(0.30, 0.75, 0.10):
        for away in np.arange(0.0, 0.45, 0.10):
            T = base_pose(c, n, dpl, right, away)
            b = T[:3, 3]
            if b[2] > 0.05 and 0 <= k["fx"] * b[0] / b[2] + k["cx"] < W and 0 <= k["fy"] * b[1] / b[2] + k["cy"] < H:
                continue
            shoulder = b + 0.333 * n
            if max(np.linalg.norm(t[:3, 3] - shoulder) for t in targets_cam) > 0.80:
                continue
            m, d = build_camera_model(panda_dir, T, k)
            ik = IK(m, d)
            sols = [ik.solve(t[:3, 3], t[:3, :3], seeds=6) for t in targets_cam]
            if not all(pe <= max_pos_err and re_ <= max_rot_err for _, pe, re_ in sols):
                continue
            r = mujoco.Renderer(m, H, W)
            r.enable_depth_rendering()
            cover, nearest = 0, np.inf
            for q, _, _ in sols:
                d.qpos[:7] = q
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera="rs")
                rd = r.render()
                cover += int((rd < 10).sum())
                nearest = min(nearest, float(rd.min()))
            r.close()
            if nearest >= 0.15 and (best is None or cover < best[0]):
                best = (cover, T)
    if best is None:
        raise RuntimeError("no base placement reaches every target")
    return best[1]


# --------------------------------------------------------------------------------------------- compositing
def static_scene_depth(depths: list[np.ndarray], human: list[np.ndarray], obj: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median depth (m) over frames without the human; object pixels ignored (it moves)."""
    stack = []
    for dep, hm, om in zip(depths, human, obj):
        if hm.any():
            continue
        d = dep.astype(np.float32) / 1000.0
        d[(dep == 0) | om] = np.nan
        stack.append(d)
    if not stack:
        return np.full(depths[0].shape, np.nan, np.float32)
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(stack), axis=0)


def composite(background: np.ndarray, robot_bgr: np.ndarray, robot_depth: np.ndarray, static_depth: np.ndarray,
              cur_depth_mm: np.ndarray, obj_mask: np.ndarray, margin: float = 0.01) -> np.ndarray:
    """Draw robot pixels unless real geometry (static scene + current object depth) is in front of them."""
    scene = static_depth.copy()
    cur = cur_depth_mm / 1000.0
    sel = obj_mask & (cur > 0)
    scene[sel] = cur[sel]
    robot = robot_depth < 10.0
    occluded = np.isfinite(scene) & (scene < robot_depth - margin)
    show = robot & ~occluded
    out = background.copy()
    out[show] = robot_bgr[show]
    return out


class RobotRenderer:
    def __init__(self, panda_dir: Path, T_cam_base: np.ndarray, k: dict):
        import mujoco

        self.mj = mujoco
        self.m, self.d = build_camera_model(panda_dir, T_cam_base, k)
        self.ik = IK(self.m, mujoco.MjData(self.m))
        self.r = mujoco.Renderer(self.m, int(k["height"]), int(k["width"]))

    def render(self, q: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:7] = q
        self.d.qpos[7:9] = np.clip(width / 2, 0.0, MAX_JAW / 2)
        self.mj.mj_forward(self.m, self.d)
        self.r.disable_depth_rendering()
        self.r.update_scene(self.d, camera="rs")
        rgb = self.r.render()[:, :, ::-1].copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(self.d, camera="rs")
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        return rgb, depth


def render_run(run: Path, panda_dir: Path, T_cam_base: np.ndarray | None = None) -> dict:
    from recast.geometry import backproject
    from recast.io.clip import Clip, read_mask

    clip = Clip(run / "clip")
    with np.load(run / "actions" / "actions.npz") as f:
        T_WE, width, valid = f["T_WE"], f["width"], f["valid"]
    with np.load(run / "keypoints" / "keypoints.npz") as f:
        T_wc = f["T_wc"]
    T_cw = invert_T(T_wc)
    targets_cam = {t: T_cw @ T_WE[t] for t in np.flatnonzero(valid)}
    depths = [clip.depth(i) for i in range(len(clip))]
    human = [read_mask(run / "masks" / "remove" / f"{n}.png") for n in clip.names]
    obj = [read_mask(run / "masks" / "object" / f"{n}.png") for n in clip.names]
    if T_cam_base is None:
        if np.allclose(T_wc, np.eye(4)):     # no calibration: virtual base on the floor
            h = depths[0].shape[0]
            rows = np.zeros_like(depths[0], bool)
            rows[h // 2:] = True
            floor = fit_floor(backproject(depths[0], rows, clip.k))
            T_cam_base = virtual_base(panda_dir, clip.k, list(targets_cam.values()), floor)
        else:
            T_cam_base = T_cw
    rr = RobotRenderer(panda_dir, T_cam_base, clip.k)
    static = static_scene_depth(depths, human, obj)
    out = run / "edited"
    out.mkdir(parents=True, exist_ok=True)
    errs, q = {}, None
    joints = np.zeros((len(clip), 7))
    for i, name in enumerate(clip.names):
        bg = cv2.imread(str(run / "inpainted" / f"{name}.png"))
        if i in targets_cam:
            p, R = targets_cam[i][:3, 3], targets_cam[i][:3, :3]
            if q is not None:
                q, pe, _ = rr.ik.solve_from(q, p, R)
            if q is None or pe > 0.005:          # first frame or lost track: multi-start
                q, pe, _ = rr.ik.solve(p, R)
            errs[i] = pe
            joints[i] = q
            rgb, rdepth = rr.render(q, width[i])
            bg = composite(bg, rgb, rdepth, static, depths[i], obj[i])
        cv2.imwrite(str(out / f"{name}.png"), bg)
    setup = {"T_cam_base": T_cam_base.tolist(), "calibrated": not np.allclose(T_wc, np.eye(4)),
             "frames_with_robot": len(errs), "ik_pos_err_mm_max": 1000 * max(errs.values()) if errs else None}
    (out / "robot_setup.json").write_text(json.dumps(setup, indent=2))
    np.savez_compressed(out / "joints.npz", q=joints, valid=np.isin(np.arange(len(clip)), list(errs)))
    return setup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--menagerie", type=Path, default=Path("third_party/mujoco_menagerie"))
    ap.add_argument("--base", type=Path, help="JSON with a 4x4 'T_cam_base' to reuse a robot placement")
    a = ap.parse_args()
    T = np.array(json.loads(a.base.read_text())["T_cam_base"]) if a.base else None
    print(json.dumps(render_run(a.run, a.menagerie / "franka_emika_panda", T), indent=2))


if __name__ == "__main__":
    main()
