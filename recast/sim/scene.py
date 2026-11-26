"""MuJoCo test scene: Franka Panda (MuJoCo Menagerie), floor, wall, target box and the task object, with the
recording camera ('rs': measured pose and real intrinsics) and a side 'overview' camera.

World frame = robot-base frame, z up, floor at z = 0. Scene dimensions come from ``configs/scene.yaml``; task
objects from ``configs/tasks/*.yaml``; held-out conditions (surface, lighting, distractors) from
``configs/eval.yaml``.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from recast.config import CONFIGS
from recast.editing.render import TCP_OFFSET, mujoco_principal, panda_xml
from recast.geometry import mat_to_quat, unit

TORQUE_LIMITS = (87, 87, 87, 87, 12, 12, 12)          # Franka joint torque limits (Nm)
FACES = ("right", "left", "up", "down", "front", "back")


@dataclass
class SceneSpec:
    obj: dict                                          # task object: shape, size/radius, mass, friction
    object_xy: np.ndarray
    object_yaw: float
    box_xy: np.ndarray
    box_size: np.ndarray                               # full extents (x, y, z)
    box_yaw: float
    T_base_cam: np.ndarray                             # recording camera pose in the base frame (OpenCV axes)
    k: dict
    wall_distance: float = 1.7
    condition: dict = field(default_factory=dict)

    @property
    def box_top(self) -> float:
        return float(self.box_size[2])

    @property
    def object_half_height(self) -> float:
        o = self.obj
        if o["shape"] == "box":
            return o["size_m"][2] / 2
        return o["radius_m"] if o.get("lying") else o["half_height_m"]


def load_scene(task_cfg: dict, condition: dict | None = None) -> SceneSpec:
    s = yaml.safe_load((CONFIGS / "scene.yaml").read_text())
    return SceneSpec(obj=task_cfg["object"], object_xy=np.array(s["object_xy"]), object_yaw=s["object_yaw"],
                     box_xy=np.array(s["box"]["xy"]), box_size=np.array(s["box"]["size"]), box_yaw=s["box"]["yaw"],
                     T_base_cam=np.array(s["camera"]["T_base_cam"]), k=s["camera"]["intrinsics"],
                     wall_distance=s["wall_distance_m"], condition=condition or {})


# --------------------------------------------------------------------------------------------- textures
def sticker_texture(size: int = 192, seed: int = 3) -> np.ndarray:
    """Stylised 3x3 Rubik's face (BGR) with black gaps (not the real colour pattern)."""
    cols = np.array([(40, 39, 214), (0, 128, 255), (240, 240, 240), (0, 220, 255), (60, 160, 40), (200, 90, 20)],
                    np.uint8)
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    cell, gap = size // 3, size // 40
    for i in range(3):
        for j in range(3):
            img[i * cell + gap:(i + 1) * cell - gap, j * cell + gap:(j + 1) * cell - gap] = cols[rng.integers(0, 6)]
    return img


def write_textures(tex_dir: Path) -> None:
    tex_dir.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(FACES):
        cv2.imwrite(str(tex_dir / f"face_{face}.png"), sticker_texture(seed=3 + i))


# --------------------------------------------------------------------------------------------- cameras
def cam_xyaxes(R_base_cam: np.ndarray) -> np.ndarray:
    """MuJoCo camera axes (x right, y up, looks along -z) from an OpenCV rotation (x right, y down, z forward)."""
    return np.concatenate([R_base_cam[:, 0], -R_base_cam[:, 1]])


def lookat_xyaxes(pos: np.ndarray, target: np.ndarray, up=(0, 0, 1.0)) -> np.ndarray:
    z = unit(pos - target)
    x = unit(np.cross(up, z))
    return np.concatenate([x, np.cross(z, x)])


def overview_pose(center: np.ndarray, cam_pos: np.ndarray, angle_deg: float = 65.0, dist: float = 1.15,
                  height: float = 0.65) -> tuple[np.ndarray, np.ndarray]:
    """Side view: the recording camera's horizontal direction from the scene, rotated by angle_deg."""
    h = unit(cam_pos[:2] - center[:2])
    a = np.radians(angle_deg)
    r = np.array([np.cos(a) * h[0] - np.sin(a) * h[1], np.sin(a) * h[0] + np.cos(a) * h[1]])
    return np.array([center[0] + dist * r[0], center[1] + dist * r[1], height]), np.array([*center[:2], 0.08])


# --------------------------------------------------------------------------------------------- MJCF
def object_xml(spec: SceneSpec) -> str:
    o, f = spec.obj, lambda v: " ".join(f"{x:.5f}" for x in v)
    pos = [*spec.object_xy, spec.object_half_height]
    c, s_ = np.cos(spec.object_yaw), np.sin(spec.object_yaw)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    if o.get("lying"):               # cylinder on its side, axis along the body x axis (jaws close across it)
        R = R @ np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    quat = mat_to_quat(R)
    contact = 'condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"'
    if o["shape"] == "box":
        hx, hy, hz = (s / 2 for s in o["size_m"])
        geom = (f'<geom name="object_geom" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" mass="{o["mass_kg"]}" '
                f'material="cube_mat" friction="{o["friction"]} 0.02 0.001" {contact}/>')
    else:
        geom = (f'<geom name="object_geom" type="cylinder" size="{o["radius_m"]:.5f} {o["half_height_m"]:.5f}" '
                f'mass="{o["mass_kg"]}" material="{"tape_mat" if o["radius_m"] > 0.02 else "battery_mat"}" '
                f'friction="{o["friction"]} 0.02 0.001" {contact}/>')
    return (f'<body name="object" pos="{f(pos)}" quat="{f(quat)}"><freejoint name="object_free"/>'
            f'{geom}</body>')


def distractors_xml(spec: SceneSpec, count: int, seed: int = 0) -> str:
    """Small clutter objects placed away from the object, the box and the straight carry path between them."""
    rng = np.random.default_rng(1000 + seed)
    a, b = spec.object_xy, spec.box_xy
    out, placed = [], 0
    while placed < count:
        p = np.array([rng.uniform(0.35, 0.85), rng.uniform(-0.30, 0.30)])
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        if np.linalg.norm(p - (a + t * (b - a))) < 0.12:
            continue
        s = rng.uniform(0.02, 0.035)
        rgba = " ".join(f"{c:.2f}" for c in [*rng.uniform(0.1, 0.9, 3), 1.0])
        out.append(f'<body name="distractor{placed}" pos="{p[0]:.4f} {p[1]:.4f} {s:.4f}"><freejoint/>'
                   f'<geom type="box" size="{s:.4f} {s:.4f} {s:.4f}" mass="0.05" rgba="{rgba}"/></body>')
        placed += 1
    return "".join(out)


def build_xml(spec: SceneSpec, panda_dir: Path, tex_dir: Path, control: str = "position",
              overview_angle: float = 65.0, seed: int = 0) -> str:
    """Scene MJCF. ``control="torque"`` swaps the arm's position actuators for torque motors (for OSC)."""
    xml = panda_xml(panda_dir)
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)  # its qpos excludes the scene's free joints
    xml = xml.replace('<option integrator="implicitfast"/>',
                      '<option integrator="implicitfast" timestep="0.002" cone="elliptic" impratio="10"/>', 1)
    if control == "torque":
        for i, lim in enumerate(TORQUE_LIMITS, start=1):
            xml = re.sub(rf'<general class="panda" name="actuator{i}" joint="joint{i}"[^>]*/>',
                         f'<motor name="actuator{i}" joint="joint{i}" ctrlrange="-{lim} {lim}"/>', xml, flags=re.S)
    cond, f = spec.condition, lambda v: " ".join(f"{x:.5f}" for x in v)
    T = spec.T_base_cam
    cp = T[:3, 3]
    px, py = mujoco_principal(spec.k["cx"], spec.k["cy"], spec.k["width"], spec.k["height"])
    wall_pt = cp + T[:3, 2] * spec.wall_distance
    wall_n = unit(np.array([cp[0] - wall_pt[0], cp[1] - wall_pt[1], 0.0]))
    wall_yaw = float(np.arctan2(wall_n[1], wall_n[0]))
    center = np.array([*(0.5 * (spec.object_xy + spec.box_xy)), 0.0])
    ov_pos, ov_tgt = overview_pose(center, cp, overview_angle)
    floor_rgba = " ".join(map(str, cond.get("floor_rgba", [0.74, 0.76, 0.76, 1.0])))
    floor_tex = ('<texture name="floor_tex" type="2d" builtin="checker" rgb1="0.42 0.36 0.30" rgb2="0.30 0.26 0.22" '
                 'width="512" height="512"/><material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>'
                 if cond.get("floor_texture") == "checker" else
                 f'<material name="floor_mat" rgba="{floor_rgba}" specular="0.1"/>')
    faces = " ".join(f'file{face}="{tex_dir / f"face_{face}.png"}"' for face in FACES)
    assets = ('<texture name="sky" type="skybox" builtin="gradient" rgb1="0.78 0.82 0.86" rgb2="0.45 0.5 0.55" '
              'width="256" height="256"/>'
              f'<texture name="stickers" type="cube" {faces}/>'
              '<material name="cube_mat" texture="stickers" specular="0.2" shininess="0.2"/>'
              '<material name="tape_mat" rgba="0.80 0.72 0.50 1" specular="0.3"/>'
              '<material name="battery_mat" rgba="0.15 0.15 0.15 1" specular="0.6"/>'
              + floor_tex +
              '<material name="wall_mat" rgba="0.66 0.55 0.40 1" specular="0.0"/>'
              '<material name="box_mat" rgba="0.62 0.43 0.24 1" specular="0.05"/>')
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    ls = cond.get("light_scale", 1.0)
    key_dir = f(cond.get("light_dir", [0, 0.3, -1]))
    bx, by, bh = spec.box_size
    k = spec.k
    world = (
        f'<camera name="rs" pos="{f(cp)}" xyaxes="{f(cam_xyaxes(T[:3, :3]))}" resolution="{k["width"]} {k["height"]}" '
        f'focalpixel="{k["fx"]} {k["fy"]}" principalpixel="{px} {py}" sensorsize="0.00384 0.00288"/>'
        f'<camera name="overview" pos="{f(ov_pos)}" xyaxes="{f(lookat_xyaxes(ov_pos, ov_tgt))}" fovy="50"/>'
        f'<light name="key" pos="0.3 -0.6 1.6" dir="{key_dir}" diffuse="{0.75 * ls:.3f} {0.75 * ls:.3f} '
        f'{0.75 * ls:.3f}" castshadow="true"/>'
        f'<light name="fill" pos="1.0 0.5 1.2" dir="-0.6 -0.3 -1" diffuse="{0.35 * ls:.3f} {0.35 * ls:.3f} '
        f'{0.35 * ls:.3f}" castshadow="false"/>'
        '<geom name="floor" type="plane" size="3 3 0.1" material="floor_mat" friction="1 0.005 0.0001"/>'
        f'<geom name="wall" type="box" size="3 0.02 1.5" pos="{f(wall_pt[:2] + wall_n[:2] * 0.02)} 1.4" '
        f'euler="0 0 {wall_yaw + np.pi / 2:.5f}" material="wall_mat" contype="0" conaffinity="0"/>'
        f'<geom name="box" type="box" size="{bx / 2:.5f} {by / 2:.5f} {bh / 2:.5f}" '
        f'pos="{spec.box_xy[0]:.5f} {spec.box_xy[1]:.5f} {bh / 2:.5f}" euler="0 0 {spec.box_yaw:.5f}" '
        'material="box_mat" friction="1 0.005 0.0001"/>'
        + object_xml(spec) + distractors_xml(spec, cond.get("count", 0), seed))
    # appended after the robot so the arm keeps qpos[0:9]; scene free joints follow
    return xml.replace("</worldbody>", world + "</worldbody>", 1)


def write_scene(spec: SceneSpec, panda_dir: Path, control: str = "position", seed: int = 0) -> Path:
    """Write the scene (and its textures) to a fresh temporary directory and return the XML path."""
    out = Path(tempfile.mkdtemp(prefix="recast_scene_"))
    tex = out / "textures"
    write_textures(tex)
    path = out / f"scene_{control}_{seed}.xml"
    path.write_text(build_xml(spec, panda_dir, tex, control, seed=seed))
    return path


__all__ = ["SceneSpec", "load_scene", "build_xml", "write_scene", "TCP_OFFSET"]
