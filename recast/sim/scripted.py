"""Scripted pick-and-place in the test scene (paper Fig. 9): top-down grasp, Cartesian waypoints tracked by IK,
contact physics. Not a learned policy. Failure variants reproduce the typical errors of hand-only labels:

    success   unchanged trajectory
    drop      gripper opens part-way through the carry (a fingertip-spacing false release)
    miss      grasp offset perpendicular to the closing axis (fingertip vs contact geometry); jaws close beside it

    MUJOCO_GL=egl python -m recast.sim.scripted --task cube --variant drop --video outputs/sim_drop.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recast.config import load_config
from recast.sim.env import CLOSED, OPEN, TIMESTEP, PandaSim
from recast.sim.scene import SceneSpec, load_scene

FPS = 30
MIN_GRASP_Z = 0.015      # Panda finger pads reach ~13 mm below the TCP: keep them off the floor


def topdown_R(yaw: float) -> np.ndarray:
    """TCP rotation: approach (z) straight down, closing axis (y) at ``yaw`` in the floor plane."""
    y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([np.cross(y, z), y, z], axis=1)


def plan_waypoints(spec: SceneSpec) -> dict[str, np.ndarray]:
    half, top = spec.object_half_height, spec.box_top
    c = np.array([*spec.object_xy, half])
    b = np.array([*spec.box_xy, 0.0])
    up = lambda p, dz: p + np.array([0.0, 0.0, dz])
    grasp = np.array([c[0], c[1], max(half + 0.004, MIN_GRASP_Z)])
    return {"pregrasp": up(c, 0.12), "grasp": grasp, "lift": up(c, 0.09),
            "above_box": np.array([b[0], b[1], top + half + 0.06]),
            "place": np.array([b[0], b[1], top + half + 0.004]),
            "retreat": np.array([b[0], b[1], top + 2 * half + 0.12])}


def is_on_top(obj_pos: np.ndarray, spec: SceneSpec, xy_tol: float = 0.03, z_tol: float = 0.012) -> bool:
    """Object resting on the box: centred within xy_tol and one half-height above the box top."""
    return bool(np.linalg.norm(obj_pos[:2] - spec.box_xy) < xy_tol
                and abs(obj_pos[2] - (spec.box_top + spec.object_half_height)) < z_tol)


def run(spec: SceneSpec, panda_dir: Path, variant: str = "success", camera: str = "overview",
        miss_offset: float = 0.039, drop_fraction: float = 0.35, record: bool = True) -> dict:
    sim = PandaSim(spec, panda_dir, control="position", render_size=(720, 640))
    frames: list[np.ndarray] = []
    every = round(1 / FPS / TIMESTEP)
    if record:
        sim.frame_hooks.append(lambda s: frames.append(s.render(camera)) if s.steps % every == 0 else None)
    R = topdown_R(spec.object_yaw)
    wp = plan_waypoints(spec)
    if variant == "miss":
        for key in ("pregrasp", "grasp", "lift"):
            wp[key] = wp[key] + R[:, 0] * miss_offset
    sim.settle(0.6)
    q_pre, *_ = sim.ik.solve(wp["pregrasp"], R)
    sim.joint_move(q_pre, 2.0)
    sim.settle(0.3)
    sim.move_to(wp["grasp"], R, 1.2)
    sim.settle(0.2)
    sim.set_gripper(CLOSED, 0.8)
    sim.settle(0.3)
    z0 = sim.object_pose()[2, 3]
    sim.move_to(wp["lift"], R, 1.0)
    lifted = float(sim.object_pose()[2, 3] - z0)
    if variant == "drop":
        sim.move_to(wp["lift"] + (wp["above_box"] - wp["lift"]) * drop_fraction, R, 2.0 * drop_fraction)
        sim.set_gripper(OPEN, 0.25)
        sim.move_to(wp["above_box"], R, 2.0 * (1 - drop_fraction))
    else:
        sim.move_to(wp["above_box"], R, 2.0)
    sim.move_to(wp["place"], R, 1.2)
    sim.settle(0.3)
    sim.set_gripper(OPEN, 0.6)
    sim.settle(0.3)
    sim.move_to(wp["retreat"], R, 1.0)
    sim.settle(1.5)
    final = sim.object_pose()[:3, 3]
    sim.close()
    return {"variant": variant, "success": is_on_top(final, spec), "final_object": final.round(4).tolist(),
            "lifted_mm": round(1000 * lifted), "frames": frames}


def main() -> None:
    import imageio.v2 as imageio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="cube")
    ap.add_argument("--variant", choices=["success", "drop", "miss"], default="success")
    ap.add_argument("--camera", choices=["overview", "rs"], default="overview")
    ap.add_argument("--menagerie", type=Path, default=Path("third_party/mujoco_menagerie"))
    ap.add_argument("--video", type=Path)
    a = ap.parse_args()
    spec = load_scene(load_config(a.task)["task"])
    res = run(spec, a.menagerie / "franka_emika_panda", a.variant, a.camera, record=a.video is not None)
    frames = res.pop("frames")
    if a.video:
        a.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(a.video, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
