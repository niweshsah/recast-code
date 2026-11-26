"""Hand reconstruction: WiLoR MANO mesh per frame, placed in metric 3D with the sensor depth (paper eq. 1).

WiLoR predicts the hand with its own virtual pinhole camera (focal = 5000/256 * max(W, H), principal point at the
image centre). For metric 3D the hand translation is re-solved for the real RealSense intrinsics so that the
projection of the predicted vertices is unchanged (linear least squares), then refined by rigid ICP of the
camera-visible mesh vertices to the depth points inside WiLoR's own rendered hand silhouette.

Output ``<run>/hands/hands.npz`` (one hand per frame, the most confident ICP-aligned one):
    present[T] bool, verts[T, 778, 3], joints[T, 21, 3] (camera frame, m), kp2d[T, 21, 2],
    icp_rmse[T] (m, the residual r_t used by the anchoring weight), conf[T], is_right[T]
plus ``overlay/`` (mesh), ``keypoints/`` (2D skeleton) and ``on_depth/`` (ICP-aligned vertices on depth) images.

    python -m recast.perception.hands runs/cube --wilor third_party/WiLoR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from recast.config import load_config
from recast.geometry import backproject, depth_window, project, solve_translation, to_open3d, visible_vertices
from recast.io.clip import Clip

HAND_BGR = (180, 119, 31)               # matplotlib tab:blue, the hand colour in every figure
HAND_RGB01 = (0.122, 0.467, 0.706)
N_VERTS, N_JOINTS = 778, 21
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
THUMB_JOINTS = (1, 2, 3, 4)
FINGER_JOINTS = tuple(range(5, 21))


def depth_gray(depth_mm: np.ndarray, near: float, far: float) -> np.ndarray:
    """Depth visualisation: near = bright, far = dark, no data = black."""
    v = np.clip((far - depth_mm.astype(np.float32)) / (far - near), 0, 1)
    g = (40 + 215 * v).astype(np.uint8)
    g[depth_mm == 0] = 0
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class WilorHandTracker:
    """Detector (YOLO) + WiLoR regressor + renderer from a pinned WiLoR checkout."""

    def __init__(self, wilor_root: Path, device: str | None = None, conf: float = 0.3):
        import torch

        self.root = Path(wilor_root).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.conf = conf
        sys.path.insert(0, str(self.root))
        cwd = os.getcwd()
        os.chdir(self.root)  # WiLoR resolves mano_data/ relative to its checkout
        try:
            from ultralytics import YOLO
            from wilor.models import load_wilor
            from wilor.utils.renderer import Renderer

            ckpt = self.root / "pretrained_models"
            self.model, self.cfg = load_wilor(checkpoint_path=str(ckpt / "wilor_final.ckpt"),
                                              cfg_path=str(ckpt / "model_config.yaml"))
            self.model = self.model.to(self.device).eval()
            self.detector = YOLO(str(ckpt / "detector.pt")).to(self.device)
            self.renderer = Renderer(self.cfg, faces=self.model.mano.faces)
        finally:
            os.chdir(cwd)

    def __call__(self, img_bgr: np.ndarray) -> list[dict]:
        """Per detected hand: vertices/joints relative to WiLoR's camera, its translation and focal, 2D keypoints,
        and the rendered silhouette."""
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        h, w = img_bgr.shape[:2]
        det = self.detector(img_bgr, conf=self.conf, verbose=False)[0]
        if len(det) == 0:
            return []
        boxes = det.boxes.data.cpu().numpy()[:, :4]
        confs = det.boxes.conf.cpu().numpy()
        right = det.boxes.cls.cpu().numpy()
        ds = ViTDetDataset(self.cfg, img_bgr, boxes, right, rescale_factor=2.0)
        batch = recursive_to(next(iter(torch.utils.data.DataLoader(ds, batch_size=len(boxes), shuffle=False))),
                             self.device)
        with torch.no_grad():
            out = self.model(batch)
        mult = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = mult * pred_cam[:, 1]
        img_size = batch["img_size"].float()
        focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size,
                                 focal).detach().cpu().numpy()
        focal = float(focal)
        k_wilor = {"fx": focal, "fy": focal, "cx": w / 2, "cy": h / 2}
        hands = []
        for n in range(len(boxes)):
            is_r = float(batch["right"][n].cpu())
            verts = out["pred_vertices"][n].detach().cpu().numpy().copy()
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().copy()
            verts[:, 0] *= 2 * is_r - 1
            joints[:, 0] *= 2 * is_r - 1
            rgba = self.renderer.render_rgba_multiple([verts], cam_t=[cam_t[n]], render_res=img_size[n],
                                                      is_right=[is_r], mesh_base_color=HAND_RGB01,
                                                      scene_bg_color=(1, 1, 1), focal_length=focal)
            hands.append({"bbox": boxes[n], "conf": float(confs[n]), "is_right": bool(is_r), "verts_rel": verts,
                          "joints_rel": joints, "cam_t": cam_t[n], "focal": focal,
                          "uv_verts": project(verts + cam_t[n], k_wilor), "kp2d": project(joints + cam_t[n], k_wilor),
                          "silhouette": rgba[:, :, 3] > 0.5})
        return hands

    def overlay(self, img_bgr: np.ndarray, hands: list[dict]) -> np.ndarray:
        """All hands rendered by WiLoR and alpha-blended over the frame."""
        if not hands:
            return img_bgr.copy()
        h, w = img_bgr.shape[:2]
        rgba = self.renderer.render_rgba_multiple([d["verts_rel"] for d in hands], cam_t=[d["cam_t"] for d in hands],
                                                  render_res=np.array([w, h], np.float32),
                                                  is_right=[float(d["is_right"]) for d in hands],
                                                  mesh_base_color=HAND_RGB01, scene_bg_color=(1, 1, 1),
                                                  focal_length=hands[0]["focal"])
        base = img_bgr.astype(np.float32)[:, :, ::-1] / 255.0
        blend = base * (1 - rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
        return (255 * blend[:, :, ::-1]).clip(0, 255).astype(np.uint8)


def register_to_depth(hand: dict, faces: np.ndarray, depth_mm: np.ndarray, k: dict, min_points: int = 200,
                      max_dist: float = 0.02, min_fitness: float = 0.3) -> dict | None:
    """Metric pose of one WiLoR hand: least-squares translation for the real intrinsics, then rigid ICP of the
    camera-visible vertices to the depth points inside the hand silhouette. None when ICP is not trustworthy."""
    import open3d as o3d
    import trimesh

    t_real = solve_translation(hand["verts_rel"], hand["uv_verts"], k)
    pts = depth_window(backproject(depth_mm, hand["silhouette"], k))
    if len(pts) < min_points:
        return None
    mesh = trimesh.Trimesh(hand["verts_rel"] + t_real, faces, process=False)
    vis, _ = visible_vertices(mesh, np.zeros(3))
    init = np.eye(4)
    init[:3, 3] = np.median(pts, axis=0) - np.median(vis, axis=0)
    res = o3d.pipelines.registration.registration_icp(
        to_open3d(vis), to_open3d(pts), max_dist, init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    if res.fitness <= min_fitness:
        return None
    T = np.asarray(res.transformation)
    return {"verts": (hand["verts_rel"] + t_real) @ T[:3, :3].T + T[:3, 3],
            "joints": (hand["joints_rel"] + t_real) @ T[:3, :3].T + T[:3, 3],
            "icp_rmse": float(res.inlier_rmse), "icp_fitness": float(res.fitness)}


def reconstruct(run: Path, cfg: dict, wilor_root: Path) -> dict:
    clip = Clip(run / "clip")
    hc = cfg["hands"]
    tracker = WilorHandTracker(wilor_root, conf=hc["detector_conf"])
    faces = tracker.model.mano.faces
    T = len(clip)
    out = {"present": np.zeros(T, bool), "verts": np.zeros((T, N_VERTS, 3), np.float32),
           "joints": np.zeros((T, N_JOINTS, 3), np.float32), "kp2d": np.zeros((T, N_JOINTS, 2), np.float32),
           "icp_rmse": np.full(T, np.nan, np.float32), "conf": np.zeros(T, np.float32),
           "is_right": np.zeros(T, bool)}
    vis_dirs = {d: run / "hands" / d for d in ("overlay", "keypoints", "on_depth")}
    for d in vis_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    depths = [clip.depth(i) for i in range(T)]
    valid = np.concatenate([d[d > 0] for d in depths])
    near, far = np.percentile(valid, [1, 99])
    for i in range(T):
        img, depth = clip.rgb(i), depths[i]
        hands = tracker(img)
        kp_img, dep_img = img.copy(), depth_gray(depth, near, far)
        best = None
        for hand in sorted(hands, key=lambda d: -d["conf"]):
            for a, b in SKELETON:
                cv2.line(kp_img, tuple(hand["kp2d"][a].astype(int)), tuple(hand["kp2d"][b].astype(int)), HAND_BGR, 2)
            for p in hand["kp2d"]:
                cv2.circle(kp_img, tuple(p.astype(int)), 3, (255, 255, 255), -1)
            if best is None:
                reg = register_to_depth(hand, faces, depth, clip.k, hc["min_depth_points"], hc["icp_max_dist_m"],
                                        hc["icp_min_fitness"])
                if reg is not None:
                    best = (hand, reg)
        if best is not None:
            hand, reg = best
            out["present"][i] = True
            out["verts"][i], out["joints"][i] = reg["verts"], reg["joints"]
            out["kp2d"][i], out["icp_rmse"][i] = hand["kp2d"], reg["icp_rmse"]
            out["conf"][i], out["is_right"][i] = hand["conf"], hand["is_right"]
            uv = project(reg["verts"], clip.k).round().astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            for u, v in uv[ok]:
                cv2.circle(dep_img, (int(u), int(v)), 1, HAND_BGR, -1)
        name = f"{clip.names[i]}.png"
        cv2.imwrite(str(vis_dirs["overlay"] / name), tracker.overlay(img, hands))
        cv2.imwrite(str(vis_dirs["keypoints"] / name), kp_img)
        cv2.imwrite(str(vis_dirs["on_depth"] / name), dep_img)
    np.savez_compressed(run / "hands" / "hands.npz", **out)
    rm = out["icp_rmse"][out["present"]]
    summary = {"frames": T, "frames_with_hand": int(out["present"].sum()),
               "icp_rmse_mm_median": float(1000 * np.median(rm)) if len(rm) else None}
    (run / "hands" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_hands(run: Path) -> dict[str, np.ndarray]:
    with np.load(run / "hands" / "hands.npz") as f:
        return {k: f[k] for k in f.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--wilor", type=Path, default=Path("third_party/WiLoR"))
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    print(json.dumps(reconstruct(a.run, load_config(a.task), a.wilor), indent=2))


if __name__ == "__main__":
    main()
