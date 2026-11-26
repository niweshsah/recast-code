"""(i) Object-state tracking: 3D object keypoints K^o_t and the policy input s^o_t (paper eq. 2).

1. On the first frame with a valid object mask, M points are chosen by farthest-point sampling over the object's
   depth points (spread over the visible surface, not clustered on texture).
2. CoTracker follows their pixels through the clip.
3. A track counts as observed when CoTracker marks it visible, it lies inside the segmentation mask, and depth is
   valid there. Observed tracks are lifted to 3D and moved into the world (robot-base) frame W.
4. Unobserved points are carried from the previous frame by the rigid transform that Kabsch alignment estimates
   from the observed points.
5. Without an object mesh, the object pose T^W_{O,t} is the Kabsch alignment of K_t to K_0
   (FoundationPose can replace this for rigid objects with a mesh).

Policy input:  s^o_t = (T^W_{E,t})^{-1} K^o_t   (keypoints in the current end-effector frame).

    python -m recast.perception.keypoints runs/cube --num-points 16
Output ``<run>/keypoints/keypoints.npz``: uv[T, M, 2], observed[T, M], K_world[T, M, 3], T_WO[T, 4, 4].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recast.geometry import (
    backproject,
    farthest_point_sampling,
    invert_T,
    kabsch,
    lift_pixels,
    project,
    transform_points,
)
from recast.io.clip import Clip, read_mask

MIN_RIGID_POINTS = 3


def init_queries(depth_mm: np.ndarray, mask: np.ndarray, k: dict, m: int, seed: int = 0) -> np.ndarray:
    """(m, 2) pixel positions of farthest-point-sampled object depth points."""
    pts = backproject(depth_mm, mask, k)
    if len(pts) < m:
        raise ValueError(f"object has only {len(pts)} depth points, need at least {m}")
    return project(pts[farthest_point_sampling(pts, m, seed)], k)


def track_cotracker(frames_rgb: np.ndarray, queries_uv: np.ndarray, t0: int, model: str = "cotracker3_offline",
                    device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CoTracker (torch.hub, facebookresearch/co-tracker). frames_rgb: (T, H, W, 3) uint8 RGB.
    Returns uv (T, M, 2) and visibility (T, M)."""
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tracker = torch.hub.load("facebookresearch/co-tracker", model).to(dev)
    video = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)[None].float().to(dev)
    q = np.concatenate([np.full((len(queries_uv), 1), t0), queries_uv], axis=1)
    queries = torch.from_numpy(q)[None].float().to(dev)
    with torch.no_grad():
        tracks, vis = tracker(video, queries=queries, backward_tracking=True)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5


def lift_tracks(uv: np.ndarray, visible: np.ndarray, depths: list[np.ndarray], masks: list[np.ndarray], k: dict,
                T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed 3D points (world frame) and the observation mask, per frame."""
    T, M = visible.shape
    pts_w, observed = np.zeros((T, M, 3)), np.zeros((T, M), bool)
    for t in range(T):
        pts, ok = lift_pixels(uv[t], depths[t], k)
        u, v = np.round(uv[t, :, 0]).astype(int), np.round(uv[t, :, 1]).astype(int)
        h, w = masks[t].shape
        inside = np.zeros(M, bool)
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        inside[inb] = masks[t][v[inb], u[inb]]
        observed[t] = visible[t] & ok & inside
        pts_w[t] = transform_points(T_wc, pts)
    return pts_w, observed


def propagate_rigid(pts_w: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill unobserved keypoints by carrying the previous frame with the Kabsch transform of the observed ones.

    Frames before the first fully usable frame are back-filled the same way, running backwards in time."""
    T, M, _ = pts_w.shape
    counts = observed.sum(1)
    start = next((t for t in range(T) if counts[t] == M), None)
    if start is None:
        start = int(counts.argmax())
        if counts[start] < MIN_RIGID_POINTS:
            raise ValueError("no frame observes enough keypoints to anchor the object")
    K = np.full((T, M, 3), np.nan)
    K[start] = pts_w[start]
    if counts[start] < M:  # keep the unobserved ones at their centroid offset until seen
        K[start, ~observed[start]] = pts_w[start, observed[start]].mean(0)
    for order in (range(start + 1, T), range(start - 1, -1, -1)):
        prev = start
        for t in order:
            obs = observed[t]
            K[t] = K[prev]
            if obs.sum() >= MIN_RIGID_POINTS:
                T_rel = kabsch(K[prev][obs], pts_w[t][obs])
                K[t] = transform_points(T_rel, K[prev])
                K[t][obs] = pts_w[t][obs]
            prev = t
    return K


def object_poses(K: np.ndarray) -> np.ndarray:
    """T^W_{O,t} as the rigid alignment of K_t to K_0 (object frame = first-frame keypoint frame at their centroid)."""
    c0 = K[0].mean(0)
    out = np.zeros((len(K), 4, 4))
    for t in range(len(K)):
        T_rel = kabsch(K[0], K[t])
        out[t] = T_rel
        out[t][:3, 3] = T_rel[:3, :3] @ c0 + T_rel[:3, 3]  # object origin at the initial centroid
    return out


def object_state(K_t: np.ndarray, T_WE_t: np.ndarray) -> np.ndarray:
    """Paper eq. (2): keypoints in the end-effector frame, s^o_t = (T^W_{E,t})^{-1} K^o_t  -> (M, 3)."""
    return transform_points(invert_T(T_WE_t), K_t)


def track_object(run: Path, num_points: int = 16, T_wc: np.ndarray | None = None) -> dict:
    import cv2

    clip = Clip(run / "clip")
    masks = [read_mask(run / "masks" / "object" / f"{n}.png") for n in clip.names]
    depths = [clip.depth(i) for i in range(len(clip))]
    t0 = next(t for t in range(len(clip)) if (masks[t] & (depths[t] > 0)).sum() >= 4 * num_points)
    queries = init_queries(depths[t0], masks[t0], clip.k, num_points)
    frames = np.stack([cv2.cvtColor(clip.rgb(i), cv2.COLOR_BGR2RGB) for i in range(len(clip))])
    uv, vis = track_cotracker(frames, queries, t0)
    T_wc = np.eye(4) if T_wc is None else T_wc
    pts_w, observed = lift_tracks(uv, vis, depths, masks, clip.k, T_wc)
    K = propagate_rigid(pts_w, observed)
    poses = object_poses(K)
    (run / "keypoints").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "keypoints" / "keypoints.npz", uv=uv, observed=observed, K_world=K, T_WO=poses,
                        T_wc=T_wc, init_frame=t0)
    summary = {"num_points": num_points, "init_frame": int(t0), "observed_fraction": float(observed.mean())}
    (run / "keypoints" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--num-points", type=int, default=16)
    ap.add_argument("--extrinsics", type=Path, help="JSON with a 4x4 'T_world_cam' (camera-to-robot calibration)")
    a = ap.parse_args()
    T_wc = np.array(json.loads(a.extrinsics.read_text())["T_world_cam"]) if a.extrinsics else None
    print(json.dumps(track_object(a.run, a.num_points, T_wc), indent=2))


if __name__ == "__main__":
    main()
