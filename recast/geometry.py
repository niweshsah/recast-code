"""Camera and rigid-body geometry shared by every stage (numpy only unless noted)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

Intrinsics = Mapping[str, float]  # {"fx", "fy", "cx", "cy", "width", "height"}


# --------------------------------------------------------------------------------------------- camera
def backproject(depth_mm: np.ndarray, mask: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Pixels inside ``mask`` with valid depth -> (N, 3) camera-frame points in metres."""
    ys, xs = np.nonzero(mask & (depth_mm > 0))
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    return np.stack([(xs - k["cx"]) / k["fx"] * z, (ys - k["cy"]) / k["fy"] * z, z], axis=1)


def project(points: np.ndarray, k: Intrinsics) -> np.ndarray:
    """(N, 3) camera-frame points -> (N, 2) pixel coordinates."""
    z = points[:, 2:3]
    return np.concatenate([k["fx"] * points[:, :1] / z + k["cx"], k["fy"] * points[:, 1:2] / z + k["cy"]], axis=1)


def lift_pixels(uv: np.ndarray, depth_mm: np.ndarray, k: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Depth lookup at (N, 2) pixel positions -> (N, 3) points and a validity mask (inside image, depth > 0)."""
    h, w = depth_mm.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(uv))
    z[ok] = depth_mm[v[ok], u[ok]] / 1000.0
    ok &= z > 0
    pts = np.stack([(uv[:, 0] - k["cx"]) / k["fx"] * z, (uv[:, 1] - k["cy"]) / k["fy"] * z, z], axis=1)
    return pts, ok


def solve_translation(verts: np.ndarray, uv: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Translation t with project(verts + t; K) ~= uv in the least-squares sense (rotation and shape fixed).

    fx (X + tx) = (u - cx)(Z + tz)  ->  fx tx - (u - cx) tz = (u - cx) Z - fx X   (same for y)
    """
    du, dv = uv[:, 0] - k["cx"], uv[:, 1] - k["cy"]
    n = len(verts)
    a, b = np.zeros((2 * n, 3)), np.zeros(2 * n)
    a[:n, 0], a[:n, 2], b[:n] = k["fx"], -du, du * verts[:, 2] - k["fx"] * verts[:, 0]
    a[n:, 1], a[n:, 2], b[n:] = k["fy"], -dv, dv * verts[:, 2] - k["fy"] * verts[:, 1]
    t, *_ = np.linalg.lstsq(a, b, rcond=None)
    return t


def depth_window(points: np.ndarray, half_width_m: float = 0.10) -> np.ndarray:
    """Keep points within +-half_width of the median depth (drops background seen through a silhouette)."""
    if len(points) == 0:
        return points
    return points[np.abs(points[:, 2] - np.median(points[:, 2])) <= half_width_m]


# --------------------------------------------------------------------------------------------- rigid transforms
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero-length vector")
    return v / n


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares rigid transform T (4x4) with T @ src ~= dst for corresponding (N, 3) point sets."""
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    cs, cd = w @ src, w @ dst
    H = (src - cs).T @ ((dst - cd) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return make_T(R, cd - R @ cs)


def farthest_point_sampling(points: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """Indices of ``m`` points chosen by iterative farthest-point sampling."""
    n = len(points)
    if n == 0:
        return np.zeros(0, int)
    m = min(m, n)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(n))]
    dist = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(m - 1):
        idx.append(int(dist.argmax()))
        dist = np.minimum(dist, np.linalg.norm(points - points[idx[-1]], axis=1))
    return np.array(idx)


# --------------------------------------------------------------------------------------------- rotations
def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two rotations (t=0 -> R0, t=1 -> R1)."""
    q0, q1 = mat_to_quat(R0), mat_to_quat(R1)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return quat_to_mat(q)


def rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1.0, 1.0))))


def rotation_mean(Rs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Chordal L2 mean of rotations (projection of the weighted average onto SO(3))."""
    w = np.ones(len(Rs)) if weights is None else np.asarray(weights, float)
    U, _, Vt = np.linalg.svd(np.tensordot(w / w.sum(), Rs, 1))
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


# --------------------------------------------------------------------------------------------- meshes
def visible_vertices(mesh, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mesh vertices visible from ``origin`` (first ray hit). Adapted from Phantom (MIT), see NOTICE."""
    import trimesh

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    pts = mesh.vertices
    dirs = (pts - origin) / np.linalg.norm(pts - origin, axis=1, keepdims=True)
    tri = intersector.intersects_first(np.tile(origin, (len(pts), 1)), dirs)
    idx = np.unique(mesh.faces[tri[tri >= 0]])
    return np.asarray(pts[idx], np.float32), idx


def to_open3d(points: np.ndarray):
    """(N, 3) array -> open3d PointCloud without non-finite points. Adapted from Phantom (MIT), see NOTICE."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    pcd.remove_non_finite_points()
    return pcd
