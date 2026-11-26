"""Hand-object interaction signals per frame, the observations of the phase model.

    d_t      minimum distance between the hand mesh and the object point cloud (m)
    kappa_t  contact measure in [0, 1]: mesh proximity, optionally fused with a pretrained hand-object contact
             detector score (Shan et al., 100DOH); the detector runs outside this package and its per-frame
             scores are read from ``<run>/contact/detector_scores.json`` when present
    v_t      object speed from the keypoint centroid (m/s)
    dd_t     time derivative of d_t (m/s)

    python -m recast.perception.contact runs/cube
Output ``<run>/contact/signals.npz`` with arrays d, kappa, v, dd, n_contact (NaN where no hand / object).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recast.geometry import backproject, depth_window
from recast.io.clip import Clip, read_mask


def nearest_distances(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from every point of ``a`` to its nearest neighbour in ``b``."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    try:
        from scipy.spatial import cKDTree

        return cKDTree(b).query(a, k=1)[0]
    except ImportError:
        out = np.empty(len(a))
        for s in range(0, len(a), chunk):
            d2 = ((a[s:s + chunk, None, :] - b[None]) ** 2).sum(-1)
            out[s:s + chunk] = np.sqrt(d2.min(1))
        return out


def proximity_contact(verts: np.ndarray, obj_pts: np.ndarray, eps: float, saturation: int = 40) -> tuple:
    """(d_t, kappa_t, number of vertices within eps). kappa saturates at ``saturation`` contact vertices."""
    dist = nearest_distances(verts, obj_pts)
    n = int((dist <= eps).sum())
    return float(dist.min()), min(1.0, n / saturation), n


def object_points(depth_mm: np.ndarray, mask: np.ndarray, k: dict) -> np.ndarray:
    return depth_window(backproject(depth_mm, mask, k), 0.10)


def compute_signals(run: Path, eps: float = 0.01, detector_weight: float = 0.5) -> dict:
    from recast.perception.hands import load_hands

    clip = Clip(run / "clip")
    hands = load_hands(run)
    T = len(clip)
    d, kappa, n_c = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, int)
    for t in range(T):
        if not hands["present"][t]:
            continue
        pts = object_points(clip.depth(t), read_mask(run / "masks" / "object" / f"{clip.names[t]}.png"), clip.k)
        if len(pts) == 0:
            continue
        d[t], kappa[t], n_c[t] = proximity_contact(hands["verts"][t], pts, eps)
    det_file = run / "contact" / "detector_scores.json"
    if det_file.exists():
        det = np.array(json.loads(det_file.read_text())["scores"], float)
        kappa = np.where(np.isnan(kappa), np.nan, (1 - detector_weight) * kappa + detector_weight * det)
    kp_file = run / "keypoints" / "keypoints.npz"
    v = np.full(T, np.nan)
    if kp_file.exists():
        with np.load(kp_file) as f:
            centroid = f["K_world"].mean(1)
        v[1:] = np.linalg.norm(np.diff(centroid, axis=0), axis=1) * clip.fps
        v[0] = v[1]
    dd = np.gradient(np.nan_to_num(d, nan=np.nanmax(d) if np.isfinite(d).any() else 0.0)) * clip.fps
    (run / "contact").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "contact" / "signals.npz", d=d, kappa=kappa, v=v, dd=dd, n_contact=n_c)
    return {"frames": T, "frames_in_contact": int((n_c > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--eps", type=float, default=0.01)
    a = ap.parse_args()
    print(json.dumps(compute_signals(a.run, a.eps), indent=2))


if __name__ == "__main__":
    main()
