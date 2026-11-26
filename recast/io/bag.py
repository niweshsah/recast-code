"""Export a frame range of a RealSense ROS 2 ``.db3`` recording to the clip format (depth aligned to colour).

    python -m recast.io.bag recording.db3 --first 300 --last 445 --out runs/cube/clip

The bag is opened read-only through sqlite; message decoding needs ``rosbags``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data/compressed"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data/compressedDepth"
COLOR_INFO = "/device_0/sensor_1/Color_0/image/camera_info"
DEPTH_INFO = "/device_0/sensor_0/Depth_0/image/camera_info"
COLOR_TF = "/device_0/sensor_1/Color_0/tf/ref_0"
DEPTH_TF = "/device_0/sensor_0/Depth_0/tf/ref_0"
DEPTH_UNITS = "/device_0/sensor_0/option/Depth_Units/value"
COMPRESSED = "sensor_msgs/msg/CompressedImage"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class CameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple = (0.0,) * 5

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": self.width,
                "height": self.height, "dist": list(self.dist),
                "K": [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]}


# --------------------------------------------------------------------------------------------- geometry
def parse_tf(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse librealsense ``rotation=r0,..,r8;translation=x,y,z`` (column-major rotation)."""
    m = re.fullmatch(r"\s*rotation=([^;]+);translation=(.+?)\s*", text)
    if not m:
        raise ValueError(f"unrecognised tf string: {text!r}")
    rot = np.array([float(v) for v in m.group(1).split(",")]).reshape(3, 3, order="F")
    trans = np.array([float(v) for v in m.group(2).split(",")])
    if trans.shape != (3,):
        raise ValueError(f"bad tf string: {text!r}")
    return rot, trans


def depth_to_color_transform(depth_tf: tuple, color_tf: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Both transforms map the reference frame into a sensor frame; return depth -> colour."""
    (r_d, t_d), (r_c, t_c) = depth_tf, color_tf
    r = r_c @ r_d.T
    return r, t_c - r @ t_d


def align_depth_to_color(depth_raw: np.ndarray, k_depth: CameraInfo, k_color: CameraInfo, rot: np.ndarray,
                         trans: np.ndarray, depth_units: float = 0.001, splat: int = 2) -> np.ndarray:
    """Re-project a 16-bit depth image into the colour camera -> float32 metres (0 = no data).

    Each depth pixel is splatted into a ``splat`` x ``splat`` block (the colour camera has a narrower field of
    view, so one depth pixel covers more than one colour pixel); the nearest point wins.
    """
    ys, xs = np.nonzero(depth_raw)
    z = depth_raw[ys, xs].astype(np.float64) * depth_units
    pts = np.stack([(xs - k_depth.cx) / k_depth.fx * z, (ys - k_depth.cy) / k_depth.fy * z, z], axis=1)
    pts = pts @ rot.T + trans
    zc = pts[:, 2]
    ok = zc > 1e-6
    u = pts[ok, 0] / zc[ok] * k_color.fx + k_color.cx
    v = pts[ok, 1] / zc[ok] * k_color.fy + k_color.cy
    zc = zc[ok]
    out = np.full(k_color.height * k_color.width, np.inf)
    u0 = np.floor(u - (splat - 1) / 2 + 0.5).astype(np.int64)
    v0 = np.floor(v - (splat - 1) / 2 + 0.5).astype(np.int64)
    for dv in range(splat):
        for du in range(splat):
            uu, vv = u0 + du, v0 + dv
            inside = (uu >= 0) & (uu < k_color.width) & (vv >= 0) & (vv < k_color.height)
            np.minimum.at(out, vv[inside] * k_color.width + uu[inside], zc[inside])
    out[~np.isfinite(out)] = 0.0
    return out.reshape(k_color.height, k_color.width).astype(np.float32)


def metres_to_mm_u16(depth_m: np.ndarray) -> np.ndarray:
    mm = np.rint(np.nan_to_num(depth_m, nan=0.0) * 1000.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


def nearest_pairs(color_ts: np.ndarray, depth_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest depth timestamp for each colour timestamp."""
    idx = np.clip(np.searchsorted(depth_ts, color_ts), 1, len(depth_ts) - 1)
    left, right = depth_ts[idx - 1], depth_ts[idx]
    return np.where(np.abs(color_ts - left) <= np.abs(right - color_ts), idx - 1, idx)


# --------------------------------------------------------------------------------------------- bag access
class Bag:
    """Read-only access to a ROS 2 sqlite bag."""

    def __init__(self, path: Path):
        from rosbags.typesys import Stores, get_typestore

        self.path = Path(path)
        self.conn = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        self.ts = get_typestore(Stores.ROS2_HUMBLE)

    def rows(self, topic: str) -> list[tuple[int, int]]:
        """[(message_rowid, timestamp_ns)] ordered by time."""
        return self.conn.execute("SELECT m.id, m.timestamp FROM messages m JOIN topics t ON t.id=m.topic_id "
                                 "WHERE t.name=? ORDER BY m.timestamp", (topic,)).fetchall()

    def first(self, topic: str):
        row = self.conn.execute("SELECT t.type, m.data FROM messages m JOIN topics t ON t.id=m.topic_id "
                                "WHERE t.name=? ORDER BY m.timestamp LIMIT 1", (topic,)).fetchone()
        return None if row is None else self.ts.deserialize_cdr(row[1], row[0])

    def message(self, rowid: int, msg_type: str = COMPRESSED):
        (data,) = self.conn.execute("SELECT data FROM messages WHERE id=?", (rowid,)).fetchone()
        return self.ts.deserialize_cdr(data, msg_type)

    def camera_info(self, topic: str) -> CameraInfo:
        msg = self.first(topic)
        k = np.asarray(msg.k, float)
        return CameraInfo(float(k[0]), float(k[4]), float(k[2]), float(k[5]), int(msg.width), int(msg.height),
                          tuple(float(d) for d in msg.d))

    def close(self):
        self.conn.close()


def decode_color(msg) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode colour frame")
    return img


def decode_depth(msg) -> np.ndarray:
    """compressedDepth = 12-byte ConfigHeader followed by a 16-bit PNG."""
    buf = bytes(msg.data)
    start = buf.find(PNG_MAGIC)
    if start < 0:
        raise ValueError("no PNG payload in compressedDepth message")
    img = cv2.imdecode(np.frombuffer(buf[start:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16:
        raise ValueError("unexpected depth payload")
    return img


def export_clip(bag_path: Path, first: int, last: int, out: Path) -> dict:
    """Write colour, aligned depth (mm), ``intrinsics.json`` and ``timestamps.csv`` for colour frames first..last."""
    out = Path(out)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    bag = Bag(bag_path)
    c_rows, d_rows = bag.rows(COLOR_TOPIC), bag.rows(DEPTH_TOPIC)
    k_c, k_d = bag.camera_info(COLOR_INFO), bag.camera_info(DEPTH_INFO)
    rot, trans = depth_to_color_transform(parse_tf(bag.first(DEPTH_TF).data), parse_tf(bag.first(COLOR_TF).data))
    units_msg = bag.first(DEPTH_UNITS)
    units = float(units_msg.data) if units_msg is not None else 0.001
    c_t = np.array([r[1] for r in c_rows], np.int64)
    d_t = np.array([r[1] for r in d_rows], np.int64)
    pair = nearest_pairs(c_t, d_t)
    first, last = max(first, 0), min(last, len(c_rows) - 1)
    skews = []
    with (out / "timestamps.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_index", "bag_color_index", "color_ts_ns", "depth_ts_ns", "skew_ms", "t_rel_s"])
        for n, i in enumerate(range(first, last + 1)):
            color = decode_color(bag.message(c_rows[i][0]))
            depth = decode_depth(bag.message(d_rows[pair[i]][0]))
            aligned = align_depth_to_color(depth, k_d, k_c, rot, trans, units)
            cv2.imwrite(str(out / "rgb" / f"{n:05d}.png"), color)
            cv2.imwrite(str(out / "depth" / f"{n:05d}.png"), metres_to_mm_u16(aligned))
            skew = (int(c_t[i]) - int(d_t[pair[i]])) / 1e6
            skews.append(abs(skew))
            w.writerow([n, i, int(c_t[i]), int(d_t[pair[i]]), f"{skew:.2f}", f"{(c_t[i] - c_t[first]) / 1e9:.4f}"])
    meta = {"color": k_c.as_dict(), "depth_sensor": k_d.as_dict(),
            "depth_to_color": {"R": rot.tolist(), "t_m": trans.tolist()},
            "aligned_depth": {"frame": "color", "dtype": "uint16", "units": "mm", "invalid": 0},
            "source": {"bag": bag.path.name, "color_frames": [first, last], "n_frames": last - first + 1,
                       "fps": float(1e9 / np.median(np.diff(c_t[first:last + 1]))),
                       "max_abs_skew_ms": float(max(skews)), "median_abs_skew_ms": float(np.median(skews))}}
    (out / "intrinsics.json").write_text(json.dumps(meta, indent=2))
    bag.close()
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--first", type=int, required=True, help="first colour frame (inclusive)")
    ap.add_argument("--last", type=int, required=True, help="last colour frame (inclusive)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = export_clip(a.bag, a.first, a.last, a.out)
    s = meta["source"]
    print(f"{s['bag']}: {s['n_frames']} frames at {s['fps']:.1f} fps, depth skew median {s['median_abs_skew_ms']:.1f} ms")


if __name__ == "__main__":
    main()
