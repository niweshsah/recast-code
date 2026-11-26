"""On-disk clip format shared by all stages.

    <clip>/rgb/00000.png|jpg     colour frames (BGR on disk)
    <clip>/depth/00000.png       uint16 millimetres aligned to colour, 0 = no return
    <clip>/intrinsics.json       {"color": {fx, fy, cx, cy, width, height, ...}, "source": {...}}
    <clip>/timestamps.csv        per-frame bag index and time

Stage outputs live next to it in the run directory (masks/, hands/, keypoints/, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Clip:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        if not (self.root / "intrinsics.json").exists():
            raise FileNotFoundError(f"{self.root} is not a clip (no intrinsics.json)")

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.root / "intrinsics.json").read_text())

    @property
    def k(self) -> dict:
        return self.meta["color"]

    @property
    def fps(self) -> float:
        return float(self.meta.get("source", {}).get("fps", 15.0))

    @cached_property
    def rgb_paths(self) -> list[Path]:
        paths = sorted(p for p in (self.root / "rgb").iterdir() if p.suffix in {".png", ".jpg"})
        if not paths:
            raise FileNotFoundError(f"no frames in {self.root / 'rgb'}")
        return paths

    @property
    def names(self) -> list[str]:
        """Frame stems ('00000', ...); stage outputs use ``<stem>.png``."""
        return [p.stem for p in self.rgb_paths]

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def rgb(self, i: int) -> np.ndarray:
        """BGR uint8."""
        return cv2.imread(str(self.rgb_paths[i]), cv2.IMREAD_COLOR)

    def depth(self, i: int) -> np.ndarray:
        """uint16 millimetres."""
        d = cv2.imread(str(self.root / "depth" / f"{self.names[i]}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"missing depth for frame {self.names[i]}")
        return d


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
