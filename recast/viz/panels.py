"""Per-step image panels for figures and the showcase video (pure tiling/tinting of saved stage outputs).

    python -m recast.viz.panels runs/cube                # all frames -> <run>/viz/steps/<step>/<frame>.png
    python -m recast.viz.panels runs/cube --frames 59 67 75 86 94

Steps: input, segmentation, keypoints, mesh, depth, removal, inpaint, robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from recast.io.clip import Clip, frame_path, read_mask

COLORS = {"hand": (180, 119, 31), "arm": (14, 127, 255), "object": (40, 39, 214)}   # BGR: blue, orange, red
STEPS = ("input", "segmentation", "keypoints", "mesh", "depth", "removal", "inpaint", "robot")


def tint(img: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.55) -> np.ndarray:
    out = img.astype(np.float32)
    for key, m in masks.items():
        out[m] = (1 - alpha) * out[m] + alpha * np.array(COLORS[key], np.float32)
    return out.astype(np.uint8)


def removal_region(img: np.ndarray, hand: np.ndarray, arm: np.ndarray, remove: np.ndarray) -> np.ndarray:
    """Hand (blue) and arm (orange) with the dilated inpainting region outlined in white."""
    out = tint(img, {"hand": hand, "arm": arm})
    contours, _ = cv2.findContours(remove.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def export_steps(run: Path, frames: list[int] | None = None) -> Path:
    clip = Clip(run / "clip")
    out = run / "viz" / "steps"
    for s in STEPS:
        (out / s).mkdir(parents=True, exist_ok=True)
    sources = {"keypoints": run / "hands" / "keypoints", "mesh": run / "hands" / "overlay",
               "depth": run / "hands" / "on_depth", "inpaint": run / "inpainted", "robot": run / "edited"}
    for i in frames if frames is not None else range(len(clip)):
        name = f"{clip.names[i]}.png"
        rgb = clip.rgb(i)
        m = {k: read_mask(run / "masks" / k / name) for k in ("hand", "arm", "object", "remove")}
        cv2.imwrite(str(out / "input" / name), rgb)
        cv2.imwrite(str(out / "segmentation" / name), tint(rgb, {k: m[k] for k in COLORS}))
        cv2.imwrite(str(out / "removal" / name), removal_region(rgb, m["hand"], m["arm"], m["remove"]))
        for step, src in sources.items():
            try:
                cv2.imwrite(str(out / step / name), cv2.imread(str(frame_path(src, clip.names[i]))))
            except FileNotFoundError:  # stage not run (or no output for this frame): show the input
                cv2.imwrite(str(out / step / name), rgb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--frames", type=int, nargs="*")
    a = ap.parse_args()
    print(export_steps(a.run, a.frames))


if __name__ == "__main__":
    main()
