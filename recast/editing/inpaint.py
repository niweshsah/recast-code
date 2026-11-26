"""(v) Remove the human hand and arm with ProPainter video inpainting.

Runs ProPainter's own ``inference_propainter.py`` from a pinned checkout over the whole clip (so it can borrow
background from other frames), then checks that pixels outside the dilated removal mask are unchanged.

    python -m recast.editing.inpaint runs/cube --propainter third_party/ProPainter
Output ``<run>/inpainted/<frame>.png`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from recast.io.clip import Clip


def outside_mask_change(src: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute 8-bit difference outside the mask (should be 0: inpainting must not touch the rest)."""
    keep = ~mask
    return float(np.abs(src.astype(int) - out.astype(int))[keep].mean()) if keep.any() else 0.0


def inpaint(run: Path, propainter: Path, mask_dilation: int = 4, subvideo_length: int = 80, fp16: bool = True) -> dict:
    clip = Clip(run / "clip")
    masks = run / "masks" / "remove"
    out = run / "inpainted"
    work = run / "_propainter_work"
    shutil.rmtree(work, ignore_errors=True)
    frames = work / "frames"          # ProPainter expects matching file names for frames and masks
    frames.mkdir(parents=True)
    for i, name in enumerate(clip.names):
        cv2.imwrite(str(frames / f"{name}.png"), clip.rgb(i))
    t0 = time.time()
    cmd = [sys.executable, "inference_propainter.py", "-i", str(frames.resolve()), "-m", str(masks.resolve()),
           "-o", str((work / "out").resolve()), "--save_frames", "--mask_dilation", str(mask_dilation),
           "--subvideo_length", str(subvideo_length), "--save_fps", str(round(clip.fps))]
    if fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, cwd=propainter, check=True)
    produced = sorted((work / "out" / "frames" / "frames").glob("*.png"))
    if len(produced) != len(clip):
        raise RuntimeError(f"ProPainter wrote {len(produced)} frames for {len(clip)} inputs")
    out.mkdir(parents=True, exist_ok=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mask_dilation + 7, 2 * mask_dilation + 7))
    diffs = []
    for i, (name, p) in enumerate(zip(clip.names, produced, strict=True)):
        res = cv2.imread(str(p))
        cv2.imwrite(str(out / f"{name}.png"), res)
        m = cv2.dilate(cv2.imread(str(masks / f"{name}.png"), cv2.IMREAD_GRAYSCALE), k) > 127
        diffs.append(outside_mask_change(clip.rgb(i), res, m))
    shutil.rmtree(work, ignore_errors=True)
    summary = {"frames": len(clip), "seconds": round(time.time() - t0, 1), "fp16": fp16,
               "mask_dilation_px": mask_dilation, "mean_abs_diff_outside_mask": float(np.mean(diffs)),
               "max_abs_diff_outside_mask": float(np.max(diffs))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--propainter", type=Path, default=Path("third_party/ProPainter"))
    ap.add_argument("--mask-dilation", type=int, default=4)
    ap.add_argument("--subvideo-length", type=int, default=80)
    a = ap.parse_args()
    print(json.dumps(inpaint(a.run, a.propainter.resolve(), a.mask_dilation, a.subvideo_length), indent=2))


if __name__ == "__main__":
    main()
