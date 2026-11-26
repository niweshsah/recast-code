"""Text-prompted segmentation and tracking of the hand, the arm and the manipulated object.

Outputs (``<run>/masks``): ``hand/``, ``arm/``, ``object/`` binary masks per frame and ``remove/`` = dilated
(hand | arm), the region handed to the inpainter. Hand and arm are kept disjoint (hand wins at the wrist).

Backends
  * ``sam3``      SAM 3 [Carion et al.]: text prompt -> detect, segment and track every instance (paper default).
                  SAM 3 needs Python >= 3.12; install it in its own env (see envs/sam3.txt).
  * ``dino_sam2`` Grounding DINO picks the best-scoring box for each prompt over the clip, SAM 2 propagates it
                  forward and backward. Used for the qualitative figures when SAM 3 weights were gated.

    python -m recast.perception.segmentation runs/cube --task cube [--backend dino_sam2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from recast.config import load_config
from recast.io.clip import Clip, write_mask

KEYS = ("hand", "arm", "object")


# --------------------------------------------------------------------------------------------- helpers
def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_prompt(cands: list[tuple[int, list, float]], avoid: list | None = None, max_iou: float = 0.3):
    """Highest-scoring (frame, box, score), skipping boxes that overlap an ``avoid`` box on the same frame."""
    ok = [c for c in cands if not avoid or all(iou(c[1], a[1]) < max_iou for a in avoid if a[0] == c[0])]
    return max(ok, key=lambda c: c[2]) if ok else None


def write_jpeg_folder(clip: Clip) -> Path:
    """Both trackers read a folder of JPEGs named by frame index."""
    tmp = Path(tempfile.mkdtemp(prefix="recast_frames_"))
    for i in range(len(clip)):
        cv2.imwrite(str(tmp / f"{i:05d}.jpg"), clip.rgb(i), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return tmp


def finalize(masks: dict[str, np.ndarray], dilate_px: int) -> dict[str, np.ndarray]:
    """Disjoint hand/arm and the dilated removal region."""
    if "arm" in masks and "hand" in masks:
        masks["arm"] &= ~masks["hand"]
    rem = np.zeros_like(next(iter(masks.values())))
    for key in ("hand", "arm"):
        if key in masks:
            rem |= masks[key]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    masks["remove"] = np.stack([cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in rem])
    return masks


# --------------------------------------------------------------------------------------------- SAM 3
class Sam3Backend:
    """SAM 3 video predictor driven by text prompts (``sam3`` package, facebookresearch/sam3)."""

    def __init__(self):
        from sam3.model_builder import build_sam3_video_predictor

        self.predictor = build_sam3_video_predictor()

    def track(self, frames_dir: Path, n: int, shape: tuple[int, int], prompts: dict[str, str], prompt_frame: int = 0):
        out = {}
        session = self.predictor.handle_request({"type": "start_session", "resource_path": str(frames_dir)})
        sid = session["session_id"]
        for key, text in prompts.items():
            self.predictor.handle_request({"type": "reset_session", "session_id": sid})
            self.predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": prompt_frame,
                                           "text": text})
            m = np.zeros((n, *shape), bool)
            for resp in self.predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
                o = resp["outputs"]
                for mask in o["out_binary_masks"]:
                    m[resp["frame_index"]] |= np.asarray(mask, bool)
            out[key] = m
        self.predictor.handle_request({"type": "close_session", "session_id": sid})
        return out


# --------------------------------------------------------------------------------------------- Grounding DINO + SAM 2
class DinoSam2Backend:
    """Grounding DINO box prompt on the best frame, then SAM 2 video propagation (forward and backward)."""

    def __init__(self, detector: str = "IDEA-Research/grounding-dino-base",
                 sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 sam2_ckpt: str = "third_party/weights/sam2.1_hiera_large.pt", stride: int = 3,
                 threshold: float = 0.3):
        self.detector, self.sam2_cfg, self.sam2_ckpt = detector, sam2_cfg, sam2_ckpt
        self.stride, self.threshold = stride, threshold

    def _candidates(self, clip: Clip, prompts: dict[str, str]) -> dict[str, list]:
        import torch
        from PIL import Image
        from transformers import pipeline

        dev = 0 if torch.cuda.is_available() else -1
        det = pipeline(model=self.detector, task="zero-shot-object-detection", device=dev)
        cands: dict[str, list] = {k: [] for k in prompts}
        for fi in range(0, len(clip), self.stride):
            img = Image.fromarray(cv2.cvtColor(clip.rgb(fi), cv2.COLOR_BGR2RGB))
            for key, text in prompts.items():
                for r in det(img, candidate_labels=[f"{text}."], threshold=self.threshold):
                    b = r["box"]
                    cands[key].append((fi, [b["xmin"], b["ymin"], b["xmax"], b["ymax"]], float(r["score"])))
        del det
        torch.cuda.empty_cache()
        return cands

    def track(self, clip: Clip, frames_dir: Path, prompts: dict[str, str]):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        cands = self._candidates(clip, prompts)
        chosen = {k: best_prompt(cands[k]) for k in prompts if k != "arm"}
        if "arm" in prompts:  # the arm box must not just be the hand again
            strong_hands = [c for c in cands.get("hand", []) if c[2] >= 0.5]
            chosen["arm"] = best_prompt(cands["arm"], avoid=strong_hands, max_iou=0.6)
        ids = {k: i for i, k in enumerate(chosen) if chosen[k] is not None}
        n, shape = len(clip), clip.rgb(0).shape[:2]
        masks = {k: np.zeros((n, *shape), bool) for k in ids}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        vp = build_sam2_video_predictor(self.sam2_cfg, self.sam2_ckpt, device=dev)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
        with torch.inference_mode(), amp:
            for reverse in (False, True):
                state = vp.init_state(video_path=str(frames_dir))
                for key, oid in ids.items():
                    fi, box, _ = chosen[key]
                    vp.add_new_points_or_box(state, frame_idx=int(fi), obj_id=oid, box=np.array(box))
                for fidx, obj_ids, logits in vp.propagate_in_video(state, reverse=reverse):
                    for j, oid in enumerate(obj_ids):
                        key = next(k for k, v in ids.items() if v == oid)
                        masks[key][fidx] |= (logits[j] > 0.0).cpu().numpy().squeeze(0)
                vp.reset_state(state)
        info = {k: (None if v is None else {"frame": v[0], "box": v[1], "score": v[2]}) for k, v in chosen.items()}
        return masks, info


# --------------------------------------------------------------------------------------------- entry point
def segment(run: Path, cfg: dict, backend: str | None = None) -> dict:
    clip = Clip(run / "clip")
    seg = cfg["segmentation"]
    prompts = {"hand": seg["prompts"]["hand"], "arm": seg["prompts"]["arm"], "object": cfg["task"]["object_prompt"]}
    backend = backend or seg["backend"]
    frames = write_jpeg_folder(clip)
    try:
        if backend == "sam3":
            masks, info = Sam3Backend().track(frames, len(clip), clip.rgb(0).shape[:2], prompts), {}
        elif backend == "dino_sam2":
            masks, info = DinoSam2Backend(stride=seg["detector_stride"], threshold=seg["detector_threshold"]).track(
                clip, frames, prompts)
        else:
            raise ValueError(f"unknown segmentation backend {backend!r}")
    finally:
        shutil.rmtree(frames, ignore_errors=True)
    masks = finalize(masks, seg["remove_dilation_px"])
    out = run / "masks"
    for key, m in masks.items():
        for i, name in enumerate(clip.names):
            write_mask(out / key / f"{name}.png", m[i])
    summary = {"backend": backend, "prompts": prompts, "prompt_boxes": info,
               "frames_nonempty": {k: int(m.any(axis=(1, 2)).sum()) for k, m in masks.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory containing clip/")
    ap.add_argument("--task", required=True)
    ap.add_argument("--backend", choices=["sam3", "dino_sam2"])
    a = ap.parse_args()
    print(json.dumps(segment(a.run, load_config(a.task), a.backend), indent=2))


if __name__ == "__main__":
    main()
