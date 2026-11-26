"""Showcase video (1920x1080, 30 fps, H.264): title -> recorded demonstration -> every pipeline step on the same
frames -> simulated Panda runs side by side (labelled A, B, C only).

    python -m recast.viz.panels runs/cube
    python -m recast.sim.scripted --task cube --variant drop    --video outputs/sim_drop.mp4
    python -m recast.sim.scripted --task cube --variant success --video outputs/sim_success.mp4
    python -m recast.sim.scripted --task cube --variant miss    --video outputs/sim_miss.mp4
    python -m recast.viz.video runs/cube --segment 52 102 --sim outputs/sim_drop.mp4 outputs/sim_success.mp4 \
        outputs/sim_miss.mp4 --out outputs/recast_showcase.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
BG, FG, MUTED, ACCENT = (18, 20, 24), (240, 240, 240), (150, 156, 166), (42, 120, 214)
FONTS = ["/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
BOLD = ["/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]

STEPS = [  # (title, subtitle, folder in <run>/viz/steps)
    ("Text-prompted segmentation", "Object (red), hand (blue) and arm (orange) masks, tracked through the video",
     "segmentation"),
    ("Hand keypoints (WiLoR)", "21 hand joints located in the image, linked along each finger", "keypoints"),
    ("MANO hand mesh (WiLoR)", "Full hand surface reconstructed from each RGB frame", "mesh"),
    ("Mesh registered to sensor depth (ICP)", "Mesh vertices (blue) aligned to the depth map: metric 3D hand", "depth"),
    ("Hand and arm removal region", "Hand (blue) and arm (orange); white outline = region given to the inpainter",
     "removal"),
    ("Video inpainting (ProPainter)", "Human hand and arm erased; background filled in, temporally consistent",
     "inpaint"),
    ("Rendered robot at the contact points", "Franka Panda rendered in MuJoCo, jaws placed at hand-object contacts",
     "robot"),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in BOLD if bold else FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
    return ImageFont.load_default(size)


def canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def text_c(d: ImageDraw.ImageDraw, y: int, s: str, f, fill=FG, cx: int = W // 2) -> None:
    d.text((cx - d.textlength(s, font=f) / 2, y), s, font=f, fill=fill)


def paste_fit(img: Image.Image, im, box: tuple[int, int, int, int]) -> None:
    im = Image.fromarray(im) if isinstance(im, np.ndarray) else im
    x, y, w, h = box
    s = min(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    img.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def header(img: Image.Image, kicker: str, title: str, subtitle: str = "") -> None:
    d = ImageDraw.Draw(img)
    d.text((80, 44), kicker.upper(), font=font(26, True), fill=ACCENT)
    d.text((80, 80), title, font=font(48, True), fill=FG)
    if subtitle:
        d.text((80, 142), subtitle, font=font(28), fill=MUTED)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def hold(img, seconds: float):
    a = np.asarray(img)
    for _ in range(round(seconds * FPS)):
        yield a


class Sequencer:
    """Writes frame streams, crossfading from the previous segment."""

    def __init__(self, writer):
        self.w, self.last, self.n = writer, None, 0

    def put(self, frame: np.ndarray) -> None:
        self.w.append_data(frame)
        self.last, self.n = frame, self.n + 1

    def stream(self, frames, xfade: float = 0.4) -> None:
        it = iter(frames)
        first = next(it)
        if self.last is not None and xfade > 0:
            a, b = self.last.astype(np.float32), first.astype(np.float32)
            n = max(round(xfade * FPS), 1)
            for i in range(1, n + 1):
                self.put((a * (1 - i / n) + b * i / n).astype(np.uint8))
        else:
            self.put(first)
        for f in it:
            self.put(f)


def title_card() -> Image.Image:
    img = canvas()
    d = ImageDraw.Draw(img)
    text_c(d, 360, "Recast", font(120, True))
    text_c(d, 520, "Object-Linked, Contact-Grounded Robot Demonstrations", font(46))
    text_c(d, 580, "from Human Videos", font(46))
    text_c(d, 700, "Human RGB-D video   /   edited robot demonstration   /   simulated Franka Panda", font(30), MUTED)
    return img


def section_card(num: int, title: str, subtitle: str) -> Image.Image:
    img = canvas()
    d = ImageDraw.Draw(img)
    text_c(d, 400, f"{num:02d}", font(64, True), ACCENT)
    text_c(d, 490, title, font(72, True))
    text_c(d, 600, subtitle, font(32), MUTED)
    return img


def demo_frames(steps: Path, names: list[str], fps_in: float):
    base = canvas()
    header(base, "Training data", "One recorded human demonstration", "Third-person RGB-D recording of the task")
    for j, n in enumerate(names):
        img = base.copy()
        paste_fit(img, load_rgb(steps / "input" / f"{n}.png"), (80, 200, W - 160, H - 260))
        ImageDraw.Draw(img).text((W - 330, H - 50), f"t = {j / fps_in:4.1f} s", font=font(26), fill=MUTED)
        yield from hold(img, 1 / fps_in)


def step_frames(steps: Path, names: list[str], k: int, title: str, sub: str, folder: str, slow: int):
    base = canvas()
    header(base, f"Pipeline step {k} of {len(STEPS)}", title, sub)
    pw, ph, y = 880, 660, 220
    xl, xr = (W - 2 * pw - 40) // 2, (W + 40) // 2
    d = ImageDraw.Draw(base)
    text_c(d, y + ph + 18, "Input video", font(28), MUTED, cx=xl + pw // 2)
    text_c(d, y + ph + 18, title, font(28, True), FG, cx=xr + pw // 2)
    bx0, bx1, by = 560, W - 560, H - 64
    for j, n in enumerate(names):
        img = base.copy()
        paste_fit(img, load_rgb(steps / "input" / f"{n}.png"), (xl, y, pw, ph))
        paste_fit(img, load_rgb(steps / folder / f"{n}.png"), (xr, y, pw, ph))
        d = ImageDraw.Draw(img)
        d.rectangle((bx0, by, bx1, by + 6), fill=(70, 74, 82))
        d.rectangle((bx0, by, bx0 + (bx1 - bx0) * j / max(len(names) - 1, 1), by + 6), fill=ACCENT)
        text_c(d, by + 18, f"frame {int(n)}", font(22), MUTED)
        yield from [np.asarray(img)] * slow


def sim_frames(videos: list[Path], title: str, tail_s: float = 2.5):
    import imageio.v2 as imageio

    clips = []
    for v in videos:
        r = imageio.get_reader(v)
        clips.append([np.asarray(f) for f in r])
        r.close()
    base = canvas()
    d = ImageDraw.Draw(base)
    text_c(d, 40, title, font(40, True))
    pw = W // len(clips)
    for j in range(len(clips)):
        text_c(d, 110, "ABCDEFG"[j], font(64, True), FG, cx=j * pw + pw // 2)
    for i in range(max(len(c) for c in clips) + round(tail_s * FPS)):
        img = base.copy()
        for j, c in enumerate(clips):
            f = c[min(i, len(c) - 1)]
            img.paste(Image.fromarray(f).resize((pw - 12, round(f.shape[0] * (pw - 12) / f.shape[1]))),
                      (j * pw + 6, 200))
        yield np.asarray(img)


def make_video(run: Path, segment: tuple[int, int], demo: tuple[int, int], sims: list[Path], out: Path,
               fps_in: float = 15.0, slow: int = 4) -> int:
    import imageio.v2 as imageio

    steps = run / "viz" / "steps"
    seg = [f"{i:05d}" for i in range(segment[0], segment[1] + 1)]
    dem = [f"{i:05d}" for i in range(demo[0], demo[1] + 1)]
    out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
    s = Sequencer(w)
    s.stream(hold(title_card(), 4.0), xfade=0)
    s.stream(hold(section_card(1, "Training data", "A recorded human demonstration"), 2.5))
    s.stream(demo_frames(steps, dem, fps_in))
    s.stream(hold(s.last, 0.8), xfade=0)
    s.stream(hold(section_card(2, "Pipeline, step by step", "Every stage on the same grasp, carry and placement"), 2.5))
    for k, (title, sub, folder) in enumerate(STEPS, 1):
        s.stream(step_frames(steps, seg, k, title, sub, folder, slow))
        s.stream(hold(s.last, 0.5), xfade=0)
    if sims:
        s.stream(hold(section_card(3, "Simulation", "Runs of the same pick-and-place task"), 2.5))
        s.stream(sim_frames(sims, "Simulation: Franka Panda, pick the object and place it on the box"))
    s.stream(hold(canvas(), 0.8), xfade=0.8)
    w.close()
    return s.n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--segment", type=int, nargs=2, required=True, help="frames shown for every pipeline step")
    ap.add_argument("--demo", type=int, nargs=2, help="frames of the demonstration section (default: segment)")
    ap.add_argument("--sim", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("outputs/recast_showcase.mp4"))
    a = ap.parse_args()
    n = make_video(a.run, tuple(a.segment), tuple(a.demo or a.segment), a.sim, a.out)
    print(f"{a.out}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
