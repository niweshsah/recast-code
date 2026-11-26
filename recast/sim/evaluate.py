"""Closed-loop evaluation in MuJoCo (paper Sec. IV), fixed before testing:

* every model runs in the same scene, arm, camera, OSC controller and gripper, with the same instruction;
* 25 object placements from seeds 0..24, executed in the same order for every model (paired comparison);
* 60 s deadline; success = the object rests on the top surface of the box at the end of the rollout;
* held-out conditions on the Cube task: unseen surface, different lighting, distractor objects;
* success rates are reported with 95% Wilson intervals; rollout time counts failures as the full 60 s.

    MUJOCO_GL=egl python -m recast.sim.evaluate --task cube --model recast --checkpoint checkpoints/recast_cube \
        --condition in_distribution --out results/cube_recast.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from recast.config import load_config, load_yaml
from recast.data.episodes import IMAGE_SIZE, state_vector
from recast.geometry import farthest_point_sampling
from recast.policy.gate import gate_gripper
from recast.sim.env import PandaSim
from recast.sim.scene import SceneSpec, load_scene
from recast.sim.scripted import is_on_top
from recast.stats import summarize

SETTLE_S = 2.0          # object must rest on the box this long (with the gripper open) to end a rollout early


class Policy(Protocol):
    def reset(self, instruction: str) -> None: ...

    def act(self, image: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, float, int | None]:
        """-> (delta pose (6), gripper in [0, 1], predicted phase or None)."""


# --------------------------------------------------------------------------------------------- placements
def sample_placement(spec: SceneSpec, seed: int, xy_jitter: float, yaw_range_deg: float) -> SceneSpec:
    rng = np.random.default_rng(seed)
    xy = spec.object_xy + rng.uniform(-xy_jitter, xy_jitter, 2)
    yaw = spec.object_yaw + np.radians(rng.uniform(-yaw_range_deg, yaw_range_deg))
    return replace(spec, object_xy=xy, object_yaw=float(yaw))


def object_surface_points(spec: SceneSpec, m: int = 16, seed: int = 0) -> np.ndarray:
    """M farthest-point-sampled surface points of the task object in its own frame (ground-truth keypoints)."""
    rng = np.random.default_rng(seed)
    o = spec.obj
    if o["shape"] == "box":
        h = np.array(o["size_m"]) / 2
        pts = rng.uniform(-h, h, (4000, 3))
        axis = rng.integers(0, 3, len(pts))
        pts[np.arange(len(pts)), axis] = np.sign(rng.uniform(-1, 1, len(pts))) * h[axis]
    else:
        r, hh = o["radius_m"], o["half_height_m"]
        a = rng.uniform(0, 2 * np.pi, 4000)
        pts = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(-hh, hh, 4000)], 1)
    return pts[farthest_point_sampling(pts, m, seed)]


# --------------------------------------------------------------------------------------------- policies
class OctoPolicy:
    """Fine-tuned (or original) Octo checkpoint, executed with receding-horizon action chunks."""

    def __init__(self, checkpoint: str, dataset: str | None = None, window: int = 2, use_phase: bool = True,
                 execute_steps: int = 1, seed: int = 0):
        import jax
        from octo.model.octo_model import OctoModel

        self.jax = jax
        self.model = OctoModel.load_pretrained(checkpoint)
        stats = self.model.dataset_statistics
        self.stats = stats[dataset]["action"] if dataset else stats["action"] if "action" in stats else None
        self.window, self.execute_steps = window, execute_steps
        self.use_phase = use_phase and "phase" in self.model.module.heads
        self.rng = jax.random.PRNGKey(seed)
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self.queue: list[tuple[np.ndarray, int | None]] = []

    def reset(self, instruction: str) -> None:
        self.task = self.model.create_tasks(texts=[instruction])
        self.history, self.queue = [], []

    def _observation(self):
        hist = self.history[-self.window:]
        pad = self.window - len(hist)
        hist = [hist[0]] * pad + hist
        obs = {"image_primary": np.stack([h[0] for h in hist])[None],
               "proprio": np.stack([h[1] for h in hist])[None],
               "timestep_pad_mask": np.array([[False] * pad + [True] * (self.window - pad)])}
        return obs

    def act(self, image, state):
        import cv2

        self.history.append((cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), state))
        if not self.queue:
            obs = self._observation()
            self.rng, key = self.jax.random.split(self.rng)
            chunk = np.asarray(self.model.sample_actions(obs, self.task, unnormalization_statistics=self.stats,
                                                         rng=key))[0]                       # (H, 7)
            phases = [None] * len(chunk)
            if self.use_phase:
                emb = self.model.run_transformer(obs, self.task, obs["timestep_pad_mask"], train=False)
                logits = self.model.module.apply({"params": self.model.params}, emb, train=False,
                                                 method=lambda m, e, train: m.heads["phase"](e, train=train))
                phases = np.asarray(logits)[0, -1].argmax(-1).tolist()
            self.queue = list(zip(chunk, phases))[: self.execute_steps]
        a, phase = self.queue.pop(0)
        return a[:6], float(a[6]), phase


class OracleKeypoints:
    """Object state from the simulator's ground-truth object pose (upper bound on online tracking quality)."""

    def __init__(self, spec: SceneSpec, m: int):
        self.local = object_surface_points(spec, m)

    def __call__(self, sim: PandaSim) -> np.ndarray:
        return sim.object_keypoints_ee(self.local)


class TrackedKeypoints:
    """Online object state as at deployment: CoTracker (online) on the robot-camera stream, keypoints lifted with
    the rendered depth and limited to the simulator's object segmentation (stands in for the SAM 3 mask)."""

    def __init__(self, spec: SceneSpec, m: int, device: str | None = None):
        import torch

        self.torch, self.m, self.spec = torch, m, spec
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.dev)
        self.frames: list[np.ndarray] = []
        self.queries = None
        self.last = None

    def __call__(self, sim: PandaSim) -> np.ndarray:
        from recast.geometry import invert_T, kabsch, lift_pixels, transform_points
        from recast.perception.keypoints import init_queries

        k, T_bc = self.spec.k, self.spec.T_base_cam
        rgb = sim.render("rs")
        r = sim.renderers[("rs", k["height"], k["width"])]
        r.enable_depth_rendering()
        depth_mm = (r.render() * 1000).astype(np.uint16)
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        seg = r.render()[..., 0]
        r.disable_segmentation_rendering()
        mask = seg == sim.mj.mj_name2id(sim.m, sim.mj.mjtObj.mjOBJ_GEOM, "object_geom")
        if self.queries is None:
            self.queries = init_queries(depth_mm, mask, k, self.m)
        self.frames.append(rgb)
        step = self.model.step
        video = self.torch.from_numpy(np.stack(self.frames[-2 * step:])).permute(0, 3, 1, 2)[None].float().to(self.dev)
        q = self.torch.from_numpy(np.concatenate([np.zeros((self.m, 1)), self.queries], 1))[None].float().to(self.dev)
        with self.torch.no_grad():
            tracks, vis = self.model(video_chunk=video, is_first_step=len(self.frames) == 1, queries=q)
        uv = tracks[0, -1].cpu().numpy() if tracks is not None else self.queries
        pts_c, ok = lift_pixels(uv, depth_mm, k)
        pts_w = transform_points(T_bc, pts_c)
        if self.last is None:
            self.last = pts_w
        elif ok.sum() >= 3:
            self.last = transform_points(kabsch(self.last[ok], pts_w[ok]), self.last)
            self.last[ok] = pts_w[ok]
        return transform_points(invert_T(sim.tcp_pose()), self.last)


# --------------------------------------------------------------------------------------------- rollout
def rollout(policy: Policy, spec: SceneSpec, panda_dir: Path, instruction: str, gate: bool, keypoints: str,
            num_points: int, deadline_s: float, control_hz: float, seed: int) -> dict:
    sim = PandaSim(spec, panda_dir, control="torque", seed=seed)
    sim.settle(0.5)
    kp = OracleKeypoints(spec, num_points) if keypoints == "oracle" else TrackedKeypoints(spec, num_points)
    policy.reset(instruction)
    dt, rested, track_ms = 1.0 / control_hz, 0.0, []
    success_time = None
    while sim.time < deadline_s:
        t0 = time.perf_counter()
        s_o = kp(sim)
        track_ms.append(1000 * (time.perf_counter() - t0))
        g_now = sim.gripper_opening() / 0.08
        state = state_vector(sim.d.qpos[:7].copy(), g_now, s_o)
        delta, grip, phase = policy.act(sim.render("rs"), state)
        if gate and phase is not None:
            grip = float(gate_gripper(grip, phase))
        sim.apply_action(delta, grip, dt)
        on_box = is_on_top(sim.object_pose()[:3, 3], spec)
        rested = rested + dt if on_box and grip > 0.5 else 0.0
        if rested >= SETTLE_S:
            success_time = sim.time
            break
    success = is_on_top(sim.object_pose()[:3, 3], spec)
    sim.close()
    return {"seed": seed, "success": bool(success), "time_s": success_time if success else deadline_s,
            "tracking_ms": float(np.mean(track_ms)) if track_ms else None}


def evaluate(task: str, model: str, checkpoint: str, condition: str, panda_dir: Path, keypoints: str,
             dataset: str | None = None) -> dict:
    cfg, ev = load_config(task), load_yaml("eval.yaml")
    base = load_scene(cfg["task"], ev["conditions"][condition])
    policy = OctoPolicy(checkpoint, dataset=dataset, use_phase=model == "recast")
    trials = []
    for seed in range(ev["rollouts"]):
        spec = sample_placement(base, seed, ev["placement"]["xy_jitter_m"], ev["placement"]["yaw_range_deg"])
        trials.append(rollout(policy, spec, panda_dir, cfg["task"]["instruction"], gate=model == "recast",
                              keypoints=keypoints, num_points=cfg["keypoints"]["num_points"],
                              deadline_s=ev["deadline_s"], control_hz=ev["control_hz"], seed=seed))
        print(f"[{task}/{model}/{condition}] seed {seed}: {'success' if trials[-1]['success'] else 'fail'}",
              flush=True)
    return {"task": task, "model": model, "condition": condition, "checkpoint": checkpoint, "keypoints": keypoints,
            "summary": summarize([t["success"] for t in trials]),
            "mean_rollout_time_s": float(np.mean([t["time_s"] for t in trials])), "trials": trials}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["cube", "tape", "battery"])
    ap.add_argument("--model", required=True, choices=["original", "hand_only", "recast"])
    ap.add_argument("--checkpoint", required=True, help="Octo checkpoint dir or hf:// path")
    ap.add_argument("--dataset", help="dataset key for action unnormalisation statistics")
    ap.add_argument("--condition", default="in_distribution",
                    choices=["in_distribution", "surface", "lighting", "distractors"])
    ap.add_argument("--keypoints", choices=["tracked", "oracle"], default="tracked")
    ap.add_argument("--menagerie", type=Path, default=Path("third_party/mujoco_menagerie"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = evaluate(a.task, a.model, a.checkpoint, a.condition, a.menagerie / "franka_emika_panda", a.keypoints,
                   a.dataset)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    print(f"{a.task}/{a.model}/{a.condition}: {s['rate']:.0f}% [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}]")


if __name__ == "__main__":
    main()
