"""Assemble processed runs into training episodes, and export them as an RLDS (TFDS) dataset for Octo.

Per step t of a demonstration:
    observation.image_primary  edited frame I_{r,t}, RGB, resized to 256x256
    observation.state          [q_t (7), g_t (1), s^o_t (M*3)]   proprioception + end-effector-frame object state
    action                     [dp (3), drot (3), g_{t+1} (1), phase_t (1)]
    language_instruction       task instruction, e.g. "put the tape on the box"

Octo's data loader forwards image keys and one proprio key only, so the object state travels inside ``state``
(split again by the point tokenizer) and the phase label travels as the last action dimension, which also
chunks it over the action horizon like the actions themselves. Normalisation is masked off for the gripper and
phase dimensions (see ``ACTION_NORMALIZATION_MASK``).

    python -m recast.data.episodes build runs/cube_* --task cube --out datasets/recast_cube   # npz episodes
    python -m recast.data.episodes rlds datasets/recast_cube --out ~/tensorflow_datasets      # TFDS / RLDS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recast.perception.keypoints import object_state

IMAGE_SIZE = 256
PROPRIO_DIM = 8                     # 7 joints + gripper
ACTION_DIM = 8                      # 6 delta pose + gripper + phase
ACTION_NORMALIZATION_MASK = [True] * 6 + [False, False]


def state_vector(q: np.ndarray, g: float, s_o: np.ndarray) -> np.ndarray:
    return np.concatenate([q, [g], s_o.reshape(-1)]).astype(np.float32)


def split_state(state: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``state_vector`` on the last axis: (proprio [..., 8], points [..., M, 3])."""
    proprio = state[..., :PROPRIO_DIM]
    points = state[..., PROPRIO_DIM:PROPRIO_DIM + 3 * num_points]
    return proprio, points.reshape(*points.shape[:-1], num_points, 3)


def build_episode(run: Path, instruction: str, hand_only: bool = False) -> dict[str, np.ndarray]:
    """Arrays for one demonstration, restricted to frames with a valid robot pose.

    ``hand_only=True`` produces the baseline variant: zero object state (no point tokens are used by that model)
    and phase set to -1 so the phase loss is masked."""
    from recast.io.clip import Clip

    clip = Clip(run / "clip")
    with np.load(run / "actions" / "actions.npz") as f:
        T_WE, act, gripper, phase, valid = f["T_WE"], f["action"], f["gripper"], f["phase"], f["valid"]
    with np.load(run / "keypoints" / "keypoints.npz") as f:
        K = f["K_world"]
    with np.load(run / "edited" / "joints.npz") as f:
        q, q_valid = f["q"], f["valid"]
    keep = np.flatnonzero(valid & q_valid)
    if len(keep) < 2:
        raise ValueError(f"{run}: fewer than two usable frames")
    images, states, actions = [], [], []
    for t in keep:
        img = cv2.imread(str(run / "edited" / f"{clip.names[t]}.png"))
        images.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_AREA))
        s_o = np.zeros_like(K[t]) if hand_only else object_state(K[t], T_WE[t])
        states.append(state_vector(q[t], gripper[t], s_o))
        actions.append(np.concatenate([act[t], [-1 if hand_only else phase[t]]]).astype(np.float32))
    return {"image": np.stack(images), "state": np.stack(states), "action": np.stack(actions),
            "instruction": np.array(instruction), "source": np.array(str(run))}


def build(runs: list[Path], instruction: str, out: Path, hand_only: bool = False) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in runs:
        try:
            ep = build_episode(run, instruction, hand_only)
        except (FileNotFoundError, ValueError) as e:
            print(f"skip {run}: {e}")
            continue
        np.savez_compressed(out / f"episode_{n:05d}.npz", **ep)
        n += 1
    (out / "meta.json").write_text(json.dumps({"episodes": n, "instruction": instruction, "hand_only": hand_only,
                                               "image_size": IMAGE_SIZE, "action_dim": ACTION_DIM}, indent=2))
    return n


# --------------------------------------------------------------------------------------------- RLDS export
def make_tfds_builder(episodes_dir: Path, name: str):
    """A TFDS GeneratorBasedBuilder over npz episodes (imports tensorflow_datasets lazily)."""
    import tensorflow_datasets as tfds

    files = sorted(Path(episodes_dir).glob("episode_*.npz"))
    with np.load(files[0]) as f:
        state_dim = f["state"].shape[1]

    class RecastDataset(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")

        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image_primary": tfds.features.Image(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
                        "state": tfds.features.Tensor(shape=(state_dim,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                    "language_instruction": tfds.features.Text(),
                    "is_first": np.bool_, "is_last": np.bool_, "is_terminal": np.bool_,
                    "reward": np.float32, "discount": np.float32,
                }),
                "episode_metadata": tfds.features.FeaturesDict({"source": tfds.features.Text()}),
            }))

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples(files)}

        def _generate_examples(self, paths):
            for p in paths:
                with np.load(p) as ep:
                    T = len(ep["action"])
                    steps = [{"observation": {"image_primary": ep["image"][t], "state": ep["state"][t]},
                              "action": ep["action"][t], "language_instruction": str(ep["instruction"]),
                              "is_first": t == 0, "is_last": t == T - 1, "is_terminal": t == T - 1,
                              "reward": float(t == T - 1), "discount": 1.0} for t in range(T)]
                    yield p.stem, {"steps": steps, "episode_metadata": {"source": str(ep["source"])}}

    RecastDataset.name = name
    return RecastDataset


def export_rlds(episodes_dir: Path, data_dir: Path, name: str) -> None:
    builder = make_tfds_builder(episodes_dir, name)(data_dir=str(data_dir))
    builder.download_and_prepare()
    print(f"RLDS dataset '{name}' written under {data_dir}")


def main() -> None:
    from recast.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("runs", type=Path, nargs="+")
    b.add_argument("--task", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--hand-only", action="store_true", help="baseline variant without object state and phases")
    r = sub.add_parser("rlds")
    r.add_argument("episodes", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        n = build(a.runs, load_config(a.task)["task"]["instruction"], a.out, a.hand_only)
        print(f"{n} episodes -> {a.out}")
    else:
        export_rlds(a.episodes, a.out, a.name or a.episodes.name)


if __name__ == "__main__":
    main()
