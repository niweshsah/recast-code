"""(iii) Contact-point gripper mapping and (iv) object-anchored pose correction -> robot action labels.

(iii) Grasp and carry frames (paper eq. 4): hand-mesh vertices within eps of the object cloud are split into thumb
      and non-thumb contacts by nearest MANO joint. With centroids c_th and c_nt:
          p = (c_th + c_nt) / 2,   y = (c_nt - c_th) / |c_nt - c_th|,   w = |c_nt - c_th|
      The approach axis u is the palm normal made orthogonal to y, and R = [y x u, y, u].
      Without contact: p = midpoint of the thumb and index tips, y along them. The two estimates are blended over a
      short window wherever the mode switches.
(iv)  At the grasp onset t_g:  T^E_O = (T^W_{E,t_g})^{-1} T^W_{O,t_g}          (eq. 5)
      During carry:           T^W_E-hat = T^W_{O,t} (T^E_O)^{-1}
                              p = lambda p_hand + (1 - lambda) p-hat,  lambda = exp(-r_t / sigma)   (eq. 6)
      r_t is the hand's ICP residual; orientations are slerped with the same weight.
      The spread of T^E_O over a carry interval measures label quality; intervals above a threshold are dropped.

    python -m recast.actions runs/cube
Output ``<run>/actions/actions.npz``: T_WE[T,4,4], width[T], gripper[T], phase[T], mode[T], valid[T], action[T,7].
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recast.geometry import (
    invert_T,
    make_T,
    rotation_angle_deg,
    rotation_mean,
    slerp,
    transform_points,
    unit,
)
from recast.perception.contact import nearest_distances
from recast.perception.hands import FINGER_JOINTS, THUMB_JOINTS
from recast.phases import CARRY, GRASP

MODE_NONE, MODE_FINGERTIP, MODE_CONTACT, MODE_ANCHORED = range(4)
MODE_NAMES = ("none", "fingertip", "contact", "object-anchored")
WRIST, THUMB_TIP, INDEX_TIP, INDEX_MCP, PINKY_MCP = 0, 4, 8, 5, 17


@dataclass
class GraspTarget:
    p: np.ndarray            # jaw midpoint (3,)
    R: np.ndarray            # [x, y = closing axis, z = approach] (3, 3)
    width: float             # jaw opening (m)
    mode: int
    n_thumb: int = 0
    n_finger: int = 0

    @property
    def T(self) -> np.ndarray:
        return make_T(self.R, self.p)


# --------------------------------------------------------------------------------------------- (iii)
def palm_normal(joints: np.ndarray) -> np.ndarray:
    """Unit normal of the palm plane (wrist, index MCP, pinky MCP), pointing from the back of the hand to the palm
    side, i.e. towards where the thumb and index tips meet."""
    n = unit(np.cross(joints[INDEX_MCP] - joints[WRIST], joints[PINKY_MCP] - joints[WRIST]))
    pinch = 0.5 * (joints[THUMB_TIP] + joints[INDEX_TIP])
    return n if np.dot(pinch - joints[WRIST], n) >= 0 else -n


def gripper_frame(p: np.ndarray, y_dir: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """R = [y x u, y, u] with y the closing axis and u the approach direction made orthogonal to y."""
    y = unit(y_dir)
    u = unit(approach - np.dot(approach, y) * y)
    return np.stack([np.cross(y, u), y, u], axis=1)


def split_contacts(verts: np.ndarray, joints: np.ndarray, obj_pts: np.ndarray, eps: float):
    """Vertices within eps of the object, split into thumb / other fingers by nearest hand joint."""
    if len(obj_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    near = verts[nearest_distances(verts, obj_pts) <= eps]
    if len(near) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    owner = np.argmin(np.linalg.norm(near[:, None, :] - joints[None], axis=2), axis=1)
    return near[np.isin(owner, THUMB_JOINTS)], near[np.isin(owner, FINGER_JOINTS)]


def grasp_from_hand(verts: np.ndarray, joints: np.ndarray, obj_pts: np.ndarray, eps: float = 0.01,
                    min_contacts: int = 3, approach: str = "palm") -> GraspTarget:
    """Paper eq. (4) when both contact groups exist, else the thumb-index fingertip fallback.

    approach="palm" uses the palm normal (paper); "wrist" uses wrist -> p (as in the qualitative figures)."""
    th, nt = split_contacts(verts, joints, obj_pts, eps)
    if len(th) >= min_contacts and len(nt) >= min_contacts:
        c_th, c_nt = th.mean(0), nt.mean(0)
        p, y, mode = 0.5 * (c_th + c_nt), c_nt - c_th, MODE_CONTACT
    else:
        p, mode = 0.5 * (joints[THUMB_TIP] + joints[INDEX_TIP]), MODE_FINGERTIP
        y = joints[INDEX_TIP] - joints[THUMB_TIP]
    u = palm_normal(joints) if approach == "palm" else p - joints[WRIST]
    return GraspTarget(p, gripper_frame(p, y, u), float(np.linalg.norm(y)), mode, len(th), len(nt))


def blend_transitions(targets: list[GraspTarget | None], frames: int) -> list[GraspTarget | None]:
    """Cross-fade pose and width over ``frames`` frames after every contact <-> fingertip switch."""
    out = list(targets)
    for t in range(1, len(targets)):
        a, b = targets[t - 1], targets[t]
        if a is None or b is None or a.mode == b.mode:
            continue
        for j in range(frames):
            if t + j >= len(targets) or targets[t + j] is None:
                break
            s = (j + 1) / (frames + 1)
            cur = targets[t + j]
            out[t + j] = GraspTarget((1 - s) * a.p + s * cur.p, slerp(a.R, cur.R, s),
                                     (1 - s) * a.width + s * cur.width, cur.mode, cur.n_thumb, cur.n_finger)
    return out


# --------------------------------------------------------------------------------------------- (iv)
def grasp_onset(phases: np.ndarray, targets: list[GraspTarget | None]) -> int | None:
    """First grasp-phase frame whose pose comes from the contact mapping (fallback: first grasp frame)."""
    grasp = [t for t in range(len(phases)) if phases[t] == GRASP and targets[t] is not None]
    contact = [t for t in grasp if targets[t].mode == MODE_CONTACT]
    return (contact or grasp or [None])[0]


def anchor(targets: list[GraspTarget | None], phases: np.ndarray, T_WO: np.ndarray, icp_rmse: np.ndarray,
           sigma: float = 0.01) -> list[GraspTarget | None]:
    """Paper eqs. (5)-(6): confidence-weighted fusion of the hand pose with the rigid-attachment prediction."""
    tg = grasp_onset(phases, targets)
    if tg is None:
        return list(targets)
    T_EO = invert_T(targets[tg].T) @ T_WO[tg]
    out = list(targets)
    for t in range(tg + 1, len(phases)):
        if phases[t] not in (GRASP, CARRY):   # release: the rigid-attachment assumption ends here
            break
        T_hat = T_WO[t] @ invert_T(T_EO)
        hand = targets[t]
        if hand is None or not np.isfinite(icp_rmse[t]):
            out[t] = GraspTarget(T_hat[:3, 3], T_hat[:3, :3], targets[tg].width, MODE_ANCHORED)
            continue
        lam = float(np.exp(-icp_rmse[t] / sigma))
        out[t] = GraspTarget(lam * hand.p + (1 - lam) * T_hat[:3, 3], slerp(T_hat[:3, :3], hand.R, lam),
                             targets[tg].width, MODE_ANCHORED, hand.n_thumb, hand.n_finger)
    return out


def anchor_translation_only(targets: list[GraspTarget | None], object_centroid: np.ndarray,
                            min_contacts: int = 3) -> list[GraspTarget | None]:
    """Simplified (iv) used for the qualitative figures: after the onset, fingertip frames that still touch the
    object keep the onset gripper-to-object offset and orientation and follow the object centroid."""
    onset = next((t for t, g in enumerate(targets) if g is not None and g.mode == MODE_CONTACT), None)
    if onset is None:
        return list(targets)
    out = list(targets)
    for t in range(onset + 1, len(targets)):
        g = targets[t]
        if g is not None and g.mode == MODE_FINGERTIP and g.n_finger >= min_contacts:
            p = targets[onset].p + (object_centroid[t] - object_centroid[onset])
            out[t] = GraspTarget(p, targets[onset].R.copy(), targets[onset].width, MODE_ANCHORED, g.n_thumb, g.n_finger)
    return out


def label_consistency(T_WE: np.ndarray, T_WO: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Std of the gripper-object transform T^E_O over the masked frames: (translation mm, rotation deg)."""
    T_EO = np.stack([invert_T(T_WE[t]) @ T_WO[t] for t in np.flatnonzero(mask)])
    if len(T_EO) < 2:
        return 0.0, 0.0
    t_std = float(1000 * np.sqrt(((T_EO[:, :3, 3] - T_EO[:, :3, 3].mean(0)) ** 2).sum(1).mean()))
    R_mean = rotation_mean(T_EO[:, :3, :3])
    angles = np.array([rotation_angle_deg(R_mean, R) for R in T_EO[:, :3, :3]])
    return t_std, float(np.sqrt((angles ** 2).mean()))


# --------------------------------------------------------------------------------------------- actions
def rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2 * np.sin(angle))


def delta_actions(T_WE: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """a_t = [dp (3, base frame), drot (3, axis-angle in the base frame), g_{t+1}]; last action repeats zero motion."""
    T = len(T_WE)
    a = np.zeros((T, 7))
    for t in range(T - 1):
        a[t, :3] = T_WE[t + 1, :3, 3] - T_WE[t, :3, 3]
        a[t, 3:6] = rotvec(T_WE[t + 1, :3, :3] @ T_WE[t, :3, :3].T)
        a[t, 6] = gripper[t + 1]
    a[-1, 6] = gripper[-1]
    return a


def fill_gaps(targets: list[GraspTarget | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate poses across short gaps (no hand detected) so every frame between the first and last target
    has a pose. Returns T_WE (T,4,4), width (T,), valid (T,)."""
    T = len(targets)
    have = [t for t, g in enumerate(targets) if g is not None]
    T_WE, width, valid = np.tile(np.eye(4), (T, 1, 1)), np.zeros(T), np.zeros(T, bool)
    if not have:
        return T_WE, width, valid
    for t in range(have[0], have[-1] + 1):
        if targets[t] is not None:
            T_WE[t], width[t] = targets[t].T, targets[t].width
        else:
            a = max(i for i in have if i < t)
            b = min(i for i in have if i > t)
            s = (t - a) / (b - a)
            T_WE[t] = make_T(slerp(targets[a].R, targets[b].R, s), (1 - s) * targets[a].p + s * targets[b].p)
            width[t] = (1 - s) * targets[a].width + s * targets[b].width
        valid[t] = True
    return T_WE, width, valid


def compute_actions(run: Path, cfg: dict) -> dict:
    from recast.io.clip import Clip, read_mask
    from recast.perception.contact import object_points
    from recast.perception.hands import load_hands

    clip = Clip(run / "clip")
    hands = load_hands(run)
    with np.load(run / "keypoints" / "keypoints.npz") as f:
        T_WO, T_wc = f["T_WO"], f["T_wc"]
    ph = json.loads((run / "phases" / "phases.json").read_text())
    from recast import PHASES

    phases = np.array([PHASES.index(p) for p in ph["phases"]])
    gripper = np.array(ph["gripper"])
    cc, ac = cfg["contact"], cfg["actions"]
    targets: list[GraspTarget | None] = []
    for t in range(len(clip)):
        if not hands["present"][t]:
            targets.append(None)
            continue
        verts, joints = transform_points(T_wc, hands["verts"][t]), transform_points(T_wc, hands["joints"][t])
        obj = transform_points(T_wc, object_points(clip.depth(t), read_mask(run / "masks" / "object" /
                                                                             f"{clip.names[t]}.png"), clip.k))
        in_contact_phase = phases[t] in (GRASP, CARRY)
        targets.append(grasp_from_hand(verts, joints, obj if in_contact_phase else np.zeros((0, 3)),
                                       cc["eps_m"], cc["min_contacts"]))
    targets = blend_transitions(targets, ac["blend_frames"])
    before = fill_gaps(targets)[0]
    targets = anchor(targets, phases, T_WO, hands["icp_rmse"], ac["anchor_sigma_m"])
    T_WE, width, valid = fill_gaps(targets)
    carry = (phases == CARRY) & valid
    std_before, std_after = label_consistency(before, T_WO, carry), label_consistency(T_WE, T_WO, carry)
    keep = std_after[0] <= ac["max_anchor_std_mm"]
    mode = np.array([MODE_NONE if g is None else g.mode for g in targets])
    (run / "actions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / "actions" / "actions.npz", T_WE=T_WE, width=width, gripper=gripper, phase=phases,
                        mode=mode, valid=valid, action=delta_actions(T_WE, gripper))
    summary = {"frames_with_pose": int(valid.sum()), "modes": {MODE_NAMES[m]: int((mode == m).sum()) for m in range(4)},
               "T_EO_std_before": {"mm": std_before[0], "deg": std_before[1]},
               "T_EO_std_after": {"mm": std_after[0], "deg": std_after[1]}, "kept": bool(keep)}
    (run / "actions" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    from recast.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    print(json.dumps(compute_actions(a.run, load_config(a.task)), indent=2))


if __name__ == "__main__":
    main()
