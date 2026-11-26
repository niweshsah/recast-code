"""CPU stages end to end on the committed sample (WiLoR, masks and inpainting outputs are included)."""

import json

import numpy as np

from recast.actions import compute_actions
from recast.config import load_config
from recast.perception.contact import compute_signals
from recast.perception.keypoints import track_object
from recast.phases import PhaseHMM, _signals, gripper_labels


def test_sample_pipeline(sample_run):
    T_wc = np.array(json.loads((sample_run / "extrinsics.json").read_text())["T_world_cam"])
    track_object(sample_run, 16, T_wc, tracker="lk")
    compute_signals(sample_run)
    path = PhaseHMM.prior().viterbi(_signals(sample_run))
    from recast import PHASES

    (sample_run / "phases").mkdir()
    (sample_run / "phases" / "phases.json").write_text(json.dumps(
        {"phases": [PHASES[s] for s in path], "gripper": gripper_labels(path).tolist()}))
    names = [PHASES[s] for s in path]
    assert names[0] == "idle" and {"approach", "grasp", "carry", "release"} <= set(names)
    assert names.index("grasp") < names.index("carry") < names.index("release")
    summary = compute_actions(sample_run, load_config("cube"))
    assert summary["frames_with_pose"] >= 20
    assert summary["T_EO_std_after"]["mm"] < summary["T_EO_std_before"]["mm"]
