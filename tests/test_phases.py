import numpy as np
import pytest

from recast.phases import CARRY, GRASP, IDLE, RELEASE, PhaseHMM, frame_f1, gripper_labels, segments


def synthetic(seed, lengths=(12, 8, 6, 15, 5, 10)):
    """One interaction cycle sampled from the prior's emissions."""
    rng = np.random.default_rng(seed)
    prior = PhaseHMM.prior()
    states = np.concatenate([np.full(n, s % 5) for s, n in enumerate(lengths)])
    x = prior.means[states] + rng.normal(size=(len(states), 4)) * np.sqrt(prior.variances[states]) * 0.5
    return x, states


def test_viterbi_recovers_phases_from_prior():
    x, y = synthetic(0)
    assert frame_f1(PhaseHMM.prior().viterbi(x), y) > 0.9


def test_fit_and_decode_held_out():
    train = [synthetic(s) for s in range(1, 11)]
    hmm = PhaseHMM.fit([x for x, _ in train], [y for _, y in train])
    x, y = synthetic(99, lengths=(5, 10, 4, 20, 6, 3))
    assert frame_f1(hmm.viterbi(x), y) > 0.9
    assert PhaseHMM.from_json(hmm.to_json()).viterbi(x).tolist() == hmm.viterbi(x).tolist()


def test_fit_rejects_forbidden_transition():
    x, _ = synthetic(0)
    bad = np.array([IDLE] * 10 + [CARRY] * (len(x) - 10))
    with pytest.raises(ValueError):
        PhaseHMM.fit([x], [bad])


def test_gripper_labels_eq3():
    ph = np.array([IDLE, IDLE, GRASP, CARRY, CARRY, RELEASE, RELEASE, RELEASE, RELEASE, IDLE])
    g = gripper_labels(ph)
    assert g[:2].tolist() == [1, 1] and g[2:5].tolist() == [0, 0, 0]
    assert np.allclose(g[5:9], [0.25, 0.5, 0.75, 1.0]) and g[9] == 1
    assert segments(ph)[0] == (IDLE, 0, 1)
