import numpy as np

from recast import actions as A
from recast.geometry import make_T, quat_to_mat, slerp, transform_points
from recast.phases import CARRY, GRASP, IDLE


def pinch_hand(width=0.06):
    """Toy 21-joint hand pinching along +y: thumb chain near y=0, other fingers near y=width."""
    joints = np.zeros((21, 3))
    joints[0] = [0.0, width / 2, 0.12]                          # wrist above
    joints[1:5] = [[0.0, 0.0, 0.04 - 0.01 * i] for i in range(4)]
    for f in range(4):
        joints[5 + 4 * f:9 + 4 * f] = [[0.01 * (f - 1.5), width, 0.04 - 0.01 * i] for i in range(4)]
    verts = np.concatenate([joints[1:5] + [0, -0.002, 0], joints[5:] + [0, 0.002, 0]])
    return verts, joints


def test_contact_grasp_matches_eq4():
    verts, joints = pinch_hand(0.06)
    cube = np.random.default_rng(0).uniform([-0.02, 0.003, 0.0], [0.02, 0.057, 0.04], (3000, 3))
    g = A.grasp_from_hand(verts, joints, cube, eps=0.01, min_contacts=3)
    assert g.mode == A.MODE_CONTACT
    assert abs(g.width - 0.064) < 0.01                          # thumb/finger contact centroids ~6.4 cm apart
    assert np.allclose(g.R @ g.R.T, np.eye(3), atol=1e-9) and abs(g.R[:, 1] @ [0, 1, 0]) > 0.99


def test_fingertip_fallback_without_object():
    verts, joints = pinch_hand()
    g = A.grasp_from_hand(verts, joints, np.zeros((0, 3)))
    assert g.mode == A.MODE_FINGERTIP and np.allclose(g.p, 0.5 * (joints[4] + joints[8]))


def test_anchoring_follows_rigid_object_when_hand_is_unreliable():
    T = 20
    phases = np.array([IDLE] * 4 + [GRASP] * 4 + [CARRY] * 12)
    T_WO = np.stack([make_T(quat_to_mat(np.array([1, 0, 0.02 * t, 0])), [0.01 * t, 0, 0.1]) for t in range(T)])
    T_EO = make_T(np.eye(3), [0, 0, 0.05])
    truth = [T_WO[t] @ np.linalg.inv(T_EO) for t in range(T)]
    rng = np.random.default_rng(0)
    targets = []
    for t in range(T):
        noisy = truth[t][:3, 3] + rng.normal(scale=0.02 if t > 4 else 0.0, size=3)
        targets.append(A.GraspTarget(noisy, truth[t][:3, :3], 0.05, A.MODE_CONTACT if t == 4 else A.MODE_FINGERTIP))
    rmse = np.full(T, 0.05)                                      # poor ICP fit: lambda = exp(-5) ~ 0
    out = A.anchor(targets, phases, T_WO, rmse, sigma=0.01)
    T_before = np.stack([g.T for g in targets])
    T_after = np.stack([g.T for g in out])
    carry = phases == CARRY
    assert A.label_consistency(T_after, T_WO, carry)[0] < 0.2 * A.label_consistency(T_before, T_WO, carry)[0]


def test_delta_actions_integrate_back():
    T = np.stack([make_T(slerp(np.eye(3), quat_to_mat(np.array([0.9, 0.1, 0.2, 0.3])), s), [s, 2 * s, 0])
                  for s in np.linspace(0, 1, 6)])
    a = A.delta_actions(T, np.linspace(1, 0, 6))
    assert np.allclose(np.cumsum(a[:-1, :3], 0)[-1], T[-1, :3, 3] - T[0, :3, 3])
    assert a.shape == (6, 7) and a[0, 6] == 0.8


def test_fill_gaps_interpolates():
    g = [A.GraspTarget(np.zeros(3), np.eye(3), 0.04, 1), None, A.GraspTarget(np.ones(3), np.eye(3), 0.06, 1)]
    T_WE, w, valid = A.fill_gaps(g)
    assert valid.all() and np.allclose(T_WE[1][:3, 3], 0.5) and np.isclose(w[1], 0.05)
    assert np.allclose(transform_points(T_WE[1], np.zeros((1, 3))), 0.5)
