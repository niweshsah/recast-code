import numpy as np

from recast.geometry import invert_T, make_T, quat_to_mat, transform_points
from recast.perception.keypoints import object_poses, object_state, propagate_rigid, track_lk


def test_propagate_rigid_fills_occluded_points():
    rng = np.random.default_rng(0)
    K0 = rng.normal(scale=0.03, size=(16, 3))
    poses = [make_T(quat_to_mat(np.array([1, 0.05 * t, 0, 0.02 * t])), [0.01 * t, 0, 0.5]) for t in range(10)]
    truth = np.stack([transform_points(T, K0) for T in poses])
    observed = rng.uniform(size=(10, 16)) > 0.4
    observed[0] = True
    noisy = np.where(observed[..., None], truth, 99.0)          # unobserved entries are garbage
    K = propagate_rigid(noisy, observed)
    assert np.abs(K - truth).max() < 1e-9
    P = object_poses(K)
    assert np.allclose(transform_points(P[5] @ invert_T(P[0]), K[0]), K[5], atol=1e-9)


def test_object_state_in_end_effector_frame():
    K = np.array([[0.5, 0.0, 0.1]])
    T_WE = make_T(np.eye(3), [0.5, 0.0, 0.2])
    assert np.allclose(object_state(K, T_WE), [[0, 0, -0.1]])


def test_lk_tracks_translating_texture():
    rng = np.random.default_rng(0)
    tex = (rng.uniform(size=(240, 320)) * 255).astype(np.uint8)
    import cv2

    tex = cv2.GaussianBlur(tex, (5, 5), 1.5)
    frames = np.stack([np.repeat(np.roll(tex, 2 * t, axis=1)[..., None], 3, -1) for t in range(6)])
    q = np.array([[100.0, 100.0], [150.0, 120.0], [200.0, 80.0]])
    uv, vis = track_lk(frames, q, t0=0)
    assert vis[1:].all() and np.allclose(uv[5] - q, [10, 0], atol=0.5)
