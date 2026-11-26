import numpy as np

from recast import geometry as g


def random_T(rng):
    q = rng.normal(size=4)
    return g.make_T(g.quat_to_mat(q), rng.normal(size=3))


def test_kabsch_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    T = random_T(rng)
    src = rng.normal(size=(20, 3))
    assert np.allclose(g.kabsch(src, g.transform_points(T, src)), T, atol=1e-9)


def test_quaternion_roundtrip_and_slerp_endpoints():
    rng = np.random.default_rng(1)
    for _ in range(20):
        R0, R1 = random_T(rng)[:3, :3], random_T(rng)[:3, :3]
        assert np.allclose(g.quat_to_mat(g.mat_to_quat(R0)), R0, atol=1e-9)
        assert np.allclose(g.slerp(R0, R1, 0.0), R0, atol=1e-6)
        assert np.allclose(g.slerp(R0, R1, 1.0), R1, atol=1e-6)
        half = g.slerp(R0, R1, 0.5)
        assert abs(g.rotation_angle_deg(R0, half) - g.rotation_angle_deg(half, R1)) < 1e-4


def test_invert_and_rotation_mean():
    rng = np.random.default_rng(2)
    T = random_T(rng)
    assert np.allclose(T @ g.invert_T(T), np.eye(4), atol=1e-12)
    R = T[:3, :3]
    assert np.allclose(g.rotation_mean(np.stack([R, R, R])), R, atol=1e-9)


def test_farthest_point_sampling_spreads_points():
    pts = np.concatenate([np.zeros((500, 3)), np.eye(3)])
    idx = g.farthest_point_sampling(pts, 4, seed=0)
    assert len(set(map(tuple, pts[idx].round(6)))) == 4


def test_projection_roundtrip_and_solve_translation():
    k = {"fx": 600.0, "fy": 610.0, "cx": 320.0, "cy": 240.0}
    rng = np.random.default_rng(3)
    verts = rng.normal(scale=0.03, size=(100, 3))
    t = np.array([0.05, -0.02, 0.6])
    uv = g.project(verts + t, k)
    assert np.allclose(g.solve_translation(verts, uv, k), t, atol=1e-6)
    depth = np.zeros((480, 640), np.uint16)
    depth[240, 320] = 750
    pts, ok = g.lift_pixels(np.array([[320.0, 240.0], [10.0, 10.0]]), depth, k)
    assert ok.tolist() == [True, False] and np.allclose(pts[0], [0, 0, 0.75])
