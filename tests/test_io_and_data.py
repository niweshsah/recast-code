import numpy as np

from recast.data.episodes import split_state, state_vector
from recast.io.bag import CameraInfo, align_depth_to_color, nearest_pairs, parse_tf
from recast.io.clip import Clip


def test_parse_tf_column_major():
    R, t = parse_tf("rotation=1,0,0,0,0,1,0,-1,0;translation=0.1,0.2,0.3")
    assert np.allclose(R, [[1, 0, 0], [0, 0, -1], [0, 1, 0]]) and np.allclose(t, [0.1, 0.2, 0.3])


def test_nearest_pairs():
    assert nearest_pairs(np.array([0, 10, 21]), np.array([1, 9, 20, 30])).tolist() == [0, 1, 2]


def test_align_identity_keeps_depth():
    k = CameraInfo(500.0, 500.0, 32.0, 24.0, 64, 48)
    raw = np.full((48, 64), 800, np.uint16)
    out = align_depth_to_color(raw, k, k, np.eye(3), np.zeros(3), 0.001, splat=1)
    assert np.allclose(out, 0.8)


def test_state_vector_roundtrip():
    q, pts = np.arange(7.0), np.random.default_rng(0).normal(size=(16, 3))
    s = state_vector(q, 0.5, pts)
    proprio, p2 = split_state(s[None], 16)
    assert np.allclose(proprio[0], [*q, 0.5]) and np.allclose(p2[0], pts, atol=1e-6)


def test_sample_clip_loads(sample_run):
    clip = Clip(sample_run / "clip")
    assert len(clip) == 37 and clip.rgb(0).shape == (480, 640, 3) and clip.depth(0).dtype == np.uint16
    assert abs(clip.fps - 7.5) < 0.1
