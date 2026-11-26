import numpy as np
import pytest

from recast.config import load_config
from recast.sim.scene import load_scene

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("task", ["cube", "tape", "battery"])
def test_scripted_pick_and_place_succeeds(task, panda_dir):
    from recast.sim.scripted import run

    res = run(load_scene(load_config(task)["task"]), panda_dir, record=False)
    assert res["success"], res


@pytest.mark.parametrize("variant", ["drop", "miss"])
def test_failure_variants_fail(variant, panda_dir):
    from recast.sim.scripted import run

    assert not run(load_scene(load_config("cube")["task"]), panda_dir, variant, record=False)["success"]


def test_osc_tracks_delta_actions(panda_dir):
    from recast.sim.env import PandaSim

    sim = PandaSim(load_scene(load_config("cube")["task"]), panda_dir, control="torque")
    x0 = sim.tcp_pose()[0, 3]
    for _ in range(20):
        sim.apply_action(np.array([0.005, 0, 0, 0, 0, 0]), 1.0, 0.1)
    assert abs((sim.tcp_pose()[0, 3] - x0) - 0.1) < 0.01


def test_conditions_build(panda_dir):
    from recast.config import load_yaml
    from recast.sim.env import PandaSim

    for name, cond in load_yaml("eval.yaml")["conditions"].items():
        sim = PandaSim(load_scene(load_config("cube")["task"], cond), panda_dir, seed=1)
        assert sim.render("rs").shape == (480, 640, 3), name
        sim.close()
