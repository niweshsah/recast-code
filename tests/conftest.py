import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "cube_demo"


def menagerie_panda() -> Path | None:
    root = Path(os.environ.get("RECAST_MENAGERIE", ROOT / "third_party" / "mujoco_menagerie"))
    p = root / "franka_emika_panda"
    return p if (p / "panda.xml").exists() else None


@pytest.fixture
def sample_run(tmp_path) -> Path:
    run = tmp_path / "cube_demo"
    shutil.copytree(SAMPLE, run)
    return run


@pytest.fixture
def panda_dir() -> Path:
    pytest.importorskip("mujoco")
    p = menagerie_panda()
    if p is None:
        pytest.skip("MuJoCo Menagerie not found (set RECAST_MENAGERIE or run scripts/setup_gpu.sh --menagerie-only)")
    return p
