"""YAML configuration: ``configs/default.yaml`` deep-merged with a task file and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in extra.items():
        out[key] = _merge(out[key], val) if isinstance(val, dict) and isinstance(out.get(key), dict) else val
    return out


def load_config(task: str | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Defaults + ``configs/tasks/<task>.yaml`` (under the ``task`` key) + dotted-key overrides."""
    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    if task is not None:
        cfg["task"] = yaml.safe_load((CONFIGS / "tasks" / f"{task}.yaml").read_text())
    for dotted, val in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    return cfg


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())
