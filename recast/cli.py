"""``recast <command> [args]``: thin dispatcher over the stage modules (each also runs with ``python -m``)."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "export": "recast.io.bag",
    "segment": "recast.perception.segmentation",
    "hands": "recast.perception.hands",
    "keypoints": "recast.perception.keypoints",
    "contact": "recast.perception.contact",
    "phases": "recast.phases",
    "actions": "recast.actions",
    "inpaint": "recast.editing.inpaint",
    "render": "recast.editing.render",
    "episodes": "recast.data.episodes",
    "scripted": "recast.sim.scripted",
    "evaluate": "recast.sim.evaluate",
    "panels": "recast.viz.panels",
    "video": "recast.viz.video",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: recast <command> [args]\n\ncommands:\n" + "\n".join(f"  {c:10s} {m}" for c, m in COMMANDS.items()))
        sys.exit(0 if len(sys.argv) < 2 else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"recast {cmd}"
    importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
