# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run Isaac Lab 2.3.2's RSL-RL player with the FORGE task registered."""

import os
import sys
from pathlib import Path


isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2"))
player = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
if not player.is_file():
    raise FileNotFoundError(f"Isaac Lab 2.3.2 player not found: {player}")

sys.path.insert(0, str(player.parent))
source = player.read_text(encoding="utf-8")
registration_anchor = "import isaaclab_tasks  # noqa: F401"
if registration_anchor not in source:
    raise RuntimeError("Isaac Lab player layout changed: task registration anchor not found.")
source = source.replace(
    registration_anchor,
    registration_anchor + "\nimport forge_shaft_insert  # noqa: F401",
    1,
)
namespace = {"__name__": "__main__", "__file__": str(player)}
exec(compile(source, str(player), "exec"), namespace)
