#!/usr/bin/env python3
"""ROS2 entry point for the notebook-aligned global RGB part detector.

This file intentionally keeps the ROS interface compatible with
global_vision_force_adapter.py while exposing a node script name that matches
global_rgb_part_detection.ipynb.
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _path in (_THIS_DIR, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from .global_vision_force_adapter import (  # type: ignore[import-not-found] # noqa: F401
        GlobalDetection,
        GlobalPartDetector,
        GlobalVisionForceAdapter,
        main,
    )
except ImportError:
    from global_vision_force_adapter import (  # noqa: F401
        GlobalDetection,
        GlobalPartDetector,
        GlobalVisionForceAdapter,
        main,
    )


if __name__ == "__main__":
    main()
