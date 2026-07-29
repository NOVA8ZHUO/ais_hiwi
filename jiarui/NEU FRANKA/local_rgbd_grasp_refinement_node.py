#!/usr/bin/env python3
"""ROS2 entry point for the notebook-aligned local RGB-D grasp refiner.

This file intentionally keeps the ROS interface compatible with
local_vision_force_adapter.py while exposing a node script name that matches
local_rgbd_grasp_refinement.ipynb.
"""

from __future__ import annotations

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from .local_vision_force_adapter import (  # type: ignore[import-not-found] # noqa: F401
        LocalDetection,
        LocalGraspDetector,
        LocalVisionForceAdapter,
        main,
    )
except ImportError:
    from local_vision_force_adapter import (  # noqa: F401
        LocalDetection,
        LocalGraspDetector,
        LocalVisionForceAdapter,
        main,
    )


if __name__ == "__main__":
    main()
