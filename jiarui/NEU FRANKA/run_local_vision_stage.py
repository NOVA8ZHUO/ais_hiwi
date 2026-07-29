#!/usr/bin/env python3
"""Start and stop RealSense together with the Local vision node."""

from __future__ import annotations

import argparse
from pathlib import Path

from vision_stage_runner import HERE, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-part", required=True)
    parser.add_argument(
        "--base-hole-target",
        choices=("hole_large", "hole_small", "hole_screw"),
        default="hole_large",
    )
    parser.add_argument("--params-file", type=Path, default=HERE / "vision_config.yaml")
    parser.add_argument("--camera-node", default="/camera/camera")
    parser.add_argument("--camera-startup-timeout-sec", type=float, default=20.0)
    parser.add_argument("--camera-arg", action="append", default=[])
    args = parser.parse_args()
    return run_stage(
        "local",
        args.params_file,
        camera_node=args.camera_node,
        camera_startup_timeout_sec=args.camera_startup_timeout_sec,
        target_part=args.target_part,
        base_hole_target=args.base_hole_target,
        camera_extra_args=args.camera_arg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
