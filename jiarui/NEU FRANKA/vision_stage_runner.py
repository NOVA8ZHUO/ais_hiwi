#!/usr/bin/env python3
"""Process lifecycle helper for one RealSense + one vision-node stage."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence


HERE = Path(__file__).resolve().parent


def _stop_process(process: subprocess.Popen[bytes] | None, label: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {label} (pid={process.pid})...", flush=True)
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=4.0)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        print(f"Warning: could not stop {label}", flush=True)


def _wait_for_camera_parameter_service(
    camera_process: subprocess.Popen[bytes],
    camera_node: str,
    timeout_sec: float,
) -> None:
    try:
        import rclpy
        from rcl_interfaces.srv import SetParameters
        from rclpy.node import Node
    except ImportError as exc:
        raise RuntimeError("ROS2 Python packages are required") from exc

    rclpy.init(args=None)
    probe = Node("vision_stage_camera_service_probe")
    client = probe.create_client(
        SetParameters, f"/{camera_node.strip('/')}/set_parameters"
    )
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    try:
        while time.monotonic() < deadline:
            return_code = camera_process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"realsense2_camera exited before becoming ready: {return_code}"
                )
            if client.wait_for_service(timeout_sec=0.25):
                return
        raise TimeoutError(
            f"Timed out waiting for /{camera_node.strip('/')}/set_parameters"
        )
    finally:
        probe.destroy_node()
        rclpy.shutdown()


def run_stage(
    stage: str,
    params_file: Path,
    camera_node: str = "/camera/camera",
    camera_startup_timeout_sec: float = 20.0,
    target_part: str | None = None,
    base_hole_target: str = "hole_large",
    camera_extra_args: Sequence[str] = (),
) -> int:
    """Run a stage until Ctrl-C or either child process exits."""
    if stage not in {"global", "local"}:
        raise ValueError("stage must be 'global' or 'local'")
    if stage == "local" and not target_part:
        raise ValueError("target_part is required for the local stage")
    params_file = params_file.expanduser().resolve()
    if not params_file.is_file():
        raise FileNotFoundError(params_file)
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 executable not found; source the ROS2 environment")

    camera_command = [
        ros2,
        "launch",
        "realsense2_camera",
        "rs_launch.py",
        "align_depth.enable:=true",
        "enable_sync:=true",
        "rgb_camera.color_profile:=640x480x30",
        "depth_module.depth_profile:=640x480x30",
        *camera_extra_args,
    ]
    node_script = (
        HERE / "global_rgb_part_detection_node.py"
        if stage == "global"
        else HERE / "local_rgbd_grasp_refinement_node.py"
    )
    vision_command = [
        sys.executable,
        str(node_script),
        "--ros-args",
        "--params-file",
        str(params_file),
        "-p",
        f"realsense_camera_node:={camera_node}",
    ]
    if stage == "local":
        vision_command.extend(
            [
                "-p",
                f"target_part:={target_part}",
                "-p",
                f"base_hole_target:={base_hole_target}",
            ]
        )

    camera_process: subprocess.Popen[bytes] | None = None
    vision_process: subprocess.Popen[bytes] | None = None
    stopping = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stopping
        print(f"Received signal {signum}; stopping {stage} stage...", flush=True)
        stopping = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        print("Starting RealSense:", " ".join(camera_command), flush=True)
        camera_process = subprocess.Popen(camera_command, start_new_session=True)
        _wait_for_camera_parameter_service(
            camera_process, camera_node, camera_startup_timeout_sec
        )
        print("Starting vision node:", " ".join(vision_command), flush=True)
        vision_process = subprocess.Popen(vision_command, start_new_session=True)
        while not stopping:
            camera_code = camera_process.poll()
            vision_code = vision_process.poll()
            if camera_code is not None:
                print(f"RealSense exited with code {camera_code}", flush=True)
                return camera_code or 1
            if vision_code is not None:
                print(f"Vision node exited with code {vision_code}", flush=True)
                return vision_code or 1
            time.sleep(0.2)
        return 0
    finally:
        _stop_process(vision_process, f"{stage} vision node")
        _stop_process(camera_process, "RealSense")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

