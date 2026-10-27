# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the calibrated wrist camera and show/save its RGB view."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--warmup_steps", type=int, default=12)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "outputs" / "wrist_camera_rgb.png",
)
parser.add_argument("--keep_open", action="store_true", help="Keep the GUI open on the wrist-camera view.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# RTX cameras must be explicitly enabled in Isaac Lab applications.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import forge_shaft_insert  # noqa: E402, F401
from isaaclab.sensors import save_images_to_file  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK_NAME = "FORGE-Shaft-Insert-Camera-v0"


def main() -> None:
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
    env = gym.make(TASK_NAME, cfg=env_cfg)
    observation, _ = env.reset(seed=42)
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # Render enough frames for Replicator to populate both annotators.
    for _ in range(args_cli.warmup_steps):
        observation, _, _, _, _ = env.step(zero_action)

    camera = env.unwrapped.scene["wrist_camera"]
    rgb = camera.data.output["rgb"]
    depth = camera.data.output["distance_to_image_plane"]
    intrinsic = camera.data.intrinsic_matrices[0]
    camera_path = str(camera._sensor_prims[0].GetPath())

    calibrated_k = torch.tensor(
        [
            [env_cfg.wrist_camera.fx, 0.0, env_cfg.wrist_camera.cx],
            [0.0, env_cfg.wrist_camera.fy, env_cfg.wrist_camera.cy],
            [0.0, 0.0, 1.0],
        ],
        device=intrinsic.device,
        dtype=intrinsic.dtype,
    )
    # Isaac Sim 5.1's USD pinhole camera cannot represent non-square pixels or
    # an off-center principal point. Isaac Lab therefore uses mean(fx, fy) and
    # the image center. Validate that explicit, documented approximation.
    effective_focal = 0.5 * (env_cfg.wrist_camera.fx + env_cfg.wrist_camera.fy)
    expected_runtime_k = torch.tensor(
        [
            [effective_focal, 0.0, 0.5 * env_cfg.wrist_camera.width],
            [0.0, effective_focal, 0.5 * env_cfg.wrist_camera.height],
            [0.0, 0.0, 1.0],
        ],
        device=intrinsic.device,
        dtype=intrinsic.dtype,
    )
    torch.testing.assert_close(intrinsic, expected_runtime_k, rtol=1.0e-5, atol=1.0e-3)
    assert rgb.shape == (1, env_cfg.wrist_camera.height, env_cfg.wrist_camera.width, 3), rgb.shape
    assert depth.shape == (1, env_cfg.wrist_camera.height, env_cfg.wrist_camera.width, 1), depth.shape
    assert torch.isfinite(rgb.float()).all(), "RGB image contains NaN or Inf."

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    save_images_to_file(rgb[..., :3].float() / 255.0, str(args_cli.output))

    finite_depth = depth[torch.isfinite(depth)]
    depth_range = (
        (float(finite_depth.min().item()), float(finite_depth.max().item()))
        if finite_depth.numel() > 0
        else (float("nan"), float("nan"))
    )
    print(f"[camera] prim_path={camera_path}")
    print(
        f"[camera] resolution={camera.cfg.width}x{camera.cfg.height} "
        f"fps={1.0 / camera.cfg.update_period:.1f} data_types={camera.cfg.data_types}"
    )
    print(f"[camera] calibrated_K={calibrated_k.detach().cpu().tolist()}")
    print(f"[camera] runtime_effective_K={intrinsic.detach().cpu().tolist()}")
    print("[camera] note=Isaac Sim 5.1 centers the principal point and averages fx/fy.")
    print(f"[camera] world_xyz_m={camera.data.pos_w[0].detach().cpu().tolist()}")
    print(f"[camera] world_quat_ros_wxyz={camera.data.quat_w_ros[0].detach().cpu().tolist()}")
    print(f"[camera] rgb_shape={tuple(rgb.shape)} depth_shape={tuple(depth.shape)}")
    print(f"[camera] finite_depth_range_m={depth_range}")
    print(f"[camera] saved_rgb={args_cli.output}")
    print(f"[camera] policy_observation_shape={tuple(observation['policy'].shape)} (still dx,dy,dz)")

    if not args_cli.headless:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(camera_path)
            print("[camera] GUI viewport switched to the calibrated wrist camera.")

    if args_cli.keep_open:
        print("[camera] Keep-open active. Close Isaac Sim or press Ctrl+C to exit.")
        while simulation_app.is_running():
            env.step(zero_action)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
