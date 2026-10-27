# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_grasp_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    shaft_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
    gripper_command: float = 0.0,
) -> None:
    """Restore the validated physical grasp while leaving the Base untouched."""
    robot: Articulation = env.scene[robot_cfg.name]
    shaft: RigidObject = env.scene[shaft_cfg.name]

    # Robot pose measured after franka_diff_ik.py reached fingertip target
    # (0.5, 0, 0.21), RPY=(180, 0, 0) degrees.
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    robot.write_joint_position_to_sim(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim(velocity=joint_vel, env_ids=env_ids)

    arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]", preserve_order=True)
    finger_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
    robot.set_joint_position_target(joint_pos[:, arm_joint_ids], joint_ids=arm_joint_ids, env_ids=env_ids)
    # Start from the measured contact-limited finger positions but command zero
    # opening.  Contact/friction—not a fixed joint—holds the independent shaft.
    finger_target = torch.full(
        (len(env_ids), len(finger_joint_ids)),
        gripper_command,
        device=env.device,
        dtype=joint_pos.dtype,
    )
    robot.set_joint_position_target(finger_target, joint_ids=finger_joint_ids, env_ids=env_ids)

    shaft_state = shaft.data.default_root_state[env_ids].clone()
    shaft_state[:, :3] += env.scene.env_origins[env_ids]
    shaft_state[:, 7:] = 0.0
    shaft.write_root_pose_to_sim(shaft_state[:, :7], env_ids=env_ids)
    shaft.write_root_velocity_to_sim(shaft_state[:, 7:], env_ids=env_ids)

    # Initialize the stateful progress reward from the exact reset geometry.
    hole_world = env.scene["base"].data.root_pos_w[env_ids] + torch.tensor(
        env.cfg.task.hole_offset_in_base,
        device=env.device,
        dtype=shaft_state.dtype,
    )
    reset_xy_error = torch.linalg.vector_norm(shaft_state[:, :2] - hole_world[:, :2], dim=-1)
    if not hasattr(env, "_forge_previous_xy_error"):
        env._forge_previous_xy_error = torch.zeros(env.num_envs, device=env.device, dtype=shaft_state.dtype)
    env._forge_previous_xy_error[env_ids] = reset_xy_error
