# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class AccumulatingCartesianDeltaPositionAction(DifferentialInverseKinematicsAction):
    """Accumulate Cartesian deltas into a persistent position target.

    Isaac Lab's standard relative Diff-IK term rebuilds its target from the
    current pose every policy step.  Under gripper load that makes zero action
    accept small compliance errors as the next target and the arm creeps.  This
    term instead integrates each normalized delta into the previous desired
    position, so a zero action actively holds the last Cartesian target.
    """

    def __init__(self, cfg, env):
        if cfg.controller.command_type != "position" or cfg.controller.use_relative_mode:
            raise ValueError(
                "AccumulatingCartesianDeltaPositionAction requires absolute position Diff-IK "
                "(command_type='position', use_relative_mode=False)."
            )
        super().__init__(cfg, env)
        self._desired_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._initialize_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.clamp(actions, min=-1.0, max=1.0)
        self._processed_actions[:] = self._raw_actions * self._scale

        current_position, current_quaternion = self._compute_frame_pose()
        if self._initialize_target.any():
            self._desired_position[self._initialize_target] = current_position[self._initialize_target]
            self._initialize_target[:] = False

        self._desired_position += self._processed_actions
        self._ik_controller.set_command(self._desired_position, ee_quat=current_quaternion)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._initialize_target[:] = True
        else:
            self._initialize_target[env_ids] = True
