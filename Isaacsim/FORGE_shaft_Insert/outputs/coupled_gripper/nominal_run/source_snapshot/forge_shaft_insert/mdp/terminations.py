# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import position_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def xy_alignment_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
) -> torch.Tensor:
    """Terminate when the Shaft_large origin is horizontally centered over the hole."""
    error = position_error(env, asset_cfg)
    xy_error = torch.linalg.vector_norm(error[:, :2], dim=-1)
    return xy_error <= env.cfg.task.success_xy_tolerance


def outside_workspace(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
) -> torch.Tensor:
    """Terminate when Shaft_large leaves the configured task workspace."""
    error = position_error(env, asset_cfg)
    xy_error = torch.linalg.vector_norm(error[:, :2], dim=-1)
    initial_dz = env.cfg.task.nominal_tcp_height_above_hole
    return (xy_error > env.cfg.task.workspace_xy_limit) | (
        torch.abs(error[:, 2] - initial_dz) > env.cfg.task.workspace_z_tolerance
    )
