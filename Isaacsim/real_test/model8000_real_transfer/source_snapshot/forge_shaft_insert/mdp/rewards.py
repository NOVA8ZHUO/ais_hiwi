# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import position_error
from .terminations import xy_alignment_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def xy_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
) -> torch.Tensor:
    """Reward reduction in XY error; moving away receives the symmetric penalty."""
    current_error = torch.linalg.vector_norm(position_error(env, asset_cfg)[:, :2], dim=-1)
    previous_error = env._forge_previous_xy_error
    progress = (previous_error - current_error) / env.cfg.task.xy_progress_scale
    env._forge_previous_xy_error = current_error.detach()
    return torch.clamp(progress, min=-1.0, max=1.0)


def action_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared normalized Cartesian action magnitude."""
    return torch.sum(torch.square(env.action_manager.action), dim=-1)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return one when the shaft is horizontally aligned with the hole."""
    return xy_alignment_success(env).to(dtype=torch.float32)
