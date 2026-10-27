# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def shaft_position_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
) -> torch.Tensor:
    """Return the independent Shaft_large rigid-body origin in world coordinates [m]."""
    shaft: RigidObject = env.scene[asset_cfg.name]
    return shaft.data.root_pos_w


def hole_target_position_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the hole entrance target in world coordinates [m]."""
    base: RigidObject = env.scene["base"]
    offset = torch.tensor(env.cfg.task.hole_offset_in_base, device=env.device, dtype=base.data.root_pos_w.dtype)
    return base.data.root_pos_w + offset


def position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("shaft"),
) -> torch.Tensor:
    """Return ``Shaft_large origin - hole_large origin`` in world axes [m]."""
    return shaft_position_w(env, asset_cfg) - hole_target_position_w(env)
