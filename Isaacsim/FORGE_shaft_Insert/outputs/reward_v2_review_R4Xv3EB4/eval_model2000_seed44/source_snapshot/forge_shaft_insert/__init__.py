# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-1 Franka grasped-shaft XY-alignment task registration."""

import sys

import gymnasium as gym


gym.register(
    id="FORGE-Shaft-Insert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "forge_shaft_insert.forge_insert_env_cfg:ForgeInsertEnvCfg",
        "rsl_rl_cfg_entry_point": "forge_shaft_insert.agents.rsl_rl_ppo_cfg:ForgeInsertPPORunnerCfg",
    },
)


def register_tasks() -> list[str]:
    """RSL-RL external callback: importing this package performs registration."""
    return sys.argv[1:]


gym.register(
    id="FORGE-Shaft-Insert-Position-Only-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "forge_shaft_insert.forge_insert_env_cfg:ForgeInsertPositionOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": "forge_shaft_insert.agents.rsl_rl_ppo_cfg:ForgeInsertPPORunnerCfg",
    },
)


gym.register(
    id="FORGE-Shaft-Insert-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "forge_shaft_insert.forge_insert_env_cfg:ForgeInsertCameraEnvCfg",
        "rsl_rl_cfg_entry_point": "forge_shaft_insert.agents.rsl_rl_ppo_cfg:ForgeInsertPPORunnerCfg",
    },
)


gym.register(
    id="FORGE-Shaft-Insert-Paper-v0",
    entry_point="forge_shaft_insert.metrics_env:ForgeMetricsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "forge_shaft_insert.forge_paper_env_cfg:ForgePaperEnvCfg",
        "rsl_rl_cfg_entry_point": "forge_shaft_insert.agents.rsl_rl_ppo_cfg:ForgePaperPPORunnerCfg",
    },
)
