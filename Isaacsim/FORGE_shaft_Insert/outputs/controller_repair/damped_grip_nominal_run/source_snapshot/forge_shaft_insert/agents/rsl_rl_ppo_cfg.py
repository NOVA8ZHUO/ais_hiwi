# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ForgeInsertPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Small PPO configuration intended only after all smoke-test stages pass."""

    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "forge_shaft_xy_align_stage1"
    run_name = ""
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg


@configclass
class ForgePaperPPORunnerCfg(ForgeInsertPPORunnerCfg):
    """Recurrent asymmetric PPO; architecture/optimizer are engineering defaults."""

    experiment_name = "forge_shaft_insert_staged"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
