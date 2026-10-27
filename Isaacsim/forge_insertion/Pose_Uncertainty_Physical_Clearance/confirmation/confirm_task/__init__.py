import gymnasium as gym
TASK='Isaac-Forge-CAD-PG-Confirmation-Direct-v0'
if TASK not in gym.registry:
 gym.register(id=TASK,entry_point='confirm_task.env:ConfirmEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':'confirm_task.config:ConfirmCfg','rl_games_cfg_entry_point':'isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml'})
