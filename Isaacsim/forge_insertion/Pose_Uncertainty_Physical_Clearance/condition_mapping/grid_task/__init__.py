import gymnasium as gym
TASK='Isaac-Forge-CAD-ClearanceMapping-Direct-v0'
if TASK not in gym.registry:
 gym.register(id=TASK,entry_point='grid_task.env:GridEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':'grid_task.config:GridCfg','rl_games_cfg_entry_point':'isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml'})
