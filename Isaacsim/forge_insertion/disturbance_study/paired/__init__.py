import gymnasium as gym
TASK='Isaac-Forge-ShaftLarge-Paired-Direct-v0'
if TASK not in gym.registry:
    gym.register(id=TASK,entry_point='paired.env:PairedEnv',disable_env_checker=True,
      kwargs={'env_cfg_entry_point':'paired.config:PairedCfg',
              'rl_games_cfg_entry_point':'isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml'})
