import gymnasium as gym
TASK_ID = 'Isaac-Forge-ShaftLarge-ZeroRand-Direct-v0'
if TASK_ID not in gym.registry:
    gym.register(id=TASK_ID, entry_point='forge_zero.env:ForgeZeroEnv', disable_env_checker=True,
        kwargs={'env_cfg_entry_point': 'forge_zero.config:ForgeZeroCfg',
                'rl_games_cfg_entry_point': 'forge_zero:ppo.yaml'})
