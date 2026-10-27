"""Registration only: safe before starting SimulationApp."""
import gymnasium as gym
TASK_ID = 'Isaac-Forge-ShaftLarge-CAD-Direct-v0'
if TASK_ID not in gym.registry:
    gym.register(id=TASK_ID,
        entry_point='isaaclab_tasks.direct.forge.forge_env:ForgeEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': 'forge_cad.config:ForgeCadCfg',
                'rl_games_cfg_entry_point': 'isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml'})
