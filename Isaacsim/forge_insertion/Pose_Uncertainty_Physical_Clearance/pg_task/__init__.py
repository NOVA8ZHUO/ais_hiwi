import gymnasium as gym
for group in ['P0G0','P1G0','P0G1','P1G1']:
    task=f'Isaac-Forge-CAD-PG-{group}-Direct-v0'
    if task not in gym.registry:
        gym.register(id=task,entry_point='pg_task.env:PGEnv',disable_env_checker=True,
            kwargs={'env_cfg_entry_point':f'pg_task.config:{group}Cfg','rl_games_cfg_entry_point':'isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml'})
