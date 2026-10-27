"""Bounded smoke test / auxiliary policy evaluation for Isaac Lab 2.3.2 FORGE.
Metrics are sampled in the reward callback before automatic resets.
"""
import argparse
import json
from pathlib import Path
import types
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--task', default='Isaac-Forge-PegInsert-Direct-v0')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--num_envs', type=int, default=None)
p.add_argument('--episodes', type=int, default=128)
p.add_argument('--checkpoint', type=str)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
launcher = AppLauncher(a)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    import forge_cad  # register independent CAD task
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    cfg.seed = a.seed
    agent_cfg = load_cfg_from_registry(a.task, 'rl_games_cfg_entry_point')
    agent_cfg['params']['seed'] = a.seed
    agent_cfg['params']['config']['num_actors'] = cfg.scene.num_envs
    dump_yaml('params/env.yaml', cfg)
    dump_yaml('params/agent.yaml', agent_cfg)
    env = gym.make(a.task, cfg=cfg)
    raw = env.unwrapped
    n = raw.num_envs
    print(f'Runtime: envs={n}, action_space={env.action_space}, observation_space={env.observation_space}', flush=True)
    returns = torch.zeros(n, device=raw.device)
    steps = torch.zeros(n, dtype=torch.long, device=raw.device)
    seen = torch.zeros(n, dtype=torch.bool, device=raw.device)
    first = torch.zeros(n, device=raw.device)
    force_sum = torch.zeros(n, device=raw.device)
    force_max = torch.zeros(n, device=raw.device)
    samples = {}
    rows = []
    original_rewards = raw._get_rewards

    def measured_rewards(self):
        reward = original_rewards()
        samples['success'] = self._get_curr_successes(self.cfg_task.success_threshold, check_rot=False).clone()
        samples['force'] = torch.linalg.vector_norm(self.force_sensor_smooth[:, :3], dim=-1).clone()
        return reward

    raw._get_rewards = types.MethodType(measured_rewards, raw)
    player = None
    if a.checkpoint:
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        params = agent_cfg['params']
        params['seed'] = a.seed
        params['config']['num_actors'] = n
        env = RlGamesVecEnvWrapper(env, params['config']['device'],
                                  params['env'].get('clip_observations', float('inf')),
                                  params['env'].get('clip_actions', 1.0))
        vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper', 'env_creator': lambda **kw: env})
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(a.checkpoint)
        player.reset()
        obs = env.reset()
        player.get_batch_size(obs['obs'] if isinstance(obs, dict) else obs, 1)
        if player.is_rnn:
            player.init_rnn()
    else:
        obs, _ = env.reset()
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        return not torch.is_tensor(value) or bool(torch.isfinite(value).all())
    # Allow enough complete synchronized waves, plus one extra wave for startup.
    bound = (a.episodes // n + 2) * (raw.max_episode_length + 2)
    with torch.inference_mode():
        for rollout_step in range(bound):
            if player:
                policy_obs = obs['obs'] if isinstance(obs, dict) else obs
                action = player.get_action(player.obs_to_torch(policy_obs), is_deterministic=player.is_deterministic)
                obs, reward, done, info = env.step(action)
                terminated = raw.reset_terminated.clone()
                truncated = raw.reset_time_outs.clone()
            else:
                action = torch.zeros((n, raw.cfg.action_space), device=raw.device)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated
            if not finite(obs) or not finite(reward) or not finite(samples):
                raise RuntimeError('Non-finite observation, reward or force')
            reward = reward.to(raw.device).reshape(n)
            done = done.to(raw.device).bool().reshape(n)
            steps += 1
            returns += reward
            success = samples['success']
            first[(~seen) & success] = steps[(~seen) & success] * raw.step_dt
            seen |= success
            force_sum += samples['force']
            force_max = torch.maximum(force_max, samples['force'])
            for idx in done.nonzero().flatten().tolist():
                if len(rows) < a.episodes:
                    rows.append({'episode': len(rows), 'env_id': idx, 'reward': returns[idx].item(), 'steps': steps[idx].item(),
                                 'success_any': seen[idx].item(), 'success_final': success[idx].item(),
                                 'first_success_seconds': first[idx].item() if seen[idx] else None,
                                 'force_mean_N': (force_sum[idx] / steps[idx]).item(),
                                 'force_max_N': force_max[idx].item(),
                                 'terminated': terminated[idx].item(), 'truncated': truncated[idx].item()})
            for tensor in [returns, steps, seen, first, force_sum, force_max]:
                tensor[done] = 0
            if player and player.is_rnn and player.states is not None:
                for state in player.states:
                    state[:, done.to(state.device), :] = 0
            if rollout_step % 50 == 0:
                print(f'Rollout step={rollout_step + 1}, completed_episodes={len(rows)}/{a.episodes}', flush=True)
            if len(rows) >= a.episodes:
                break
    if len(rows) < a.episodes:
        raise RuntimeError(f'Only {len(rows)} of {a.episodes} episodes completed')
    report = {'protocol': 'auxiliary fixed-seed evaluation; not verified paper protocol' if player else 'zero-action smoke test, not policy performance',
              'seed': a.seed, 'checkpoint': a.checkpoint, 'episodes': len(rows),
              'success_rate_any': sum(r['success_any'] for r in rows) / len(rows),
              'success_rate_final': sum(r['success_final'] for r in rows) / len(rows),
              'mean_episode_reward': sum(r['reward'] for r in rows) / len(rows),
              'force_definition': 'L2 norm of smoothed wrist force, sampled at control rate',
              'rows': rows}
    Path('evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
except BaseException:
    import traceback
    Path('failure.txt').write_text(traceback.format_exc())
    raise
finally:
    if env is not None:
        env.close()
    app.close()
