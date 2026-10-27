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
    import forge_cad
    import paired
    import grid_task
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
    cad = True
    max_depth = torch.full((n,), -float('inf'), device=raw.device)
    original_rewards = raw._get_rewards

    def measured_rewards(self):
        reward = original_rewards()
        samples['success'] = self._get_curr_successes(self.cfg_task.success_threshold, check_rot=False).clone()
        samples['force'] = torch.linalg.vector_norm(self.force_sensor_smooth[:, :3], dim=-1).clone()
        if cad:
            samples['depth'] = self.cfg_task.fixed_asset_cfg.height - (self.held_pos[:,2] - self.fixed_pos[:,2])
            samples['xy_error'] = torch.linalg.vector_norm(self.held_pos[:,:2] - self.fixed_pos[:,:2],dim=-1)
            samples['grasp_width'] = self.joint_pos[:,7:].sum(-1).clone()
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
    # Record applied reset state and tape fingerprints before rollout.
    import os, hashlib
    from unittest.mock import patch
    initial = {
        'sample_sha256': raw.tape.meta['sha256'], 'stream_sha256':raw.tape.meta['stream_sha256'],
        'active_factors': sorted(raw.active),
        'hole_estimate_bias_m':raw.init_fixed_pos_obs_noise.cpu().tolist(),
        'clearance_mm':float(os.environ['PG_CLEARANCE_MM']), 'xy_error_radius_mm':float(os.environ['PG_ERROR_MM']),
        'base_position':raw.fixed_pos.cpu().tolist(), 'base_quaternion':raw.fixed_quat.cpu().tolist(),
        'held_position':raw.held_pos.cpu().tolist(), 'masses':raw._held_asset.root_physx_view.get_masses().tolist(),
        'friction':raw._fixed_asset.root_physx_view.get_material_properties().tolist(),
        'gains':raw.task_prop_gains.cpu().tolist(),'ema':raw.ema_factor.cpu().tolist(),
        'force_threshold':raw.contact_penalty_thresholds.cpu().tolist(),
        'dead_zone':raw.dead_zone_thresholds.cpu().tolist(),'quaternion_sign':raw.flip_quats.cpu().tolist(),
    }
    Path('initial_state.json').write_text(json.dumps(initial,indent=2))
    policy_calls = []
    def paired_normal_sample(distribution, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        noise=raw.tape.tensor('policy',distribution.loc.device)[raw.eval_step]
        assert noise.shape==distribution.loc.shape
        policy_calls.append(raw.eval_step)
        # Identical Normal(mu,sigma) law, with common pre-generated epsilon.
        with torch.no_grad():return distribution.loc+distribution.scale*noise
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        return not torch.is_tensor(value) or bool(torch.isfinite(value).all())
    # Allow enough complete synchronized waves, plus one extra wave for startup.
    bound = (a.episodes // n + 2) * (raw.max_episode_length + 2)
    with torch.inference_mode():
        for rollout_step in range(bound):
            raw.eval_step=rollout_step
            if player:
                policy_obs = obs['obs'] if isinstance(obs, dict) else obs
                assert not player.is_deterministic, 'Protocol requires stochastic policy'
                with patch.object(torch.distributions.Normal, 'sample', paired_normal_sample):
                    action = player.get_action(player.obs_to_torch(policy_obs), is_deterministic=False)
                assert policy_calls[-1:]==[rollout_step] and len(policy_calls)==rollout_step+1
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
            if cad:
                max_depth = torch.maximum(max_depth, samples['depth'])
            for idx in done.nonzero().flatten().tolist():
                if len(rows) < a.episodes:
                    rows.append({'episode': len(rows), 'env_id': idx, 'reward': returns[idx].item(), 'steps': steps[idx].item(),
                                 'success_any': seen[idx].item(), 'success_final': success[idx].item(),
                                 'first_success_seconds': first[idx].item() if seen[idx] else None,
                                 'force_mean_N': (force_sum[idx] / steps[idx]).item(),
                                 'force_max_N': force_max[idx].item(),
                                 'terminated': terminated[idx].item(), 'truncated': truncated[idx].item()})
                    if cad:
                        rows[-1].update(final_insertion_depth_m=samples['depth'][idx].item(),
                            max_insertion_depth_m=max_depth[idx].item(), final_xy_error_m=samples['xy_error'][idx].item(),
                            final_grasp_width_m=samples['grasp_width'][idx].item())
            for tensor in [returns, steps, seen, first, force_sum, force_max]:
                tensor[done] = 0
            max_depth[done] = -float('inf')
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
    if cad:
        report['cad_geometry'] = {'seating_depth_m': .047, 'official_success_depth_min_m': .04512,
            'episodes_exceeding_50mm_depth': sum(r['max_insertion_depth_m'] > .050 for r in rows),
            'note': 'Depth/width diagnostics do not change the official reward or success predicate.'}
    assert a.episodes==n, 'Exactly one synchronized episode per environment for paired sample IDs'
    report.update(sample_sha256=raw.tape.meta['sha256'],active_factors=sorted(raw.active),policy_sample_count=len(policy_calls))
    report.update(clearance_mm=float(os.environ['PG_CLEARANCE_MM']), xy_error_radius_mm=float(os.environ['PG_ERROR_MM']))
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
