"""Finite deterministic replay with current task settings and pre-reset control traces."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--steps', type=int, default=160)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', required=True, help='Episode CSV; control trace saved alongside it.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import json
from pathlib import Path
import gymnasium as gym
import torch
import forge_shaft_insert
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from forge_shaft_insert.forge_paper_env_cfg import ForgePaperEnvCfg
from forge_shaft_insert.agents.rsl_rl_ppo_cfg import ForgePaperPPORunnerCfg

env = None
try:
    cfg = ForgePaperEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
    cfg.task.metrics_csv_path = args.output
    cfg.task.trace_num_envs = args.num_envs
    env = RslRlVecEnvWrapper(gym.make('FORGE-Shaft-Insert-Paper-v0', cfg=cfg), clip_actions=1.)
    agent = ForgePaperPPORunnerCfg()
    agent.algorithm.num_mini_batches = 1
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    for step in range(args.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
            runner.alg.policy.reset(dones)
    summary = dict(checkpoint=str(Path(args.checkpoint).resolve()), seed=args.seed,
                   deterministic=True, settings='Current task settings; not a reconstruction of historical trajectories',
                   metrics=env.unwrapped.episode_metrics.summary())
    Path(args.output).with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('POLICY_DIAGNOSTIC', json.dumps(summary), flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
