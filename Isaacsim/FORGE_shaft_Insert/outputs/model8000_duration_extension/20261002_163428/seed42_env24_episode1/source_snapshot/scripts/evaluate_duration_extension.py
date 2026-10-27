"""Deterministic recurrent policy evaluation with a fixed episode quota per environment."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--episodes_per_env', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--target-env', type=int, required=True)
parser.add_argument('--target-episode', type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.num_envs > 0 and args.episodes_per_env > 0
from pathlib import Path
out = Path(args.output).resolve()
if out.exists():
    raise FileExistsError(f'Refusing to overwrite evaluation: {out}')
app = AppLauncher(args).app
import csv, hashlib, json, math, sys, time
import gymnasium as gym
import torch
import forge_shaft_insert
from forge_shaft_insert import experiment_archive as archive
from forge_shaft_insert.forge_paper_env_cfg import ForgePaperEnvCfg
from forge_shaft_insert.agents.rsl_rl_ppo_cfg import ForgePaperPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


env = None
status = 'failed'
archive.begin(out, argv=sys.argv)
try:
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = ForgePaperEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
    cfg.task.metrics_csv_path = str(out/'episode_metrics.csv')
    cfg.task.trace_num_envs = args.num_envs
    counts = [0]*args.num_envs
    def selective_timeout(e):
        result = e.episode_length_buf >= e.max_episode_length
        if counts[args.target_env]+1 == args.target_episode:
            result[args.target_env] = e.episode_length_buf[args.target_env] >= round(30/e.step_dt)
        return result
    cfg.terminations.time_out.func = selective_timeout
    agent = ForgePaperPPORunnerCfg()
    agent.algorithm.num_mini_batches = 1
    (out/'settings.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str)+'\n')
    (out/'agent_settings.json').write_text(json.dumps(agent.to_dict(), indent=2, default=str)+'\n')
    training_manifest = json.loads((checkpoint.parent/'experiment_manifest.json').read_text())
    root = Path(__file__).resolve().parents[1]
    differences = [name for name, digest in training_manifest['source_sha256'].items()
                   if name.startswith(('forge_shaft_insert/', 'assets/')) and hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
    if differences:
        raise RuntimeError(f'Task source differs from checkpoint training snapshot: {differences}')
    env = RslRlVecEnvWrapper(gym.make('FORGE-Shaft-Insert-Paper-v0', cfg=cfg), clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    model = runner.alg.policy
    assert not model.training
    original_state = state_hash(model)
    obs, _ = env.reset()
    model.reset(torch.ones(args.num_envs, dtype=torch.bool, device=args.device))
    started=time.monotonic()
    target_row=None
    for step in range(args.target_episode*env.unwrapped.max_episode_length+151):
        with torch.inference_mode():
            obs,_,dones,info=env.step(policy(obs))
            model.reset(dones)
        for row in info.get('episode_metrics',[]):
            i=row['env_id'];counts[i]+=1
            if i==args.target_env and counts[i]==args.target_episode:
                target_row=row
        if (step+1)%150==0:print('EXTENSION_PROGRESS',step+1,counts[args.target_env],flush=True)
        if target_row is not None:break
    assert target_row is not None
    assert state_hash(model)==original_state
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash
    summary=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,seed=args.seed,
                 target_env=args.target_env,target_episode=args.target_episode,target_result=target_row,
                 elapsed_s=time.monotonic()-started,policy_state_unchanged=True,
                 protocol='Only target episode timeout changed from20 to30s; other episodes remain20s; inspect trace at25s. Historical prefix equivalence must be checked separately.')
    (out/'extension_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('EXTENSION_RESULT',json.dumps(summary),flush=True)
    status='completed'
finally:
    if env is not None:
        env.close()
    archive.finish(status)
    app.close()
