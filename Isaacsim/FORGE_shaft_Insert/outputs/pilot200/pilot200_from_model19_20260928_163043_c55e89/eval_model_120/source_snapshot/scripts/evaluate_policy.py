"""Deterministic recurrent policy evaluation with a fixed episode quota per environment."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num_envs', type=int, default=32)
parser.add_argument('--episodes_per_env', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
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
    agent = ForgePaperPPORunnerCfg()
    agent.algorithm.num_mini_batches = 1
    (out/'settings.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str)+'\n')
    (out/'agent_settings.json').write_text(json.dumps(agent.to_dict(), indent=2, default=str)+'\n')
    training_manifest = json.loads((checkpoint.parent/'experiment_manifest.json').read_text())
    root = Path(__file__).resolve().parents[1]
    differences = [name for name, digest in training_manifest['source_sha256'].items()
                   if name.startswith('forge_shaft_insert/') and hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
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
    counts = [0]*args.num_envs
    selected = []
    started = time.monotonic()
    max_steps = args.episodes_per_env * env.unwrapped.max_episode_length
    for step in range(max_steps):
        with torch.inference_mode():
            obs, _, dones, info = env.step(policy(obs))
            model.reset(dones)
        for row in info.get('episode_metrics', []):
            i = row['env_id']
            counts[i] += 1
            if counts[i] <= args.episodes_per_env:
                selected.append(dict(row, eval_episode_for_env=counts[i]))
        if (step+1) % 150 == 0:
            print('EVAL_PROGRESS', json.dumps(dict(steps=step+1, selected=len(selected), quota=args.num_envs*args.episodes_per_env, min_completed=min(counts), elapsed_s=time.monotonic()-started)), flush=True)
        if min(counts) >= args.episodes_per_env:
            break
    assert len(selected) == args.num_envs*args.episodes_per_env
    assert state_hash(model) == original_state, 'Policy or normalization state changed during inference.'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    with (out/'evaluation_episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    n = len(selected)
    rates = {name: sum(row[name] for row in selected)/n for name in ['success_any','success_final','success_stable','fully_seated_final','seated_depth_final']}
    p = rates['success_final']; z=1.96; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        deterministic=True, training=False, policy_state_unchanged=True, task_source_matches_training=True,
        seed=args.seed, num_envs=args.num_envs, episodes_per_env=args.episodes_per_env,
        completed_episodes=n, raw_completed_episodes=sum(counts), control_steps=step+1,
        elapsed_s=time.monotonic()-started, rates=rates, success_final_wilson95=[center-half,center+half],
        terminations={name:sum(row[name] for row in selected) for name in ['time_out','outside_workspace','invalid_grasp_width','release_complete']},
        mean_episode_force_p95_N=sum(row['force_p95_N'] for row in selected)/n,
        peak_force_N=max(row['force_peak_N'] for row in selected),
        mean_episode_action_saturation=sum(row['action_saturation'] for row in selected)/n,
        mean_episode_duration_s=sum(row['duration_s'] for row in selected)/n,
        protocol='First four episodes per environment by default; all environmental randomization retained, policy mean actions, no randomized initial episode lengths. Raw CSV includes unselected extra episodes of faster environments; evaluation_episodes.csv is the benchmark denominator.')
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('EVALUATION_RESULT',json.dumps(summary),flush=True)
    status='completed'
finally:
    if env is not None:
        env.close()
    archive.finish(status)
    app.close()
