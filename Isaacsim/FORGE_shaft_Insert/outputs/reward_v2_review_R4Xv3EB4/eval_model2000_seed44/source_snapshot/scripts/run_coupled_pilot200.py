"""Controlled coupled-gripper retraining; stop after 200 updates and three evaluations."""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
import uuid
from run_pilot200 import PROJECT, INITIAL, digest, execute, motion_stats

ROOT = PROJECT.parent / 'logs/rsl_rl/forge_shaft_insert_coupled_release25'
OLD = PROJECT / 'outputs/pilot200/pilot200_from_model19_20260928_163043_c55e89/eval_model_218'
RECHECK = PROJECT / 'outputs/policy_recheck/20260929_133528'


def main():
    assert digest(INITIAL) == '071c17f0d60e4de4b22eec5d13c0e266233421e5de43786b101ad98962c7da7a'
    name = 'coupled200_from_model19_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    session = PROJECT / 'outputs/coupled_pilot200' / name
    session.mkdir(parents=True, exist_ok=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    # Upstream resume resolves run names inside the experiment root. This link
    # exposes the original immutable checkpoint without copying or modifying it.
    link = ROOT / 'initial_model19_original_physics'
    if not link.exists():
        link.symlink_to(INITIAL.parent, target_is_directory=True)
    assert link.resolve() == INITIAL.parent.resolve()
    command = [sys.executable, str(PROJECT/'scripts/train.py'), '--task', 'FORGE-Shaft-Insert-Paper-v0',
               '--headless', '--num_envs', '128', '--max_iterations', '200', '--seed', '42',
               '--resume', '--load_run', link.name, '--checkpoint', 'model_19.pt', '--run_name', name,
               'agent.num_steps_per_env=32', 'agent.algorithm.num_mini_batches=4', 'agent.save_interval=10']
    plan = dict(status='running', initial_checkpoint=str(INITIAL), initial_sha256=digest(INITIAL),
                train_command=command, additional_updates=200, seeds=[42,43,44],
                change='Native equal-extension finger coupling; task rewards/control/randomization unchanged from repaired configuration.',
                comparison_scope='Historical old-physics model218 versus retrained coupled-physics model218; combined physics and training effect.')
    def save():
        (session/'pilot_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    save()
    with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:
        f.write(f'\n## {name}\n用户授权实际运行耦合夹爪200轮对照训练。相同原始model19及优化器起点，128环境、seed42、rollout32、4 minibatches、保存间隔10；新增200轮至model218。奖励、控制、随机化及成功条件本轮不调整。最终32环境×4回合、seed42/43/44均值策略评估，与历史独立夹爪model218比较；此比较包含物理模型变化，不能单独归因于学习。状态和原始日志：{session}。不自动续训1000轮。\n')
    print('SESSION:', session, flush=True)
    try:
        execute(command, session/'training.log')
        runs = list(ROOT.glob('*_'+name))
        assert len(runs) == 1, runs
        run = runs[0]
        plan['training_run'] = str(run); save()
        assert json.loads((run/'experiment_manifest.json').read_text())['status'] == 'completed'
        import torch
        assert torch.load(run/'model_218.pt', map_location='cpu', weights_only=False)['iter'] == 218
        records = []
        for seed in (42,43,44):
            output = session / f'eval_model_218_seed{seed}'
            execute([sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32',
                     '--episodes_per_env','4','--seed',str(seed),'--checkpoint',str(run/'model_218.pt'),
                     '--output',str(output)], session/f'eval_seed{seed}.log')
            for group, folder in [('old', OLD if seed == 42 else RECHECK/f'model_218_seed{seed}'), ('coupled', output)]:
                summary = json.loads((folder/'evaluation_summary.json').read_text())
                assert summary['completed_episodes'] == 128 and summary['seed'] == seed
                records.append(dict(group=group, seed=seed, folder=str(folder), summary=summary, motion=motion_stats(folder)))
        (session/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
        report = '# 耦合夹爪200轮对照\n\n每项128独立回合，均值动作，保留环境随机化。相同seed不保证策略间逐回合采样完全相同。\n\n|夹爪|seed|成功率|严格落座率|超时|平均回合P95 N|峰值N|倾角峰值°|速度峰值mm/s|动作饱和率|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in records:
            s=r['summary']; m=r['motion']
            report += f"|{r['group']}|{r['seed']}|{s['rates']['success_final']:.2%}|{s['rates']['fully_seated_final']:.2%}|{s['terminations']['time_out']}|{s['mean_episode_force_p95_N']:.2f}|{s['peak_force_N']:.2f}|{m['max_gripped_tilt_deg']:.2f}|{m['max_physics_tcp_speed_mm_s']:.2f}|{s['mean_episode_action_saturation']:.2%}|\n"
        for group in ('old','coupled'):
            selected=[r for r in records if r['group']==group]
            count=sum(round(r['summary']['rates']['success_final']*128) for r in selected)
            report += f'\n{group}：成功 {count}/384 = {count/384:.2%}。\n'
        report += '\n此对照衡量夹爪修复并重训的整体效果，不能将差异全部归因于RL。已停止，不自动续训。\n'
        (session/'COMPARISON.md').write_text(report)
        plan['status']='completed'; save()
        with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:
            f.write(f'\n{name}完成，报告：{session}/COMPARISON.md。\n')
        print(report, flush=True)
    except BaseException as e:
        plan['status']='interrupted_or_failed'; plan['error']=repr(e); save()
        raise


if __name__ == '__main__':
    main()
