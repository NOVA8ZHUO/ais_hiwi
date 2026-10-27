"""Resume preserved model820 for 10000 additional PPO updates, then evaluate saved milestones."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import uuid
from run_pilot200 import PROJECT, digest, execute, motion_stats

ROOT = PROJECT.parent/'logs/rsl_rl/forge_shaft_insert_coupled_release25'
BACKUP = ROOT/'preserved_model820_20260929'
BASELINE = PROJECT/'outputs/coupled1000/coupled1000_20260929_154037_d798fa'
EXPECTED = 'bb300c0661534ab629cd73d76906bc65f0dc55d94ab6a342fe0e6715f6a4252f'
MILESTONES = [1820,2820,4820,6820,8820,10819]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--training-run',type=Path,help='Evaluate an already completed run without repeating training.')
    args=parser.parse_args()
    initial=BACKUP/'model_820.pt'
    if digest(initial)!=EXPECTED: raise RuntimeError('Preserved checkpoint checksum mismatch')
    manifest=json.loads((BACKUP/'experiment_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        if name.startswith(('forge_shaft_insert/','assets/')) and digest(PROJECT/name)!=expected:
            raise RuntimeError(f'Task changed since model820: {name}')
    name='model820_plus10000_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    session=PROJECT/'outputs/model820_plus10000'/name
    train=[sys.executable,str(PROJECT/'scripts/train.py'),'--task','FORGE-Shaft-Insert-Paper-v0',
           '--headless','--num_envs','128','--max_iterations','10000','--seed','42','--resume',
           '--load_run',BACKUP.name,'--checkpoint','model_820.pt','--run_name',name,
           'agent.num_steps_per_env=32','agent.algorithm.num_mini_batches=4','agent.save_interval=20']
    if args.dry_run:
        print(shlex.join(train))
        print('Checkpoint / additional updates:',[(i,i-819) for i in MILESTONES])
        print('Uninterrupted training; evaluate all milestones afterward with seed42; final also seeds43/44.')
        return
    session.mkdir(parents=True,exist_ok=False)
    plan=dict(status='starting',initial_checkpoint=str(initial),initial_sha256=EXPECTED,
              train_command=train,additional_updates=10000,milestones=[(i,i-819) for i in MILESTONES],
              evaluation_timing='After uninterrupted training, not during training',training_run=None)
    def save(): (session/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    def log(message):
        with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:f.write('\n'+message+'\n')
    save(); print('SESSION:',session,flush=True)
    log(f'{name}启动：从备份model820新增10000轮；128环境、rollout32、4 minibatches、seed42，保留优化器与自适应学习率设置。奖励/观测/控制/随机化不变；保存间隔10→20，仅改变保存频率。计划：{session}/plan.json。')
    try:
        if args.training_run:
            run=args.training_run.resolve()
        else:
            plan['status']='training';save()
            execute(train,session/'training.log')
            matches=list(ROOT.glob('*_'+name))
            if len(matches)!=1: raise RuntimeError(f'Unexpected runs: {matches}')
            run=matches[0]
        plan['training_run']=str(run);save()
        if json.loads((run/'experiment_manifest.json').read_text())['status']!='completed':
            raise RuntimeError('Training not complete')
        import torch
        for i in MILESTONES:
            if torch.load(run/f'model_{i}.pt',map_location='cpu',weights_only=False)['iter']!=i:
                raise RuntimeError('Checkpoint iteration mismatch')
        plan['status']='evaluating';save()
        records=[]
        def collect(label,folder):
            s=json.loads((folder/'evaluation_summary.json').read_text())
            if not (s['completed_episodes']==128 and s['deterministic'] and not s['training']
                    and s['policy_state_unchanged'] and s['task_source_matches_training']):
                raise RuntimeError(f'Invalid evaluation: {folder}')
            records.append(dict(label=label,folder=str(folder),summary=s,motion=motion_stats(folder)))
            (session/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
        for seed in (42,43,44):collect(f'原model820 seed{seed}',BASELINE/f'eval_model_820_seed{seed}')
        for i in MILESTONES:
            for seed in ((42,43,44) if i==10819 else (42,)):
                folder=session/f'eval_model_{i}_seed{seed}'
                cmd=[sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32',
                     '--episodes_per_env','4','--seed',str(seed),'--checkpoint',str(run/f'model_{i}.pt'),'--output',str(folder)]
                execute(cmd,session/f'eval_model_{i}_seed{seed}.log')
                collect(f'新增{i-819}轮/model{i} seed{seed}',folder)
        report='# model820新增10000轮独立评估\n\n每行32环境×前4回合，均值策略动作，保留随机化。中间检查点仅seed42，不能与最终三seed混合排名；GUI另行运行。\n\n|模型|成功率|严格落座率|超时|P95 N|峰值N|倾角峰值°|TCP峰值mm/s|动作饱和|浅插失败|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in records:
            s=r['summary'];m=r['motion']
            report+=f"|{r['label']}|{s['rates']['success_final']:.2%}|{s['rates']['fully_seated_final']:.2%}|{s['terminations']['time_out']}|{s['mean_episode_force_p95_N']:.2f}|{s['peak_force_N']:.2f}|{m['max_gripped_tilt_deg']:.2f}|{m['max_physics_tcp_speed_mm_s']:.2f}|{s['mean_episode_action_saturation']:.2%}|{m['shallow_stalls_2_to_25mm']}|\n"
        for label,subset in [('原model820',records[:3]),('新增10000轮',records[-3:])]:
            n=sum(round(r['summary']['rates']['success_final']*128) for r in subset)
            report+=f'\n{label}：{n}/384 = {n/384:.2%}。\n'
        report+='\n相同seed不保证逐回合误差配对。接触峰值为控制频率采样。保存原模型，不自动用最终模型替换最佳模型。\n'
        (session/'COMPARISON.md').write_text(report)
        gui=[sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--num_envs','4','--episodes_per_env','4',
             '--seed','43','--checkpoint',str(run/'model_10819.pt'),'--output',str(session/'gui_final_seed43')]
        (session/'gui_command.sh').write_text('#!/usr/bin/env bash\nset -e\ncd '+shlex.quote(str(PROJECT.parent))+'\n'+shlex.join(gui)+'\n')
        if digest(initial)!=EXPECTED: raise RuntimeError('Preserved checkpoint changed')
        plan['status']='completed';save();log(f'{name}完成：{session}/COMPARISON.md。GUI未自动运行。')
        print('FINISHED:',session/'COMPARISON.md',flush=True)
    except BaseException as error:
        plan['status_before_error']=plan['status'];plan['status']='failed_or_interrupted';plan['error']=repr(error);save()
        raise


if __name__=='__main__':main()
