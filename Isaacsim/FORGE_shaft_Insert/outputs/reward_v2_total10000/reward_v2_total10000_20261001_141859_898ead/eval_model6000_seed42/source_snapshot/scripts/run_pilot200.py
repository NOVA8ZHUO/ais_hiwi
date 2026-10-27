"""Resume model_19 for 200 updates, then evaluate three checkpoints; never auto-extend."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RUN_ROOT = WORKSPACE/'logs/rsl_rl/forge_shaft_insert_release25'
INITIAL = RUN_ROOT/'2026-09-28_15-34-34_release25_gui_20260928_152608/model_19.pt'
BASELINE = PROJECT/'outputs/policy_baseline/model19_seed42_32x4'


def checkpoints(start=19, updates=200, interval=10):
    return [math.ceil((start+n-1)/interval)*interval for n in (50,100)] + [start+updates-1]


def commands(python, run_name, training_run):
    train = [python, str(PROJECT/'scripts/train.py'), '--task', 'FORGE-Shaft-Insert-Paper-v0',
             '--headless', '--num_envs', '128', '--max_iterations', '200', '--seed', '42',
             '--resume', '--load_run', INITIAL.parent.name, '--checkpoint', INITIAL.name,
             '--run_name', run_name, 'agent.num_steps_per_env=32',
             'agent.algorithm.num_mini_batches=4', 'agent.save_interval=10']
    evals = [[python,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32',
              '--episodes_per_env','4','--seed','42','--checkpoint',str(training_run/f'model_{i}.pt')]
             for i in checkpoints()]
    return train, evals


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_initial():
    baseline = json.loads((BASELINE/'evaluation_summary.json').read_text())
    if digest(INITIAL) != baseline['checkpoint_sha256']:
        raise RuntimeError('Initial checkpoint differs from measured baseline.')
    manifest = json.loads((INITIAL.parent/'experiment_manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if name.startswith(('forge_shaft_insert/', 'assets/')) and digest(PROJECT/name) != expected:
            raise RuntimeError(f'Task source changed since baseline: {name}')


def execute(cmd, log):
    print('\nRUN:',shlex.join(cmd),flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(cmd,cwd=WORKSPACE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
                stream.write(line); stream.flush()
            code=process.wait()
            if code:
                raise RuntimeError(f'Child exited {code}; see {log}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise


def motion_stats(folder):
    counts={}; speed=tilt=0.
    with (folder/'episode_metrics_control_trace.csv').open() as stream:
        for row in csv.DictReader(stream):
            i=int(row['env_id']); count=counts.get(i,0)
            if count < 4:
                speed=max(speed,float(row['physics_peak_speed_0']))
                if not float(row['releasing_0']) and float(row['elapsed_s_0'])>=1:
                    tilt=max(tilt,math.degrees(math.acos(max(-1,min(1,float(row['shaft_axis_world_2']))))))
            if any(float(row.get(k+'_0',0)) for k in ('time_out','outside_workspace','invalid_grasp_width','release_complete')):
                counts[i]=count+1
    with (folder/'evaluation_episodes.csv').open() as stream:
        episodes=list(csv.DictReader(stream))
    shallow=sum(not int(x['success_final']) and 2<=float(x['final_depth_mm'])<25 for x in episodes)
    return dict(max_physics_tcp_speed_mm_s=speed*1000,max_gripped_tilt_deg=tilt,shallow_stalls_2_to_25mm=shallow)


def report(session, folders):
    records=[]
    for label,folder in folders:
        s=json.loads((folder/'evaluation_summary.json').read_text())
        records.append(dict(label=label,folder=str(folder),summary=s,motion=motion_stats(folder)))
    (session/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
    text='# 200轮pilot独立评估比较\n\n每个检查点32环境×每环境前4回合，seed42，策略均值动作；环境随机化保留。\n\n'
    text+='| 模型 | 成功率 | 严格落座率 | 超时 | 浅插停滞2–25mm | 平均回合P95 N | 峰值 N | 夹持倾角峰值° | TCP物理步峰值mm/s | 动作饱和率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in records:
        s=row['summary']; m=row['motion']
        text+=f"| {row['label']} | {s['rates']['success_final']:.1%} | {s['rates']['fully_seated_final']:.1%} | {s['terminations']['time_out']} | {m['shallow_stalls_2_to_25mm']} | {s['mean_episode_force_p95_N']:.2f} | {s['peak_force_N']:.2f} | {m['max_gripped_tilt_deg']:.2f} | {m['max_physics_tcp_speed_mm_s']:.2f} | {s['mean_episode_action_saturation']:.2%} |\n"
    change=records[-1]['summary']['rates']['success_final']-records[0]['summary']['rates']['success_final']
    text+=f'\n新增200轮模型相对基准成功率变化：{change*100:+.2f}个百分点。\n'
    text+='\n**到此停止，不自动续训1000轮。** 根据各检查点成功率、置信区间与力/倾角/速度变化共同判断；单次小幅变化不能证明学习收益。浅插停滞仅按终局深度分类，不替代轨迹诊断。不同策略相同seed不保证逐回合误差样本完全相同。接触峰值为控制频率采样；动作饱和率按回合平均。\n'
    (session/'COMPARISON.md').write_text(text)
    with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:
        f.write(f'\n128环境新增200轮pilot及三次独立评估已完成，报告：{session}/COMPARISON.md。最终相对基准成功率变化{change*100:+.2f}个百分点；停止等待审阅，不自动续训。\n')
    print(text,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--training-run',type=Path,help='Evaluate an already completed pilot run, without training again.')
    args=parser.parse_args()
    verify_initial()
    run_name='pilot200_from_model19_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    session=PROJECT/'outputs/pilot200'/run_name
    train,evals=commands(sys.executable,run_name,args.training_run or RUN_ROOT/'<new_training_run>')
    if args.dry_run:
        print('Training:',shlex.join(train))
        for i,cmd in zip(checkpoints(),evals):
            print(f'Evaluate +{i-19+1} updates:',shlex.join(cmd+['--output',str(session/f'eval_model_{i}')]))
        print('No training started. No task, reward or control settings changed.')
        return
    session.mkdir(parents=True,exist_ok=False)
    plan=dict(initial_checkpoint=str(INITIAL),initial_sha256=digest(INITIAL),baseline=str(BASELINE),
              train_command=train,checkpoints=checkpoints(),additional_updates=[52,102,200],status='running')
    def save(): (session/'pilot_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save()
    try:
        if args.training_run:
            training_run=args.training_run.resolve()
        else:
            execute(train,session/'training.log')
            matches=list(RUN_ROOT.glob('*_'+run_name))
            if len(matches)!=1: raise RuntimeError(f'Expected one unique training run, found {matches}')
            training_run=matches[0]
        plan['training_run']=str(training_run); save()
        manifest=json.loads((training_run/'experiment_manifest.json').read_text())
        if manifest['status']!='completed': raise RuntimeError('Training did not complete; refusing to evaluate it as 200 updates.')
        import torch
        for i in checkpoints():
            ckpt=training_run/f'model_{i}.pt'
            state=torch.load(ckpt,map_location='cpu',weights_only=False)
            if state['iter']!=i: raise RuntimeError(f'Checkpoint iteration mismatch: {ckpt}')
            del state
        _,evals=commands(sys.executable,run_name,training_run)
        folders=[('model_19基准',BASELINE)]
        for i,cmd in zip(checkpoints(),evals):
            output=session/f'eval_model_{i}'
            execute(cmd+['--output',str(output)],session/f'eval_model_{i}.log')
            folders.append((f'新增{i-18}轮/model_{i}',output))
        report(session,folders)
        plan['status']='completed'; save()
        print(f'FINISHED: {session}/COMPARISON.md\nStopped after pilot; no 1000-iteration continuation.',flush=True)
    except BaseException as error:
        plan['status']='interrupted_or_failed'; plan['error']=repr(error); save()
        raise


if __name__=='__main__':
    main()
