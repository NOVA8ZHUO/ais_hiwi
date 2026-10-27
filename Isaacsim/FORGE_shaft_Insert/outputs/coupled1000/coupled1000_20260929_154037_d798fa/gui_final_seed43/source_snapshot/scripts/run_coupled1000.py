"""Continue coupled pilot to 1000 cumulative updates, evaluate checkpoints, optionally open GUI."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys
import uuid

from run_pilot200 import PROJECT, digest, execute, motion_stats

PILOT = PROJECT/'outputs/coupled_pilot200/coupled200_from_model19_20260929_145548_bddcdd'
ROOT = PROJECT.parent/'logs/rsl_rl/forge_shaft_insert_coupled_release25'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true', help='Open final deterministic four-environment GUI evaluation after headless evaluations.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    initial = Path(json.loads((PILOT/'pilot_plan.json').read_text())['training_run'])/'model_218.pt'
    manifest = json.loads((initial.parent/'experiment_manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if name.startswith(('forge_shaft_insert/', 'assets/')) and digest(PROJECT/name) != expected:
            raise RuntimeError(f'Task differs from coupled pilot: {name}')
    name = 'coupled1000_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    session = PROJECT/'outputs/coupled1000'/name
    train = [sys.executable,str(PROJECT/'scripts/train.py'),'--task','FORGE-Shaft-Insert-Paper-v0',
             '--headless','--num_envs','128','--max_iterations','800','--seed','42','--resume',
             '--load_run',initial.parent.name,'--checkpoint',initial.name,'--run_name',name,
             'agent.num_steps_per_env=32','agent.algorithm.num_mini_batches=4','agent.save_interval=10']
    # RSL restores iter=218 and starts its next loop at 218. Thus 800 updates
    # end at 1017, not 1018. Intermediate save indices are multiples of ten.
    checkpoints = [(420,403),(620,603),(820,803),(1017,1000)]
    def evaluation(run, index, seed, output, gui=False):
        cmd=[sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--checkpoint',str(run/f'model_{index}.pt'),
             '--num_envs','4' if gui else '32','--episodes_per_env','4','--seed',str(seed),'--output',str(output)]
        return cmd if gui else cmd+['--headless']
    if args.dry_run:
        print(shlex.join(train))
        print('Checkpoints (file index, cumulative new updates):',checkpoints)
        print('Each checkpoint: seed42, 32 environments x 4 episodes; final also seeds43/44.')
        print('GUI:',args.gui,'4 environments x 4 episodes, seed43, mean actions; no learning.')
        return
    session.mkdir(parents=True,exist_ok=False)
    plan=dict(status='training',initial_checkpoint=str(initial),initial_sha256=digest(initial),train_command=train,
              previous_updates=200,additional_updates=800,total_updates=1000,checkpoints=checkpoints,
              evaluation_timing='Evaluate saved intermediate checkpoints after uninterrupted training finishes.')
    def save(): (session/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save()
    print('SESSION:',session,flush=True)
    try:
        execute(train,session/'training.log')
        runs=list(ROOT.glob('*_'+name))
        if len(runs)!=1: raise RuntimeError(f'Ambiguous training run: {runs}')
        run=runs[0]; plan['training_run']=str(run)
        if json.loads((run/'experiment_manifest.json').read_text())['status']!='completed':
            raise RuntimeError('Training incomplete')
        import torch
        for index,_ in checkpoints:
            if torch.load(run/f'model_{index}.pt',map_location='cpu',weights_only=False)['iter']!=index:
                raise RuntimeError('Checkpoint iteration mismatch')
        plan['status']='evaluating'; save()
        folders=[('耦合200轮基准 seed42',PILOT/'eval_model_218_seed42')]
        for index,total in checkpoints:
            output=session/f'eval_model_{index}_seed42'
            execute(evaluation(run,index,42,output),session/f'eval_model_{index}_seed42.log')
            folders.append((f'累计{total}轮 seed42',output))
        for seed in (43,44):
            output=session/f'eval_model_1017_seed{seed}'
            execute(evaluation(run,1017,seed,output),session/f'eval_model_1017_seed{seed}.log')
            folders.extend([(f'耦合200轮基准 seed{seed}',PILOT/f'eval_model_218_seed{seed}'),
                            (f'累计1000轮 seed{seed}',output)])
        records=[]
        table='# 耦合夹爪累计1000轮评估\n\n每行32环境×4回合；策略均值动作，环境随机化保留。\n\n|模型|成功率|严格落座率|超时|平均回合P95 N|峰值N|倾角峰值°|TCP峰值mm/s|饱和率|浅插失败|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for label,folder in folders:
            summary=json.loads((folder/'evaluation_summary.json').read_text())
            if summary['completed_episodes']!=128: raise RuntimeError('Incomplete evaluation')
            m=motion_stats(folder)
            records.append(dict(label=label,folder=str(folder),summary=summary,motion=m))
            table+=f"|{label}|{summary['rates']['success_final']:.2%}|{summary['rates']['fully_seated_final']:.2%}|{summary['terminations']['time_out']}|{summary['mean_episode_force_p95_N']:.2f}|{summary['peak_force_N']:.2f}|{m['max_gripped_tilt_deg']:.2f}|{m['max_physics_tcp_speed_mm_s']:.2f}|{summary['mean_episode_action_saturation']:.2%}|{m['shallow_stalls_2_to_25mm']}|\n"
        (session/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
        report_path=session/'COMPARISON.md'
        for label,base in [('耦合200轮',PILOT),('耦合1000轮',session)]:
            index=218 if base==PILOT else 1017
            summaries=[json.loads((base/f'eval_model_{index}_seed{s}'/'evaluation_summary.json').read_text()) for s in (42,43,44)]
            count=sum(round(s['rates']['success_final']*s['completed_episodes']) for s in summaries)
            table+=f'\n{label}三个seed合计成功：{count}/384（{count/384:.2%}）。\n'
        table+='\n均为保留随机化的独立均值策略评估；GUI单列，不混入384回合统计。相同seed不保证策略间逐回合随机样本相同。\n'
        report_path.write_text(table)
        gui=evaluation(run,1017,43,session/'gui_final_seed43',gui=True)
        (session/'gui_command.sh').write_text('#!/usr/bin/env bash\nset -e\ncd '+shlex.quote(str(PROJECT.parent))+'\n'+shlex.join(gui)+'\n')
        plan['status']='headless_completed'; save()
        if args.gui:
            execute(gui,session/'gui_final.log')
            summary=json.loads((session/'gui_final_seed43/evaluation_summary.json').read_text())
            if summary['completed_episodes']!=16: raise RuntimeError('GUI evaluation incomplete')
        plan['status']='completed'; save()
        with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as stream:
            stream.write(f'\n累计1000轮及独立评估已完成：{report_path}；GUI执行={args.gui}。\n')
        print('FINISHED:',report_path,'\nGUI command:',session/'gui_command.sh',flush=True)
    except BaseException as error:
        plan['error']=repr(error); plan['status_before_error']=plan['status']; plan['status']='failed_or_interrupted'; save()
        raise


if __name__=='__main__':
    main()
