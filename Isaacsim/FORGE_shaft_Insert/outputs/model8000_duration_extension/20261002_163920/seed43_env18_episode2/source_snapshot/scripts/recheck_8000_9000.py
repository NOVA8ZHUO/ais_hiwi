"""Evaluate 8000/9000 seeds43/44 and compare with completed 9997 evaluations."""
import json,csv,sys
from pathlib import Path
from datetime import datetime
from run_pilot200 import PROJECT,execute,motion_stats,digest

def main():
 base=PROJECT/'outputs/reward_v2_total10000/reward_v2_total10000_20261001_141859_898ead'
 run=Path(json.loads((base/'plan.json').read_text())['training_run'])
 out=PROJECT/'outputs/recheck_8000_9000'/datetime.now().strftime('%Y%m%d_%H%M%S')
 out.mkdir(parents=True,exist_ok=False)
 plan=dict(status='evaluating',training=False,models=[8000,9000],seeds=[43,44],training_run=str(run))
 def save(): (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 save();print('OUTPUT:',out,flush=True)
 try:
  for model in [8000,9000]:
   for seed in [43,44]:
    f=out/f'eval_model{model}_seed{seed}'
    execute([sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32','--episodes_per_env','4','--seed',str(seed),'--checkpoint',str(run/f'model_{model}.pt'),'--output',str(f)],out/f'eval_{model}_{seed}.log')
    s=json.loads((f/'evaluation_summary.json').read_text())
    assert s['completed_episodes']==128 and s['policy_state_unchanged'] and s['task_source_matches_training'] and s['checkpoint_sha256']==digest(run/f'model_{model}.pt')
  records={}
  report='# 8000/9000三种子复评（成功率优先）\n\n每模型seed42/43/44各128回合；32环境×前4回合，均值动作、环境随机化保留。seed42与9997沿用已有结果。\n\n|模型|seed42|seed43|seed44|合计成功|完成松爪严格落座|平均回合P95 N|峰值N|倾角峰值°|TCP峰值mm/s|动作饱和率|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
  for model in [8000,9000,9997]:
   rows=[];es=[]
   for seed in [42,43,44]:
    f=(base if seed==42 or model==9997 else out)/f'eval_model{model}_seed{seed}'
    s=json.loads((f/'evaluation_summary.json').read_text());assert s['completed_episodes']==128 and s['seed']==seed
    rows.append(dict(seed=seed,summary=s,motion=motion_stats(f)))
    es+=list(csv.DictReader((f/'evaluation_episodes.csv').open()))
   a=dict(success=sum(int(e['success_final']) for e in es),release_seated=sum(int(e['release_complete']) and int(e['fully_seated_final']) for e in es),seated=sum(int(e['fully_seated_final']) for e in es),timeout=sum(int(e['time_out']) for e in es),outside=sum(int(e['outside_workspace']) for e in es),width=sum(int(e['invalid_grasp_width']) for e in es),p95=sum(r['summary']['mean_episode_force_p95_N'] for r in rows)/3,peak=max(r['summary']['peak_force_N'] for r in rows),tilt=max(r['motion']['max_gripped_tilt_deg'] for r in rows),speed=max(r['motion']['max_physics_tcp_speed_mm_s'] for r in rows),sat=sum(r['summary']['mean_episode_action_saturation'] for r in rows)/3,shallow=sum(r['motion']['shallow_stalls_2_to_25mm'] for r in rows))
   records[str(model)]=dict(aggregate=a,rows=rows)
   rates='|'.join(f"{r['summary']['rates']['success_final']:.2%}" for r in rows)
   report+=f"|{model}|{rates}|{a['success']}/384 ({a['success']/384:.2%})|{a['release_seated']}/384|{a['p95']:.2f}|{a['peak']:.2f}|{a['tilt']:.2f}|{a['speed']:.2f}|{a['sat']:.2%}|\n"
  report+='\n按三seed总体成功率优先比较，不仅取最高单seed。小幅差异不足以证明总体最优；相同seed不保证逐回合随机误差配对。接触峰值为控制频率采样。未训练、未替换部署包。\n'
  (out/'COMPARISON.md').write_text(report);(out/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
  plan['status']='completed';save()
  with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:f.write('\n8000/9000补评完成：'+str(out/'COMPARISON.md')+'\n'+report+'\n')
  print(report,flush=True);print('FINISHED:',out/'COMPARISON.md',flush=True)
 except BaseException as e:
  plan['status']='failed_or_interrupted';plan['error']=repr(e);save();raise
if __name__=='__main__':main()
