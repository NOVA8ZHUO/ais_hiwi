"""Continue preserved model2999 for 1000 additional updates; evaluate intermediate and final checkpoints afterward."""
import argparse,json,sys,uuid
from datetime import datetime
from pathlib import Path
from run_pilot200 import PROJECT,execute,motion_stats,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 base=PROJECT/'outputs/reward_v2_3000/reward_v2_scratch_20260930_174656_a1f198'
 backup=PROJECT.parent/'logs/rsl_rl/forge_reward_v2_scratch/preserved_reward_v2_model2999'
 expected=json.loads((backup/'preservation.json').read_text())['sha256']
 assert digest(backup/'model_2999.pt')==expected
 manifest=json.loads((backup/'experiment_manifest.json').read_text())
 for path,h in manifest['source_sha256'].items():
  if path.startswith(('forge_shaft_insert/','assets/')) and digest(PROJECT/path)!=h:
   raise RuntimeError(f'Task changed since model2999: {path}')
 name='reward_v2_continue2999_' +datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
 root=PROJECT.parent/'logs/rsl_rl/forge_reward_v2_scratch'
 output=PROJECT/'outputs/reward_v2_continue1000'/name
 cmd=[sys.executable,str(PROJECT/'scripts/train.py'),'--task','FORGE-Shaft-Insert-Paper-v0','--headless','--num_envs','128','--max_iterations','1000','--seed','42','--run_name',name,'agent.experiment_name=forge_reward_v2_scratch','--resume','--load_run',backup.name,'--checkpoint','model_2999.pt','agent.num_steps_per_env=32','agent.algorithm.num_mini_batches=4','agent.save_interval=50']
 if a.dry_run:
  import shlex
  print(shlex.join(cmd));print('Resume weights and optimizer; eval3250 (+252), 3500 (+502), 3998 (+1000), after uninterrupted training.');return
 output.mkdir(parents=True,exist_ok=False)
 plan=dict(status='training',train_command=cmd,resume=True,updates=1000,initial_checkpoint=str(backup/"model_2999.pt"),initial_sha256=expected)
 def save(): (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 save();print('OUTPUT:',output,flush=True)
 try:
  execute(cmd,output/'training.log')
  runs=list(root.glob('*_'+name));assert len(runs)==1
  run=runs[0];plan['training_run']=str(run)
  assert json.loads((run/'experiment_manifest.json').read_text())['status']=='completed'
  plan['status']='evaluating';save();records=[]
  import torch
  for index in [3250,3500,3998]:
   assert torch.load(run/f'model_{index}.pt',map_location='cpu',weights_only=False)['iter']==index
  for seed in [42,43,44]:
   f=base/f'eval_model2999_seed{seed}'
   records.append(dict(index=2999,seed=seed,summary=json.loads((f/'evaluation_summary.json').read_text()),motion=motion_stats(f)))
  for index in [3250,3500,3998]:
   for seed in ([42,43,44] if index==3998 else [42]):
    folder=output/f'eval_model{index}_seed{seed}'
    execute([sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32','--episodes_per_env','4','--seed',str(seed),'--checkpoint',str(run/f'model_{index}.pt'),'--output',str(folder)],output/f'eval_{index}_{seed}.log')
    s=json.loads((folder/'evaluation_summary.json').read_text());assert s['completed_episodes']==128 and s['policy_state_unchanged'] and s['task_source_matches_training']
    records.append(dict(index=index,seed=seed,summary=s,motion=motion_stats(folder)))
    (output/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
  report='# 从2999新增1000轮独立评估\n\n每行128回合，均值动作、保留随机化；不同奖励下训练reward不直接比较。\n\n|模型|seed|成功率|严格落座率|P95 N|峰值N|动作饱和率|\n|---|---:|---:|---:|---:|---:|---:|\n'
  for r in records:
   s=r['summary'];report+=f"|{r['index']}|{r['seed']}|{s['rates']['success_final']:.2%}|{s['rates']['fully_seated_final']:.2%}|{s['mean_episode_force_p95_N']:.2f}|{s['peak_force_N']:.2f}|{s['mean_episode_action_saturation']:.2%}|\n"
  for index in [2999,3998]:
   subset=[r for r in records if r['index']==index]
   count=sum(round(r['summary']['rates']['success_final']*128) for r in subset)
   report+=f'\nmodel{index}三个seed：{count}/384 = {count/384:.2%}。\n'
  (output/'COMPARISON.md').write_text(report);plan['status']='completed';save()
  print('FINISHED:',output/'COMPARISON.md',flush=True)
 except BaseException as e:
  plan['status']='failed_or_interrupted';plan['error']=repr(e);save();raise
if __name__=='__main__':main()
