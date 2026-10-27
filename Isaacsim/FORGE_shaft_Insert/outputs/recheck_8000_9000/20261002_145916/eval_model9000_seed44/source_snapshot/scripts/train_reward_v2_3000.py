"""Fresh 3000-update PPO experiment; evaluate intermediate and final checkpoints afterward."""
import argparse,json,sys,uuid
from datetime import datetime
from pathlib import Path
from run_pilot200 import PROJECT,execute,motion_stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 name='reward_v2_scratch_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
 root=PROJECT.parent/'logs/rsl_rl/forge_reward_v2_scratch'
 output=PROJECT/'outputs/reward_v2_3000'/name
 cmd=[sys.executable,str(PROJECT/'scripts/train.py'),'--task','FORGE-Shaft-Insert-Paper-v0','--headless','--num_envs','128','--max_iterations','3000','--seed','42','--run_name',name,'agent.experiment_name=forge_reward_v2_scratch','agent.resume=False','agent.num_steps_per_env=32','agent.algorithm.num_mini_batches=4','agent.save_interval=100']
 if a.dry_run:
  import shlex
  print(shlex.join(cmd));print('Fresh weights and optimizer; eval model1000 (+1001), model2000 (+2001), model2999 (+3000).');return
 output.mkdir(parents=True,exist_ok=False)
 plan=dict(status='training',train_command=cmd,resume=False,updates=3000)
 def save(): (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 save();print('OUTPUT:',output,flush=True)
 try:
  execute(cmd,output/'training.log')
  runs=list(root.glob('*_'+name));assert len(runs)==1
  run=runs[0];plan['training_run']=str(run)
  assert json.loads((run/'experiment_manifest.json').read_text())['status']=='completed'
  plan['status']='evaluating';save();records=[]
  for index in [1000,2000,2999]:
   for seed in ([42,43,44] if index==2999 else [42]):
    folder=output/f'eval_model{index}_seed{seed}'
    execute([sys.executable,str(PROJECT/'scripts/evaluate_policy.py'),'--headless','--num_envs','32','--episodes_per_env','4','--seed',str(seed),'--checkpoint',str(run/f'model_{index}.pt'),'--output',str(folder)],output/f'eval_{index}_{seed}.log')
    s=json.loads((folder/'evaluation_summary.json').read_text());assert s['completed_episodes']==128 and s['policy_state_unchanged'] and s['task_source_matches_training']
    records.append(dict(index=index,seed=seed,summary=s,motion=motion_stats(folder)))
    (output/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
  report='# 新奖励从零3000轮独立评估\n\n每行128回合，均值动作、保留随机化；不同奖励下训练reward不直接比较。\n\n|模型|seed|成功率|严格落座率|P95 N|峰值N|动作饱和率|\n|---|---:|---:|---:|---:|---:|---:|\n'
  for r in records:
   s=r['summary'];report+=f"|{r['index']}|{r['seed']}|{s['rates']['success_final']:.2%}|{s['rates']['fully_seated_final']:.2%}|{s['mean_episode_force_p95_N']:.2f}|{s['peak_force_N']:.2f}|{s['mean_episode_action_saturation']:.2%}|\n"
  (output/'COMPARISON.md').write_text(report);plan['status']='completed';save()
  print('FINISHED:',output/'COMPARISON.md',flush=True)
 except BaseException as e:
  plan['status']='failed_or_interrupted';plan['error']=repr(e);save();raise
if __name__=='__main__':main()
