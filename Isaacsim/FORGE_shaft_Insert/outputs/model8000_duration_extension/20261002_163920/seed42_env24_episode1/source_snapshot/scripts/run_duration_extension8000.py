"""Replay eight historical late-entry failures; extend only the selected episode."""
import csv,json,sys
from pathlib import Path
from datetime import datetime
from run_pilot200 import PROJECT,execute

def trace(folder,env,episode):
 count=1;rows=[]
 for r in csv.DictReader((folder/'episode_metrics_control_trace.csv').open()):
  if int(r['env_id'])!=env:continue
  if count==episode:rows.append(r)
  if any(float(r[k+'_0']) for k in ['time_out','outside_workspace','invalid_grasp_width','release_complete']):
   if count==episode:break
   count+=1
 return rows

def main():
 base=PROJECT/'outputs/reward_v2_total10000/reward_v2_total10000_20261001_141859_898ead'
 old=PROJECT/'outputs/recheck_8000_9000/20261002_145916'
 run=Path(json.loads((base/'plan.json').read_text())['training_run'])
 cases=[r for r in json.loads((PROJECT/'outputs/model8000_gui_audit/historical_episode_audit.json').read_text()) if not r['success'] and r['final_depth_mm']>=2]
 assert len(cases)==8
 out=PROJECT/'outputs/model8000_duration_extension'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
 plan=dict(status='running',cases=cases,training=False,checkpoint=str(run/'model_8000.pt'))
 def save(): (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 save();print('OUTPUT:',out,flush=True);results=[]
 try:
  for c in cases:
   seed,env,ep=c['seed'],c['env'],c['episode'];folder=out/f'seed{seed}_env{env}_episode{ep}'
   execute([sys.executable,str(PROJECT/'scripts/evaluate_duration_extension.py'),'--headless','--checkpoint',str(run/'model_8000.pt'),'--output',str(folder),'--num_envs','32','--seed',str(seed),'--target-env',str(env),'--target-episode',str(ep)],out/(folder.name+'.log'))
   s=json.loads((folder/'extension_summary.json').read_text());new=trace(folder,env,ep);ref=trace((base if seed==42 else old)/f'eval_model8000_seed{seed}',env,ep)
   prefix=[r for r in new if float(r['elapsed_s_0'])<=20.0001]
   fields=[f'{n}_{i}' for n,size in [('shaft_error',3),('tcp_error',3),('action',5),('hole_bias',3),('grasp_offset_hand',3)] for i in range(size)]
   errors={k:max((abs(float(a[k])-float(b[k])) for a,b in zip(prefix,ref)),default=0) for k in fields}
   match=len(prefix)==len(ref) and all(v<=1e-5 for v in errors.values())
   row=dict(seed=seed,env=env,episode=ep,historical_depth20_mm=c['final_depth_mm'],prefix_matches_1e5=match,prefix_lengths=[len(prefix),len(ref)],prefix_errors=errors,final_result=s['target_result'])
   for duration in [20,25,30]:
    selected=[r for r in new if float(r['elapsed_s_0'])<=duration+1e-4];last=selected[-1]
    row[str(duration)]=dict(depth_mm=-float(last['shaft_error_2'])*1000,success_at_boundary=bool(float(last['engaged_25mm_0'])),success_any=any(float(r['engaged_25mm_0']) for r in selected),fully_seated=bool(float(last['fully_seated_0'])),release_complete=bool(float(last['release_complete_0'])),last_sample_s=float(last['elapsed_s_0']))
   results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  text='# model8000指定8次浅插失败的时长对照\n\n固定权重；仅指定回合延长至30s，25s为同轨迹截点。不同步复现须单独解读，不能算作原失败被救回。入孔成功与完成松爪分别统计。\n\n|seed/env/episode|前20秒匹配|20秒深度mm|25秒成功/完成松爪|30秒成功/完成松爪|\n|---|---|---:|---|---|\n'
  for r in results:text+=f"|{r['seed']}/{r['env']}/{r['episode']}|{r['prefix_matches_1e5']}|{r['20']['depth_mm']:.2f}|{r['25']['success_at_boundary']}/{r['25']['release_complete']}|{r['30']['success_at_boundary']}/{r['30']['release_complete']}|\n"
  for d in ['25','30']:
   matched=[r for r in results if r['prefix_matches_1e5']]
   text+=f"\n{d}秒：全部重跑成功{sum(r[d]['success_at_boundary'] for r in results)}/8；严格前缀匹配子集成功{sum(r[d]['success_at_boundary'] for r in matched)}/{len(matched)}。\n"
  text+='\n该测试只覆盖8次历史浅插失败，不代表384回合整体新成功率。未改全局20秒配置或部署包。\n'
  (out/'COMPARISON.md').write_text(text);plan['status']='completed';save()
  with (PROJECT/'docs/EXPERIMENT_CHANGELOG.md').open('a') as f:f.write('\n8000时长扩展验证完成：'+str(out/'COMPARISON.md')+'\n'+text)
  print('FINISHED:',out/'COMPARISON.md',flush=True)
 except BaseException as e:
  plan['status']='failed';plan['error']=repr(e);save();raise
if __name__=='__main__':main()
