import csv,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
base=P/'outputs/reward_v2_total10000/reward_v2_total10000_20261001_141859_898ead'
recheck=P/'outputs/recheck_8000_9000/20261002_145916'
results=[]
for seed in [42,43,44]:
 folder=(base if seed==42 else recheck)/f'eval_model8000_seed{seed}'
 es={(int(e['env_id']),int(e['eval_episode_for_env'])):e for e in csv.DictReader((folder/'evaluation_episodes.csv').open())}
 traces={};counts={}
 for r in csv.DictReader((folder/'episode_metrics_control_trace.csv').open()):
  env=int(r['env_id']);ep=counts.get(env,0)+1
  if ep<=4:traces.setdefault((env,ep),[]).append(r)
  if any(float(r[k+'_0']) for k in ['time_out','outside_workspace','invalid_grasp_width','release_complete']):counts[env]=ep
 for key,rows in traces.items():
  e=es[key]
  def ar(k):return np.array([float(r[k]) for r in rows])
  t=ar('elapsed_s_0');depth=-ar('shaft_error_2')*1000;xy=np.hypot(ar('shaft_error_0'),ar('shaft_error_1'))*1000
  active=(ar('motion_enabled_before_action_0')>0)&(ar('releasing_0')==0)
  tilt=np.degrees(np.arccos(np.clip(ar('shaft_axis_world_2'),-1,1)));stage=ar('stage_0').astype(int)
  fx=np.hypot(ar('contact_force_0'),ar('contact_force_1'));fz=ar('contact_force_2');force=np.hypot(fx,fz)
  ids=np.flatnonzero(active);idx=ids[np.argmax(tilt[ids])]
  entry=np.flatnonzero(active&(depth>=2)&(xy<=13))
  row=dict(seed=seed,env=key[0],episode=key[1],success=int(e['success_final']),final_depth_mm=float(e['final_depth_mm']),final_xy_mm=float(e['final_xy_mm']),max_depth_mm=float(depth[ids].max()),final_stage=int(stage[ids[-1]]),tilt_max_deg=float(tilt[idx]),tilt_max_time_s=float(t[idx]),tilt_above5_s=float(np.sum(tilt[ids]>5)/15),tilt_above10_s=float(np.sum(tilt[ids]>10)/15),peak_tilt_depth_mm=float(depth[idx]),peak_tilt_xy_mm=float(xy[idx]),peak_tilt_stage=int(stage[idx]),peak_tilt_force_N=float(force[idx]),first_entry_s=float(t[entry[0]]) if len(entry) else None,tip_tcp_final_xy_mm=float(np.hypot(ar('shaft_error_0')[-1]-ar('tcp_error_0')[-1],ar('shaft_error_1')[-1]-ar('tcp_error_1')[-1])*1000),finger_diff_peak_mm=float(abs(ar('finger_difference_0')[ids]).max()*1000))
  for j,n in enumerate('xyz'):row['hole_bias_'+n+'_mm']=float(ar('hole_bias_'+str(j))[0]*1000);row['grasp_'+n+'_mm']=float(ar('grasp_offset_hand_'+str(j))[0]*1000)
  row['deadzone_xy_N']=[float(ar('deadzone_'+str(j))[0]) for j in [0,1]]
  window=ids[t[ids]>=t[ids[-1]]-5]
  row['last5_gain_mm']=float(depth[window[-1]]-depth[window[0]])
  row['last5_speed_z_mm_s']=float(np.mean(ar('tcp_velocity_2')[window])*1000)
  row['last5_xy_force_N']=float(np.mean(fx[window]));row['last5_abs_z_force_N']=float(np.mean(abs(fz[window])))
  row['last5_down_command_N']=float(np.mean(np.maximum(-ar('command_force_2')[window],0)))
  row['last5_xy_command_N']=float(np.mean(np.hypot(ar('command_force_0')[window],ar('command_force_1')[window])))
  row['search_fraction']=float(np.mean(stage[ids]==1));row['probe_returns']=int(sum(stage[i-1]==2 and stage[i]==1 for i in ids if i>0))
  if len(entry):
   post=ids[ids>=entry[0]];anchor=depth[post[0]];at=t[post[0]];stall=0
   for i in post:
    if depth[i]>=anchor+1:stall=max(stall,t[i]-at);anchor=depth[i];at=t[i]
   row['longest_no_new_1mm_s']=float(max(stall,t[post[-1]]-at));row['post_search_fraction']=float(np.mean(stage[post]==1));row['post_stage_switches']=int(np.count_nonzero(np.diff(stage[post])));row['retreat_mm']=float(depth[post].max()-depth[post[-1]])
  results.append(row)
assert len(results)==384
failed=[r for r in results if not r['success']];assert len(failed)==30
(out/'historical_episode_audit.json').write_text(json.dumps(results,indent=2)+'\n')
with (out/'historical_failures.csv').open('w') as f:
 fields=list(dict.fromkeys(k for r in results for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(failed)
for label,rs in [('no_entry_final',[r for r in failed if r['final_depth_mm']<2]),('shallow_final',[r for r in failed if r['final_depth_mm']>=2])]:
 print(label,'N',len(rs),'entered',sum(r['first_entry_s'] is not None for r in rs),'finalstages',{s:sum(r['final_stage']==s for r in rs) for s in [1,2,3]})
 for k in ['final_depth_mm','final_xy_mm','last5_gain_mm','last5_xy_force_N','last5_abs_z_force_N','last5_down_command_N','last5_xy_command_N','probe_returns','tilt_max_deg','tip_tcp_final_xy_mm']:
  print(k,'median',round(float(np.median([r[k] for r in rs])),3),'range',tuple(round(v,3) for v in [min(r[k] for r in rs),max(r[k] for r in rs)]))
print('SHALLOW',json.dumps([r for r in failed if r['final_depth_mm']>=2]))
print('HIGH_TILT',json.dumps(sorted(results,key=lambda r:r['tilt_max_deg'],reverse=True)[:5]))
