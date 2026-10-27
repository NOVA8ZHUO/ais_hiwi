import csv,json,sys,math,hashlib
from pathlib import Path
import numpy as np
P=Path('/home/masais/Xinyu/Isaacsim/FORGE_shaft_Insert')
base=P/'outputs/coupled1000/coupled1000_20260929_154037_d798fa'
out=P/'outputs/model820_plus10000/model820_plus10000_20260929_175136_62085c'
folders={'820':base/'eval_model_820_seed42','4900':out/'eval_model_4900_seed42'}
results={}; detail=[]
for model,folder in folders.items():
 summary=json.loads((folder/'evaluation_summary.json').read_text());manifest=json.loads((Path(summary['checkpoint']).parent/'experiment_manifest.json').read_text())
 for name in ['forge_shaft_insert/mdp/forge_paper.py','forge_shaft_insert/mdp/insertion_supervisor.py','forge_shaft_insert/metrics_env.py']:
  assert hashlib.sha256((P/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
 episodes={(int(r['env_id']),int(r['eval_episode_for_env'])):r for r in csv.DictReader((folder/'evaluation_episodes.csv').open())}
 print('keys',model,list(episodes)[:3])
 count={}; traces={}
 for r in csv.DictReader((folder/'episode_metrics_control_trace.csv').open()):
  env=int(r['env_id']);ep=count.get(env,0)+1
  if ep<=4:traces.setdefault((env,ep),[]).append(r)
  if any(float(r[k+'_0']) for k in ['time_out','outside_workspace','invalid_grasp_width','release_complete']):count[env]=ep
 rows=[]
 for key,trace in traces.items():
  e=episodes[key]
  def ar(name):return np.array([float(r[name]) for r in trace])
  t=ar('elapsed_s_0');d=-ar('shaft_error_2')*1000;xy=np.hypot(ar('shaft_error_0'),ar('shaft_error_1'))*1000
  assert abs(d[-1]-float(e['final_depth_mm']))<.02
  mode=ar('stage_0').astype(int);active=(ar('motion_enabled_before_action_0')>0)&(ar('releasing_0')==0)
  entry=np.flatnonzero((d>=2)&(xy<=13)&active)
  group='success' if int(e['success_final']) else ('shallow_fail' if 2<=float(e['final_depth_mm'])<25 else 'other_fail')
  row=dict(model=model,env=key[0],episode=key[1],group=group,final_depth=float(e['final_depth_mm']),final_xy=float(e['final_xy_mm']),entry_s=None)
  if len(entry):
   start=entry[0]; mask=active & (np.arange(len(t))>=start);ids=np.flatnonzero(mask)
   # first passage >=2 mm with XY inside target bore region; not proof of full engagement.
   row['entry_s']=float(t[start]);row['remaining_s']=float(t[-1]-t[start]);row['post_s']=len(ids)/15
   # Longest interval without setting a new running-best depth by >=1 mm.
   anchor=d[start];anchor_t=t[start];longest=0
   for i in ids:
    if d[i]>=anchor+1:
     longest=max(longest,t[i]-anchor_t);anchor=d[i];anchor_t=t[i]
   longest=max(longest,t[ids[-1]]-anchor_t)
   row['stall_record_1mm_s']=float(longest)
   row['post_progress_mm']=float(max(d[ids])-d[start]);row['peak_depth_mm']=float(max(d[ids]))
   row['retreat_from_peak_mm']=float(max(d[ids])-d[ids[-1]])
   row['switches']=int(np.count_nonzero(np.diff(mode[ids])))
   for st,n in [(1,'search'),(2,'probe'),(3,'insert')]:row[n+'_fraction']=float(np.mean(mode[ids]==st))
   row['ever_insert']=bool(np.any(mode[ids]==3));row['end_stage']=int(mode[ids[-1]])
   row['probe_to_search']=int(sum(mode[i-1]==2 and mode[i]==1 for i in ids if i>start))
   fx=ar('contact_force_0');fy=ar('contact_force_1');fz=ar('contact_force_2');lat=np.hypot(fx,fy)
   search_ids=ids[mode[ids]==1]
   if len(search_ids):
    estimated_depth=-(ar('tcp_error_2')-ar('hole_bias_2'))*1000
    row['search_lateral_ge2_fraction']=float(np.mean(lat[search_ids]>=2))
    row['search_fz_ge1_fraction']=float(np.mean(fz[search_ids]>=1))
    row['search_est_depth_gt16_fraction']=float(np.mean(estimated_depth[search_ids]>16))
    eligible=search_ids[estimated_depth[search_ids]>16]
    row['search_deep_samples']=len(eligible)
    row['search_deep_lateral_ge2_samples']=int(np.sum(lat[eligible]>=2))
   else:
    for k in ['search_lateral_ge2_fraction','search_fz_ge1_fraction','search_est_depth_gt16_fraction','search_deep_samples','search_deep_lateral_ge2_samples']:row[k]=0
   row['command_xy_mean']=float(np.mean(np.hypot(ar('command_force_0')[ids],ar('command_force_1')[ids])))
   row['xy_action_abs_mean']=float(np.mean(np.abs(np.stack([ar('action_0')[ids],ar('action_1')[ids]]))))
   for n,v in [('contact_xy',lat),('contact_z_abs',abs(fz)),('command_down',np.maximum(-ar('command_force_2'),0)),('press',ar('press_force_0')),('tip_tcp_xy_mm',np.hypot(ar('shaft_error_0')-ar('tcp_error_0'),ar('shaft_error_1')-ar('tcp_error_1'))*1000)]:
    row[n+'_mean']=float(np.mean(v[ids]));row[n+'_p95']=float(np.quantile(v[ids],.95))
   yaw=ar('action_3');row['yaw_abs_mean']=float(np.mean(abs(yaw[ids])));row['yaw_sat']=float(np.mean(abs(yaw[ids])>=.99));row['z_action_mean']=float(np.mean(ar('action_2')[ids]))
   q=[ar('hand_quat_'+str(j)) for j in range(4)];w,x,y,z=q
   angle=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
   row['hand_yaw_path_deg']=float(np.sum(abs(np.diff(angle[ids])))*180/np.pi)
   row['hand_yaw_range_deg']=float(np.ptp(angle[ids])*180/np.pi)
   row['tilt_mean_deg']=float(np.mean(np.degrees(np.arccos(np.clip(ar('shaft_axis_world_2')[ids],-1,1)))))
   row['end_depth_gain_last2s_mm']=float(d[ids[-1]]-d[ids[np.argmin(abs(t[ids]-(t[ids[-1]]-2)))]] )
   row['hole_bias_z_mm']=float(ar('hole_bias_2')[start]*1000)
  rows.append(row);detail.append(row)
 groups={}
 for g in ['success','shallow_fail','other_fail']:
  rs=[r for r in rows if r['group']==g];entered=[r for r in rs if r['entry_s'] is not None]
  agg=dict(n=len(rs),entered=len(entered),early_by10=sum(r['entry_s']<=10 for r in entered),late_after15=sum(r['entry_s']>15 for r in entered),stall_ge5=sum(r['stall_record_1mm_s']>=5 for r in entered),ever_insert=sum(r['ever_insert'] for r in entered),end_stages={str(k):sum(r['end_stage']==k for r in entered) for k in [1,2,3]})
  if entered:
   for k,v in entered[0].items():
    if k not in ['model','env','episode','group'] and isinstance(v,(float,int)):
     vals=[r[k] for r in entered];agg[k]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)),min=float(min(vals)),max=float(max(vals)))
  groups[g]=agg
 results[model]=groups
(out/'SHALLOW_TRAJECTORY_AUDIT.json').write_text(json.dumps(results,indent=2)+'\n')
keys=list(dict.fromkeys(k for r in detail for k in r))
with (out/'shallow_trajectory_episodes.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(detail)
print('Wrote audit files')
