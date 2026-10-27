import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('FORGE_shaft_Insert/outputs/model8000_gui_audit');d=json.loads((p/'historical_episode_audit.json').read_text());n=next(r for r in d if r['seed']==42 and not r['success'] and r['final_depth_mm']<2);keys=[(n['env'],n['episode']),(24,1),(9,4)];traces={k:[] for k in keys};counts={}
f=Path('FORGE_shaft_Insert/outputs/reward_v2_total10000/reward_v2_total10000_20261001_141859_898ead/eval_model8000_seed42/episode_metrics_control_trace.csv')
for r in csv.DictReader(f.open()):
 env=int(r['env_id']);ep=counts.get(env,0)+1
 if (env,ep) in traces:traces[(env,ep)].append(r)
 if any(float(r[x+'_0']) for x in ['time_out','outside_workspace','invalid_grasp_width','release_complete']):counts[env]=ep
fig,axs=plt.subplots(4,3,figsize=(14,10),sharex='col')
for col,(key,rs) in enumerate(traces.items()):
 a=lambda k:np.array([float(r[k]) for r in rs]);t=a('elapsed_s_0')
 axs[0,col].plot(t,-a('shaft_error_2')*1000);axs[0,col].axhline(25,ls='--',color='g');axs[0,col].set_title(f'seed42 env{key[0]} episode{key[1]}')
 axs[1,col].plot(t,np.hypot(a('contact_force_0'),a('contact_force_1')),label='XY');axs[1,col].plot(t,abs(a('contact_force_2')),label='abs Z');axs[1,col].legend()
 axs[2,col].plot(t,np.degrees(np.arccos(np.clip(a('shaft_axis_world_2'),-1,1))))
 axs[3,col].step(t,a('stage_0'));axs[3,col].set_yticks([0,1,2,3],['SETTLE','SEARCH','PROBE','INSERT']);axs[3,col].set_xlabel('Time (s)')
for i,label in enumerate(['Depth mm','Contact N','Tilt deg','Stage']):axs[i,0].set_ylabel(label)
for ax in axs.flat:ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(p/'failure_traces.png',dpi=150)
