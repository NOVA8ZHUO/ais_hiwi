from pathlib import Path
import json,hashlib,re,torch,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R=Path('/home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance');out=R/'training_analysis';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];audit={}
for g in ['P0G0','P1G0','P0G1','P1G1']:
 for seed in range(3):
  run=R/'training'/g/f'seed_{seed}';done=json.loads((run/'done.json').read_text());assert done['manifest_sha256']==sha(R/'training_manifest.json')
  states=[]
  for p,h in done['checkpoints'].items():
   assert sha(p)==h;d=torch.load(p,map_location='cpu',weights_only=False);assert d['epoch']==200;states.append(d)
  assert len(states)==2
  assert states[0]['model'].keys()==states[1]['model'].keys()
  assert all(torch.equal(v,states[1]['model'][k]) for k,v in states[0]['model'].items())
  chosen=[p for p in done['checkpoints'] if '_rew__' in p];assert len(chosen)==1
  audit[f'{g}_seed{seed}']={'checkpoint':chosen[0],'checkpoint_sha256':sha(chosen[0]),'both_epoch':200,'policy_and_normalization_tensors_equal':True,'all_checkpoints':done['checkpoints']}
  event=EventAccumulator(str(next(run.rglob('events.out*'))),size_guidance={'scalars':0});event.Reload()
  ss=event.Scalars('successes/iter');rw=event.Scalars('rewards/iter')
  nonfinite=sum(not np.isfinite(x.value) for tag in event.Tags()['scalars'] for x in event.Scalars(tag))
  assert nonfinite==0
  values=np.array([x.value for x in ss]);rolling=np.convolve(values,np.ones(10)/10,mode='valid');ix=np.flatnonzero(rolling>=.95)
  log=(run/'console.log').read_text(errors='replace');seconds=float(re.findall(r'Training time: ([0-9.]+)',log)[-1])
  row=dict(group=g,seed=seed,last20_success_mean=float(values[-20:].mean()),last_success=float(values[-1]),last20_success_min=float(values[-20:].min()),last20_reward_mean=float(np.mean([x.value for x in rw[-20:]])),first_10record_95_epoch=ss[int(ix[0])+9].step if len(ix) else None,training_seconds=seconds,last_success_step=ss[-1].step,last_reward_step=rw[-1].step,nonfinite_scalars=nonfinite)
  row['success_curve']=[[x.step,x.value] for x in ss];row['reward_curve']=[[x.step,x.value] for x in rw];rows.append(row)
(out/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2));(out/'training_statistics.json').write_text(json.dumps(rows,indent=2))
for g in ['P0G0','P1G0','P0G1','P1G1']:
 a=[x for x in rows if x['group']==g];print(g,[(round(x['last20_success_mean']*100,2),x['first_10record_95_epoch'],round(x['last20_reward_mean'],2)) for x in a], 'mean',np.mean([x['last20_success_mean'] for x in a])*100)
print('hours',sum(x['training_seconds'] for x in rows)/3600)
