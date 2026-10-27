from pathlib import Path
import shutil,json,sys,hashlib
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCriticRecurrent
root=Path('/home/masais/Xinyu/Isaacsim')
old=root/'real_test/model2999_real_transfer'
out=root/'real_test/model8000_real_transfer'
out.mkdir(exist_ok=False)
run=root/'logs/rsl_rl/forge_reward_v2_scratch/2026-10-01_14-19-06_reward_v2_total10000_20261001_141859_898ead'
for name in ['runtime','params','source_snapshot']:
 shutil.copytree((old if name=='runtime' else run)/name,out/name,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['verify_package.py','requirements-inference.txt','versions.json','source_environment_pip_freeze.txt']:
 shutil.copy2(old/name,out/name)
shutil.copy2(run/'experiment_manifest.json',out/'experiment_manifest.json')
shutil.copy2(run/'model_8000.pt',out/'model_8000.pt')
sys.path.insert(0,str(out/'runtime'))
from policy import PortableActor
sd=torch.load(out/'model_8000.pt',map_location='cpu',weights_only=False)['model_state_dict']
m=PortableActor().eval()
m.load_state_dict({('rnn.'+k.removeprefix('memory_a.rnn.')):v for k,v in sd.items() if k.startswith('memory_a.rnn.')} | {k:v for k,v in sd.items() if k.startswith('actor.')} | {'mean':sd['actor_obs_normalizer._mean'],'std':sd['actor_obs_normalizer._std']})
export=torch.jit.script(m);export.save(str(out/'policy_actor.ts'))
maxerr=0.;torch.manual_seed(8000)
with torch.inference_mode():
 for batch in [1,4]:
  dummy=TensorDict({'policy':torch.zeros(batch,29),'critic':torch.zeros(batch,53)},batch_size=[batch])
  ref=ActorCriticRecurrent(dummy,{'policy':['policy'],'critic':['critic']},5,actor_obs_normalization=True,critic_obs_normalization=True,actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='elu',rnn_type='lstm',rnn_hidden_dim=128,rnn_num_layers=1).eval()
  ref.load_state_dict(sd)
  h=torch.zeros(1,batch,128);c=h.clone()
  vectors={'observations':[],'raw_actions':[],'reset':[]}
  for step in range(200):
   done=torch.tensor([(step+i)%37==0 for i in range(batch)])
   ref.reset(done);h[:,done]=0;c[:,done]=0
   obs=m.mean+(m.std+.01)*torch.randn(batch,29)
   expected=ref.act_inference(TensorDict({'policy':obs,'critic':torch.zeros(batch,53)},batch_size=[batch]))
   raw,act,h,c=export(obs,h,c)
   torch.testing.assert_close(raw,expected,atol=1e-6,rtol=1e-6)
   torch.testing.assert_close(act,expected.clamp(-1,1),atol=1e-6,rtol=1e-6)
   maxerr=max(maxerr,float((raw-expected).abs().max()))
   if batch==1:
    vectors['observations'].append(obs.clone());vectors['raw_actions'].append(expected.clone());vectors['reset'].append(bool(done[0]))
  if batch==1:torch.save(vectors,out/'reference_test_vectors.pt')
(out/'inference_validation.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256((out/'model_8000.pt').read_bytes()).hexdigest(),torch_version=torch.__version__,reference='rsl_rl ActorCriticRecurrent.act_inference loaded original model8000',sequence_steps_per_batch=200,batches=[1,4],partial_resets=True,max_abs_error=maxerr,hardware_tested=False),indent=2)+'\n')
reports=out/'evaluation_reference';reports.mkdir()
for name,src in [('recheck',root/'FORGE_shaft_Insert/outputs/recheck_8000_9000/20261002_145916'),('gui_audit',root/'FORGE_shaft_Insert/outputs/model8000_gui_audit'),('duration_extension',root/'FORGE_shaft_Insert/outputs/model8000_duration_extension/20261002_163920')]:
 dest=reports/name;dest.mkdir()
 for f in src.iterdir():
  if f.is_file() and f.suffix in ['.md','.json','.csv']:
   shutil.copy2(f,dest/f.name)
for name in ['FORGE_ALIGNMENT.md','EXPERIMENT_CHANGELOG.md']:
 (out/'background').mkdir(exist_ok=True);shutil.copy2(root/'FORGE_shaft_Insert/docs'/name,out/'background'/name)
shutil.copy2(__file__,out/'build_export_reference.py')
print('EXPORTED',out,'maxerr',maxerr)
