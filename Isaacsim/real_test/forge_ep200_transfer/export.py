"""Run in the source env_jingyun environment. Does not control hardware."""
from pathlib import Path
import hashlib,json,shutil,sys,importlib.metadata
import torch,yaml
from rl_games.algos_torch import model_builder
from portable_actor import PortableActor
ROOT=Path('/home/masais/Xinyu/Isaacsim')
OUT=Path(__file__).resolve().parent
STUDY=ROOT/'forge_insertion/disturbance_study'
lock=json.loads((STUDY/'lock.json').read_text())
torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta={'models':{},'hardware_tested':False,'versions':{k:importlib.metadata.version(k) for k in ['torch','rl-games','numpy','PyYAML']}}
for name,entry in lock['models'].items():
    src=Path(entry['checkpoint']);assert sha(src)==entry['checkpoint_sha256']
    dest=OUT/name;dest.mkdir(exist_ok=True)
    shutil.copy2(src,dest/'checkpoint_ep200.pth')
    # Original training configs, not a selectively randomized evaluation config.
    params=src.parent.parent/'params'
    if not params.exists():raise RuntimeError('Missing training config '+str(params))
    shutil.copytree(params,dest/'training_params',dirs_exist_ok=True)
    cfg=yaml.safe_load((params/'agent.yaml').read_text())
    sd=torch.load(src,map_location='cpu',weights_only=False)
    assert sd['epoch']==200
    model=model_builder.ModelBuilder().load(cfg['params']).build(dict(actions_num=7,input_shape=(24,),num_seqs=1,value_size=1,normalize_value=True,normalize_input=True)).eval()
    model.load_state_dict(sd['model'],strict=True)
    actor=PortableActor(model).eval();ts=torch.jit.script(actor);ts.save(str(dest/'policy_actor.ts'))
    maxerr=0.;vectors=[]
    with torch.inference_mode():
        for batch in [1,4]:
            torch.manual_seed(701+batch)
            h=torch.zeros(2,batch,1024);c=h.clone();rh=h.clone();rc=c.clone()
            for step in range(80):
                reset=torch.tensor([(step+i)%19==0 for i in range(batch)])
                h[:,reset]=0;c[:,reset]=0;rh[:,reset]=0;rc[:,reset]=0
                obs=actor.mean+torch.sqrt(actor.variance+actor.epsilon)*torch.randn(batch,24)
                if step%17==0:obs=obs+10*torch.sqrt(actor.variance+actor.epsilon)
                eps=torch.randn(batch,7)
                ref=model(dict(is_train=False,obs=obs.clone(),rnn_states=(rh,rc),seq_length=1))
                action,mu,sigma,h,c=ts(obs,h,c,eps)
                rh,rc=ref['rnn_states']
                expected=(ref['mus']+ref['sigmas']*eps).clamp(-1,1)
                for actual,target in [(mu,ref['mus']),(sigma,ref['sigmas']),(h,rh),(c,rc),(action,expected)]:
                    torch.testing.assert_close(actual,target,atol=2e-5,rtol=2e-5)
                    maxerr=max(maxerr,(actual-target).abs().max().item())
                if batch==1:vectors.append(dict(observation=obs.clone(),epsilon=eps.clone(),reset=reset.clone(),action=expected.clone(),mu=ref['mus'].clone(),sigma=ref['sigmas'].clone(),h=rh.clone(),c=rc.clone()))
    torch.save(vectors,dest/'reference_vectors.pt')
    info={**entry,'export_checkpoint':'checkpoint_ep200.pth','epoch':sd['epoch'],'max_abs_error':maxerr,'steps_per_batch':80,'batches':[1,4],'partial_resets_tested':True,'normalization_clipping_tested':True,'hardware_tested':False}
    (dest/'validation.json').write_text(json.dumps(info,indent=2));meta['models'][name]=info
snapshot=OUT/'source_reference';snapshot.mkdir(exist_ok=True)
lab=ROOT/'IsaacLab_2_3_2/source/isaaclab_tasks/isaaclab_tasks/direct'
for directory in ['forge','factory']:
    shutil.copytree(lab/directory,snapshot/directory,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for directory,target in [(ROOT/'forge_insertion/forge_cad','forge_cad'),(ROOT/'forge_insertion/0randomization/forge_zero','forge_zero')]:
    shutil.copytree(directory,snapshot/target,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['EXPERIMENT_RATIONALE.md','MODEL_COMPARISON.md','FINAL_ANALYSIS.md']:
    shutil.copy2(STUDY/name,OUT/name)
(OUT/'manifest.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
