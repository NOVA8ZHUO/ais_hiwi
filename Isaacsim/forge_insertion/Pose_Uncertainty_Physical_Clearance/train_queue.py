"""Explicitly user-launched queue. Dry run is the default; never auto-launch on import."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;LAB=PROJECT.parent/'IsaacLab_2_3_2'
GROUPS=['P0G0','P1G0','P0G1','P1G1'];SEEDS=[0,1,2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(**data):
 path=ROOT/'training_status.json'
 d=json.loads(path.read_text()) if path.exists() else {}
 d.update(data,updated_unix=time.time(),pid=os.getpid())
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,PROJECT]+[LAB/'source'/x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]))
 jobs=[]
 for seed in SEEDS:
  for group in GROUPS:
   cmd=[sys.executable,str(ROOT/'train_entry.py'),str(LAB/'scripts/reinforcement_learning/rl_games/train.py'),'--task',f'Isaac-Forge-CAD-PG-{group}-Direct-v0','--num_envs','128','--max_iterations','200','--seed',str(seed),'--headless']
   jobs.append(dict(group=group,seed=seed,command=cmd))
 if not a.execute:
  print(json.dumps({'mode':'DRY RUN; no simulator or training launched','runs':len(jobs),'epochs_per_run':200,'envs':128,'jobs':jobs},indent=2));return
 frozen=json.loads((ROOT/'training_manifest.json').read_text())
 for file,h in frozen['sha256'].items():assert digest(Path(file))==h,'Changed training source/asset: '+file
 outputs=ROOT/'training';outputs.mkdir(exist_ok=True)
 with (outputs/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status(state='preflight',completed_runs=0,total_runs=12,error=None)
  def call(cmd,path):
   path.mkdir(parents=True,exist_ok=True);(path/'command.json').write_text(json.dumps(cmd,indent=2))
   with (path/'console.log').open('w') as log:subprocess.run(cmd,cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=86400 if 'seed_' in path.name else 900)
  # Always run the four-group mixed-asset integration gate before training.
  for group in GROUPS:
   status(state='preflight',group=group,seed=None)
   print('PREFLIGHT',group,flush=True)
   path=outputs/'preflight'/group
   call([sys.executable,str(ROOT/'check_mixed.py'),'--group',group,'--headless'],path)
   assert json.loads((path/'mixed_preflight.json').read_text())['status']=='passed'
  if a.preflight_only:
   status(state='preflight_complete');return
  for i,j in enumerate(jobs):
   path=outputs/j['group']/f"seed_{j['seed']}"
   if (path/'done.json').exists():
    done=json.loads((path/'done.json').read_text())
    assert done['manifest_sha256']==digest(ROOT/'training_manifest.json')
    for file,h in done['checkpoints'].items():assert digest(Path(file))==h
    print('SKIP completed',j['group'],j['seed'],flush=True);continue
   # Never silently overwrite partial training; fresh retry requires preserving it first.
   if path.exists():raise RuntimeError('Partial run exists; preserve/rename before retry: '+str(path))
   status(state='training',group=j['group'],seed=j['seed'],completed_runs=i,current_run_started_unix=time.time(),console=str(path/'console.log'))
   print(f"TRAIN {i+1}/{len(jobs)} {j['group']} seed={j['seed']}",flush=True)
   call(j['command'],path)
   ckpts=list(path.glob('**/nn/*ep_200*.pth'));assert ckpts,'Missing final epoch-200 checkpoint'
   (path/'done.json').write_text(json.dumps({'manifest_sha256':digest(ROOT/'training_manifest.json'),'checkpoints':{str(x):digest(x) for x in ckpts}},indent=2))
  status(state='complete',completed_runs=12)
  print('COMPLETE: 12 training runs',flush=True)
if __name__=='__main__':
 try:main()
 except BlockingIOError:raise SystemExit('Training queue already running')
 except BaseException as exc:
  if '--execute' in sys.argv:status(state='failed',error=repr(exc))
  raise
