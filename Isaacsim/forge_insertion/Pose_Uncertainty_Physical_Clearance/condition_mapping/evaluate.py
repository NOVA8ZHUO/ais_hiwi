"""Resumable serial evaluator with immutable model/source guards and per-job deadlines."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from samples import generate
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'disturbance_study'
PROJECT=ROOT.parent
LAB=PROJECT.parent/'IsaacLab_2_3_2'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_lock():
    lock=json.loads((ROOT/'lock.json').read_text())
    for p,h in lock['sha256'].items():
        if digest(p)!=h:raise RuntimeError('Source/checkpoint changed; do not mix experiments: '+p)
    return lock

def run_case(condition,active,seed,n=128,timeout=900,area=None,model='zero'):
    lock=verify_lock()
    selected=lock['models'][model]
    area=area or str(HERE/('results' if model=='zero' else 'results_randomized'))
    samples=HERE/'samples'/f'n{n}_seed{seed}.npz';generate(samples,seed,n)
    run=ROOT/area/condition/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
    identity={'condition':condition,'active':sorted(active),'seed':seed,'num_envs':n,'model':model,'checkpoint_sha256':selected['checkpoint_sha256'],
              'lock_sha256':digest(ROOT/'lock.json'),'sample_sha256':digest(samples)}
    path=run/'evaluation.json'
    if path.exists() and (run/'done.json').exists():
        done=json.loads((run/'done.json').read_text())
        assert done['identity']==identity
        assert done['evaluation_sha256']==digest(path),'Cached evaluation changed: '+str(path)
        return json.loads(path.read_text())
    if path.exists():path.unlink()
    failure=run/'failure.txt'
    if failure.exists():failure.unlink()
    cmd=[sys.executable,str(HERE/'rollout_grid.py'),'--task','Isaac-Forge-CAD-ClearanceMapping-Direct-v0',
         '--seed',str(seed),'--num_envs',str(n),'--episodes',str(n),'--checkpoint',selected['checkpoint'],'--headless']
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    env['PYTHONPATH']=os.pathsep.join(map(str,[HERE,ROOT,PROJECT]+[LAB/'source'/x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]))
    env['FORGE_ACTIVE_FACTORS']=json.dumps(sorted(active));env['FORGE_SAMPLE_FILE']=str(samples)
    (run/'command.json').write_text(json.dumps(cmd,indent=2))
    start=time.monotonic();print(f'START {model} {condition} seed={seed} n={n}',flush=True)
    with (run/'console.log').open('w') as log:
        child=subprocess.Popen(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            import signal
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if code!=0 or not path.exists() or failure.exists():raise RuntimeError(f'Failed job {run}; see console.log / failure.txt')
    data=json.loads(path.read_text())
    assert data['episodes']==n and data['sample_sha256']==identity['sample_sha256']
    assert sorted(data['active_factors'])==sorted(active)
    assert sorted(r['env_id'] for r in data['rows'])==list(range(n))
    assert data['policy_sample_count']==data['rows'][0]['steps']
    (run/'done.json').write_text(json.dumps({'identity':identity,'seconds':time.monotonic()-start,'evaluation_sha256':digest(path)},indent=2))
    print(f'DONE {model} {condition} seed={seed}: {data["success_rate_final"]:.2%}',flush=True)
    return data


def require_preflight():
    d=json.loads((ROOT/'preflight_passed.json').read_text())
    assert d['status']=='passed'
    for name,expected in d['validated_sha256'].items():
        assert digest(name)==expected,'Re-run preflight for changed runtime file: '+name

    paired=json.loads((ROOT/'two_model_preflight_passed.json').read_text())
    assert paired['status']=='passed','Two-model preflight missing'
    lock=json.loads((ROOT/'lock.json').read_text())
    assert paired['checkpoint_sha256']=={m:d['checkpoint_sha256'] for m,d in lock['models'].items()},'Model checkpoints differ from validated pair'
